# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmarking utilities for measuring function execution time using clock ticks."""

import warnings
from functools import partial

import jax
from jax.experimental.mosaic.gpu.profiler import _event_elapsed, _event_record


def measure_clock_ticks(f, *args, **kwargs) -> tuple[float, float]:
    """Measure the execution time of a function in clock ticks.

    The measurement process:
    1. Performs warmup runs to account for potential JIT compilation or library loading
    2. Calibrates the clock tick rate using sleep operations
    3. Measures execution time over multiple iterations
    4. Converts GPU event time to clock ticks using the calibrated rate

    Args:
        f: The function to benchmark
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        tuple: A tuple containing the average clock rate in Hz and the average time per function call in seconds.

        Example:
        def my_function(x, y):
            return x + y

        rate, time = measure_clock_ticks(my_function, 1, 2)
        clock_ticks = rate * time
        print(f"Function took {clock_ticks} clock ticks")
    """
    from cuequivariance_ops_jax import noop, sleep, synchronize

    def run_func(state):
        """Wrapper function that calls the target function and ensures proper data flow."""
        args, kwargs = state
        outputs = f(*args, **kwargs)
        # Use noop to prevent compiler optimizations while maintaining data dependencies
        args, kwargs, _ = noop((args, kwargs, outputs))
        return (args, kwargs)

    second_sleep_time: float = 30e-6

    @partial(jax.jit, static_argnums=(0, 1))
    def run_bench(n_iter: int, sleep_time: float, state):
        # Warmup phase: execute once to trigger potential JIT compilation or library loading
        _, state = _event_record(state, copy_before=True)
        state = run_func(state)
        _, state = _event_record(state, copy_before=False)

        # Pre-measurement clock rate calibration
        # Sleep to fill the CUDA stream and measure clock rate
        # Assumption: each operation takes less than 10us
        sleep_time = sleep_time + 10e-6 * n_iter
        ticks_before, state = sleep(sleep_time, state)
        rate_before = ticks_before / sleep_time

        # Main measurement phase
        start_event, state = _event_record(state, copy_before=True)
        for _ in range(n_iter):
            state = run_func(state)
        end_event, state = _event_record(state, copy_before=False)

        # Post-measurement clock rate calibration
        # Use 30us as a reasonable time to measure clock ticks
        ticks_after, state = sleep(second_sleep_time, state)
        rate_after = ticks_after / second_sleep_time

        # Synchronize to check if the CPU lags behind the GPU or not
        sync_time, state = synchronize(state)

        # Calculate average time per function call using GPU events
        # Call noop to ensure sleep+sync is executed before event_elapsed
        end_event, state = noop((end_event, state))
        total_time = 1e-3 * _event_elapsed(start_event, end_event)
        avg_time = total_time / n_iter

        return avg_time, rate_before, rate_after, sync_time

    # Adaptive iteration counting to find optimal measurement parameters
    n_iter = 1
    sleep_time = 50e-6
    rejections: list[str] = []

    for attempt in range(10):
        avg_time, rate_before, rate_after, sync_time = jax.tree.map(
            float, run_bench(n_iter, sleep_time, (args, kwargs))
        )
        avg_rate = (rate_before + rate_after) / 2

        # print(
        #     f"DEBUG: Attempt {attempt + 1}, n_iter={n_iter}, "
        #     f"sleep_time={sleep_time * 1e6:.1f}us, "
        #     f"sync_time={sync_time * 1e6:.1f}us, "
        #     f"avg_time={avg_time * 1e6:.1f}us, "
        #     f"rate_before={rate_before / 1e9:.2f} GHz, "
        #     f"rate_after={rate_after / 1e9:.2f} GHz"
        # )

        # If synchronization time is small, it indicates the CPU is lagging behind the GPU
        target_sync_time = second_sleep_time + 20e-6
        if sync_time < target_sync_time:
            sleep_time += (target_sync_time - sync_time) + 50e-6
            rejections.append(
                f"CPU lagging behind GPU (will sleep {sleep_time * 1e3:.1f} ms)"
            )
            continue

        # Ensure measurement duration is long enough for accuracy (at least 20us total)
        min_time = 20e-6
        if n_iter * avg_time < min_time:
            # Increase iterations to reach minimum measurement time
            target = 100e-6  # Target 100us total measurement time
            n_iter = int(target / avg_time)
            rejections.append(
                f"Too short measurement time (will measure {n_iter} iterations)"
            )
            continue

        # Check if clock rates are consistent (within 1% tolerance)
        # Inconsistent rates indicate timing measurement issues
        tolerance = 0.01
        max_rate = max(rate_before, rate_after)
        if abs(rate_before - rate_after) > tolerance * max_rate:
            rejections.append(
                f"Clock rate variation too high "
                f"({abs(rate_before - rate_after) / 1e6:.2f} MHz variation)"
            )
            continue

        return avg_rate, avg_time

    rejection_details = "\n".join(
        f"  Attempt #{i + 1}: {reason}" for i, reason in enumerate(rejections)
    )
    warnings.warn(
        f"Was not able to reach a satisfying measurement in {len(rejections)} attempts. "
        f"Rejection reasons:\n{rejection_details}"
    )
    return avg_rate, avg_time
