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

    try:
        from jax.experimental.mosaic.gpu.profiler import _event_elapsed, _event_record
    except ImportError:
        try:
            from cuequivariance_ops_jax import event_elapsed as _event_elapsed
            from cuequivariance_ops_jax import event_record as _event_record
        except ImportError:
            raise ImportError(
                "_event_elapsed and/or _event_record not found in jax.experimental.mosaic.gpu.profiler\n"
                "They are known to be available in jax>=0.4.36,<=0.7.1"
            )

    def run_func(state):
        """Wrapper function that calls the target function and ensures proper data flow."""
        args, kwargs = state
        outputs = f(*args, **kwargs)
        # Use noop to prevent compiler optimizations while maintaining data dependencies
        args, kwargs, _ = noop((args, kwargs, outputs))
        return (args, kwargs)

    tick_time: float = 10e-6

    @partial(jax.jit, static_argnums=(0, 1))
    def run_bench(n_warm: int, n_iter: int, fill_time: float, state):
        # First sleep: fill CUDA stream
        _, state = sleep(fill_time, state)

        # Warmup phase: execute multiple times to stabilize GPU clocks and trigger JIT compilation
        _, state = _event_record(state, copy_before=True)
        for _ in range(n_warm):
            state = run_func(state)
        _, state = _event_record(state, copy_before=False)

        # Second sleep: measure clock ticks
        ticks_before, state = sleep(tick_time, state)

        # Main measurement phase
        start_event, state = _event_record(state, copy_before=True)
        for _ in range(n_iter):
            state = run_func(state)
        end_event, state = _event_record(state, copy_before=False)

        # Third sleep: measure clock ticks
        ticks_after, state = sleep(tick_time, state)

        # Synchronize to check if the CPU lags behind the GPU or not
        sync_time, state = synchronize(state)

        # Call noop to ensure instruction order is preserved
        start_event, end_event, ticks_before, ticks_after, state = noop(
            (start_event, end_event, ticks_before, ticks_after, state)
        )
        total_time = 1e-3 * _event_elapsed(start_event, end_event)
        avg_time = total_time / n_iter

        rate_before = ticks_before / tick_time
        rate_after = ticks_after / tick_time

        return avg_time, rate_before, rate_after, sync_time

    # Adaptive iteration counting to find optimal measurement parameters
    n_warm = 2
    n_iter = 5
    fill_base = 50e-6  # Base time for stream filling
    rejections: list[str] = []
    best_measurement = None
    best_variation = float("inf")

    for attempt in range(20):
        fill_time = fill_base + 10e-6 * (n_warm + n_iter)
        avg_time, rate_before, rate_after, sync_time = jax.tree.map(
            float, run_bench(n_warm, n_iter, fill_time, (args, kwargs))
        )
        avg_rate = (rate_before + rate_after) / 2

        if best_measurement is None:
            best_measurement = (avg_rate, avg_time)

        # print(
        #     f"DEBUG: Attempt {attempt + 1}, n_iter={n_iter}, n_warm={n_warm}, "
        #     f"fill_time={fill_time * 1e6:.1f}us, "
        #     f"sync_time={sync_time * 1e6:.1f}us, "
        #     f"avg_time={avg_time * 1e6:.1f}us, "
        #     f"rate_before={rate_before / 1e9:.2f} GHz, "
        #     f"rate_after={rate_after / 1e9:.2f} GHz"
        # )

        # If synchronization time is small, it indicates the CPU is lagging behind the GPU
        target_sync_time = tick_time + 20e-6
        if sync_time < target_sync_time:
            fill_base += (target_sync_time - sync_time) + 50e-6
            rejections.append(
                f"CPU lagging behind GPU (will sleep {fill_base * 1e3:.1f} ms)"
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
        diff, max_tol = (
            abs(rate_before - rate_after),
            0.02 * max(rate_before, rate_after),
        )

        # Track the best measurement (lowest clock rate variation) for fallback
        if diff < best_variation:
            best_variation = diff
            best_measurement = (avg_rate, avg_time)

        if diff > max_tol:
            rejections.append(
                f"Clock rate variation too high "
                f"({diff / 1e6:.2f} MHz variation > {max_tol / 1e6:.2f} MHz)"
            )
            if n_warm < n_iter:
                n_warm = n_iter
            else:
                n_warm += round(n_warm * 0.2) + 1
            continue

        return avg_rate, avg_time

    # If we get here, no measurement met all criteria
    # Return the best measurement (lowest clock rate variation) we found
    rejection_details = "\n".join(
        f"  Attempt #{i + 1}: {reason}" for i, reason in enumerate(rejections)
    )
    warnings.warn(
        f"Was not able to reach a satisfying measurement in {len(rejections)} attempts. "
        f"Returning measurement with lowest clock rate variation ({best_variation / 1e6:.2f} MHz). "
        f"Rejection reasons:\n{rejection_details}"
    )
    return best_measurement
