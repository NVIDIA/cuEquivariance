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
import numpy as np
from jax.experimental.mosaic.gpu.profiler import _event_elapsed, _event_record


def measure_clock_ticks(f, *args, **kwargs) -> float:
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
        float: The execution time in clock ticks per function call

    Example:
        >>> def my_function(x, y):
        ...     return x + y
        >>> clock_ticks = measure_clock_ticks(my_function, 1, 2)
        >>> clock_ticks > 0  # Returns positive number of clock ticks
        True
    """
    from cuequivariance_ops_jax import noop, sleep

    def run_func(state):
        """Wrapper function that calls the target function and ensures proper data flow."""
        args, kwargs = state
        outputs = f(*args, **kwargs)
        # Use noop to prevent compiler optimizations while maintaining data dependencies
        args, kwargs, _ = noop((args, kwargs, outputs))
        return (args, kwargs)

    @partial(jax.jit, static_argnums=(0,))
    def run_bench(n_iter: int, state):
        # Warmup phase: execute once to trigger potential JIT compilation or library loading
        _, state = _event_record(state, copy_before=True)
        state = run_func(state)
        _, state = _event_record(state, copy_before=False)

        # Pre-measurement clock rate calibration
        # Sleep to fill the CUDA stream and measure clock rate
        # Assumption: each operation takes less than 10us, add 100us buffer
        sleep_time = 10e-6 * n_iter + 100e-6
        ticks_before, state = sleep(sleep_time, state)
        rate_before = ticks_before / sleep_time

        # Main measurement phase
        start_event, state = _event_record(state, copy_before=True)
        for _ in range(n_iter):
            state = run_func(state)
        end_event, state = _event_record(state, copy_before=False)

        # Post-measurement clock rate calibration
        # Use 30us as a reasonable time to measure clock ticks
        calib_time = 30e-6
        ticks_after, state = sleep(calib_time, state)
        rate_after = ticks_after / calib_time

        # Calculate average time per function call using GPU events
        # Call noop to ensure sleep is executed before event_elapsed
        end_event, state = noop((end_event, state))
        total_time = 1e-3 * _event_elapsed(start_event, end_event)
        avg_time = total_time / n_iter

        return avg_time, rate_before, rate_after

    # Adaptive iteration counting to find optimal measurement parameters
    success = False
    n_iter = 1
    max_tries = 5

    for attempt in range(max_tries):
        avg_time, rate_before, rate_after = jax.tree.map(
            np.array, run_bench(n_iter, (args, kwargs))
        )

        # Check if clock rates are consistent (within 1% tolerance)
        # Inconsistent rates indicate timing measurement issues
        tolerance = 0.01
        max_rate = max(rate_before, rate_after)
        if abs(rate_before - rate_after) > tolerance * max_rate:
            continue

        # Ensure measurement duration is long enough for accuracy (at least 20us total)
        min_time = 20e-6
        if n_iter * avg_time < min_time:
            # Increase iterations to reach minimum measurement time
            target = 100e-6  # Target 100us total measurement time
            n_iter = int(target / avg_time)
            continue

        success = True
        break

    if not success:
        warnings.warn(f"Potentially bad measurement of clock ticks for {f.__name__}.")

    # Convert seconds to clock ticks using average of before/after rates
    avg_rate = (rate_before + rate_after) / 2
    ticks_per_call = avg_time * avg_rate

    return ticks_per_call
