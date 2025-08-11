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

import time
import warnings

import torch


def measure_clock_ticks(f, *args, **kwargs) -> tuple[float, float]:
    """Measure the execution time of a function in clock ticks.

    Args:
        f: The function to benchmark
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        tuple: A tuple containing the average clock rate in Hz and the average time per function call in seconds.

        Example:
        def my_function(x, y):
            return x + y

        rate, time = measure_clock_ticks(my_function, x, y)
        clock_ticks = rate * time
        print(f"Function took {clock_ticks} clock ticks")
    """
    from cuequivariance_ops_torch import sleep

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    x = torch.tensor(0, device="cuda")

    fill_base = 200e-6  # Base time for stream filling
    n_warm = 0
    n_iter = 5
    rejections: list[str] = []
    best_measurement = None
    best_variation = float("inf")

    for attempt in range(20):
        fill_time = torch.tensor(30e-6 * (n_warm + n_iter) + fill_base, device="cuda")
        tick_time = torch.tensor(30e-6, device="cuda")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # First sleep: fill CUDA stream
        _, x = sleep(fill_time, x)

        for _ in range(n_warm):
            _ = f(*args, **kwargs)

        # Second sleep: measure clock ticks
        ticks_before, x = sleep(tick_time, x)

        start_event.record()
        for _ in range(n_iter):
            _ = f(*args, **kwargs)
        end_event.record()

        # Third sleep: measure clock ticks
        ticks_after, x = sleep(tick_time, x)

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        sync_time = time.perf_counter() - t0

        total_time: float = start_event.elapsed_time(end_event) * 1e-3
        avg_time = total_time / n_iter

        rate_before = ticks_before.item() / tick_time.item()
        rate_after = ticks_after.item() / tick_time.item()
        avg_rate = (rate_before + rate_after) / 2

        # print(
        #     f"DEBUG: Attempt {attempt + 1}, n_iter={n_iter}, n_warm={n_warm}, "
        #     f"total_sleep={sleep_before_time.item() * 1e6:.1f}us, "
        #     f"sync_time={sync_time * 1e6:.1f}us, "
        #     f"avg_time={avg_time * 1e6:.1f}us, "
        #     f"rate_before={rate_before / 1e9:.2f} GHz, "
        #     f"rate_after={rate_after / 1e9:.2f} GHz"
        # )

        if attempt == 0:
            # Always skip the first iteration to allow for JIT compilation
            diff = abs(rate_before - rate_after)
            best_variation = diff
            best_measurement = (avg_rate, avg_time)
            rejections.append("First iteration (always skipped)")
            continue

        if sync_time < 50e-6:
            # If synchronization is too fast, it may indicate that the CPU is lagging behind the GPU.
            fill_base += 50e-6 - sync_time + 500e-6
            rejections.append(
                f"CPU lagging behind GPU (will sleep {fill_base * 1e3:.1f} ms)"
            )
            continue

        if n_iter * avg_time < 20e-6:
            # Avoid measurement overheads by measuring for at least 20 microseconds
            n_iter = int(100e-6 / avg_time)
            rejections.append(
                f"Too short measurement time (will measure {n_iter} iterations)"
            )
            continue

        diff, max_tol = (
            abs(rate_before - rate_after),
            0.02 * max(rate_before, rate_after),
        )

        # Track the best measurement (lowest clock rate variation) for fallback
        if diff < best_variation:
            best_variation = diff
            best_measurement = (avg_rate, avg_time)

        if diff > max_tol:
            # If the clock rate varies too much, increase warmup and retry
            rejections.append(
                f"Clock rate variation too high "
                f"({diff / 1e6:.2f} MHz variation is bigger than {max_tol / 1e6:.2f} MHz)"
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
