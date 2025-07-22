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
    """
    from cuequivariance_ops_torch import sleep

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    compiled_func = torch.compile(f)
    x = torch.tensor(0, device="cuda")

    t_sleep_before = 200e-6  # 200 microseconds
    n_iter = 1

    for attempt in range(10):
        sleep_before_time = torch.tensor(30e-6 * n_iter + t_sleep_before, device="cuda")
        sleep_after_time = torch.tensor(30e-6, device="cuda")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Sleep to (i) measure clock ticks and (ii) ensure the stream is kept full
        ticks_before, x = sleep(sleep_before_time, x)

        start_event.record()
        for _ in range(n_iter):
            _ = compiled_func(*args, **kwargs)
        end_event.record()

        ticks_after, x = sleep(sleep_after_time, x)

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        sync_time = time.perf_counter() - t0

        total_time: float = start_event.elapsed_time(end_event) * 1e-3
        avg_time = total_time / n_iter

        rate_before = ticks_before.item() / sleep_before_time.item()
        rate_after = ticks_after.item() / sleep_after_time.item()
        avg_clock_rate = (rate_before + rate_after) / 2

        if attempt == 0:
            # Always skip the first iteration to allow for JIT compilation
            continue

        if sync_time < 20e-6:
            # If synchronization is too fast, it may indicate that the CPU is lagging behind the GPU.
            t_sleep_before += 200e-6  # Add 200 microseconds
            continue

        if n_iter * avg_time < 20e-6:
            # Avoid measurement overheads by measuring for at least 20 microseconds
            n_iter = int(100e-6 / avg_time)
            continue

        if abs(rate_before - rate_after) > 0.01 * max(rate_before, rate_after):
            # If the clock rate varies too much, simply retry
            continue

        return avg_clock_rate, avg_time

    warnings.warn(f"Potentially bad measurement of clock ticks for {f.__name__}.")
    return avg_clock_rate, avg_time
