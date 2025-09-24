# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from cuequivariance_torch.benchmarking import measure_clock_ticks


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
@pytest.mark.parametrize("size", [4, 64, 1024])
def test_benchmarking(size):
    x = torch.ones((size, 32), dtype=torch.float32, device="cuda")
    y = torch.ones((size, 32), dtype=torch.float32, device="cuda")

    def f(x, y):
        return x * y

    rate, time = measure_clock_ticks(f, x, y)
    assert rate > 0
    assert time > 0
