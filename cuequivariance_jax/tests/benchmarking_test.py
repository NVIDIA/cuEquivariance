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

import importlib.util

import jax.numpy as jnp
import pytest
from cuequivariance_jax.benchmarking import measure_clock_ticks


@pytest.mark.skipif(
    not importlib.util.find_spec("cuequivariance_ops_jax"),
    reason="cuequivariance_ops_jax is not installed",
)
@pytest.mark.parametrize("size", [4, 64, 1024])
def test_benchmarking(size):
    x = jnp.ones((size, 32), dtype=jnp.float32)
    y = jnp.ones((size, 32), dtype=jnp.float32)

    def f(x, y):
        return x * y

    rate, time = measure_clock_ticks(f, x, y)
    assert rate > 0
    assert time > 0
