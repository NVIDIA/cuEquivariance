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
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex


@cue.assume("SO3", cue.ir_mul)
def test_segments():
    x = cuex.RepArray("2x0 + 1", jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    x0, x1 = x.segments
    assert x0.shape == (1, 2)
    assert x1.shape == (3, 1)
    y = cuex.from_segments("2x0 + 1", [x0, x1], x.shape)
    assert x.irreps == y.irreps
    assert x.layout == y.layout
    assert jnp.allclose(x.array, y.array)


@cue.assume("SO3", cue.ir_mul)
def test_slice_by_mul():
    x = cuex.RepArray("2x0 + 1", jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    x = x.slice_by_mul[1:]
    assert x.irreps == cue.Irreps("0 + 1")
    assert x.layout == cue.ir_mul
    assert jnp.allclose(x.array, jnp.array([1.0, 0.0, 0.0, 0.0]))
