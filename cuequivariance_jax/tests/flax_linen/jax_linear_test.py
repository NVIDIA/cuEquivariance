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
import jax
import jax.numpy as jnp
import pytest

import cuequivariance as cue
import cuequivariance_jax as cuex


@pytest.mark.parametrize("layout_in", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("layout_out", [cue.ir_mul, cue.mul_ir])
def test_explicit_linear(layout_in, layout_out):
    try:
        import flax  # noqa
    except ImportError:
        pytest.skip("flax not installed")

    x = cuex.RepArray(cue.Irreps("SO3", "3x0 + 2x1"), jnp.ones((16, 9)), layout_in)
    linear = cuex.flax_linen.Linear(cue.Irreps("SO3", "2x0 + 1"), layout_out)
    w = linear.init(jax.random.key(0), x)
    y: cuex.RepArray = linear.apply(w, x)
    assert y.shape == (16, 5)
    assert y.irreps == cue.Irreps("SO3", "2x0 + 1")
    assert y.layout == layout_out


@cue.assume("SO3", cue.ir_mul)
def test_implicit_linear():
    try:
        import flax  # noqa
    except ImportError:
        pytest.skip("flax not installed")

    x = cuex.RepArray("3x0 + 2x1", jnp.ones((16, 9)))
    linear = cuex.flax_linen.Linear("2x0 + 1")
    w = linear.init(jax.random.key(0), x)
    y = linear.apply(w, x)
    assert y.shape == (16, 5)
    assert y.irreps == "2x0 + 1"
