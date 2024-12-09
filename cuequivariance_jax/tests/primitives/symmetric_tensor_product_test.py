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
import numpy as np

import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance import descriptors

jax.config.update("jax_enable_x64", True)


def test_custom_jvp():
    e = descriptors.symmetric_contraction(
        3 * cue.Irreps(cue.O3, "0e + 1o"),
        3 * cue.Irreps(cue.O3, "0e"),
        [0, 1, 2, 3, 4],
    )
    w = np.random.randn(2, e.inputs[0].irreps.dim)
    x = np.random.randn(2, e.inputs[1].irreps.dim)

    A = jax.grad(
        lambda x: jnp.sum(
            cuex.symmetric_tensor_product(e.ds, w, x, use_custom_primitive=True) ** 2
        )
    )(x)
    B = jax.grad(
        lambda x: jnp.sum(
            cuex.symmetric_tensor_product(e.ds, w, x, use_custom_primitive=False) ** 2
        )
    )(x)

    np.testing.assert_allclose(A, B, atol=1e-10, rtol=1e-10)


def test_shapes():
    ds = descriptors.symmetric_contraction(
        cue.Irreps(cue.O3, "0e + 1o"), cue.Irreps(cue.O3, "0e"), [4]
    ).ds
    W = ds[0].operands[0].size
    X = ds[0].operands[1].size
    Y = ds[0].operands[-1].size
    r = np.random.randn

    assert cuex.symmetric_tensor_product(ds, r(32, W), r(32, X)).shape == (32, Y)
    assert cuex.symmetric_tensor_product(ds, r(W), r(X)).shape == (Y,)
    assert cuex.symmetric_tensor_product(ds, r(W), r(32, X)).shape == (32, Y)
    assert cuex.symmetric_tensor_product(ds, r(32, W), r(X)).shape == (32, Y)
    assert cuex.symmetric_tensor_product(ds, r(2, 1, 2, 3, W), r(3, 1, 1, X)).shape == (
        2,
        3,
        2,
        3,
        Y,
    )
