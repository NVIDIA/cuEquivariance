# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import *

import jax
import jax.numpy as jnp
import numpy as np

import cuequivariance as cue
from cuequivariance import descriptors
import cuequivariance_jax as cuex

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
