# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import jax

import cuequivariance as cue
import cuequivariance_jax as cuex


def test_special_double_backward():
    e = cue.equivariant_tensor_product.symmetric_contraction(
        2 * cue.Irreps("O3", "0e + 1o + 2e"), 2 * cue.Irreps("O3", "0e + 1o"), [1, 2]
    )
    irreps_w = e.inputs[0].irreps
    irreps_x = e.inputs[1].irreps
    h = cuex.equivariant_tensor_product(e)

    h0 = lambda w, x: h(w, x).array.sum() ** 2
    h1 = lambda w, x: jax.grad(h0, 1)(w, x).array.sum() ** 2

    w = jax.random.normal(jax.random.key(0), (1, irreps_w.dim))
    x = cuex.IrrepsArray(
        irreps_x, jax.random.normal(jax.random.key(1), (3, irreps_x.dim)), cue.ir_mul
    )
    jax.grad(h1, 0)(w, x)
