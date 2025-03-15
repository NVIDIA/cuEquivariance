# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

jax.config.update("jax_enable_x64", True)


def test_one_operand():
    d = cue.SegmentedTensorProduct.empty_segments([1])
    [out] = cuex.segmented_polynomial(
        cue.SegmentedPolynomial(
            [], [cue.SegmentedOperand.empty_segments(1)], [(cue.Operation([0]), d)]
        ),
        [],
        [jax.ShapeDtypeStruct((2, 1), jnp.float32)],
    )
    np.testing.assert_array_equal(out, np.array([[0.0], [0.0]]))

    d.add_path(0, c=123)
    [out] = cuex.segmented_polynomial(
        cue.SegmentedPolynomial(
            [], [cue.SegmentedOperand.empty_segments(1)], [(cue.Operation([0]), d)]
        ),
        [],
        [jax.ShapeDtypeStruct((2, 1), jnp.float32)],
    )
    np.testing.assert_array_equal(out, np.array([[123.0], [123.0]]))


def test_UnshapedArray_bug():
    e = cue.descriptors.symmetric_contraction(
        cue.Irreps("O3", "0e"), cue.Irreps("O3", "0e"), [0, 1]
    )
    w = jnp.ones((1, 2))
    x = jnp.ones((2, 1))

    def f(w, x):
        [out] = cuex.segmented_polynomial(
            e.polynomial, [w, x], [jax.ShapeDtypeStruct((2, 1), jnp.float32)]
        )
        return jnp.sum(out)

    jax.jit(jax.grad(f, 0))(w, x)


def test_multiple_operand_shape_bug():
    # This was causing an issue in the past.
    # Before, it was not possible to have an input
    # with a different shape than the output of the same operand.
    def h(x):
        e = cue.descriptors.spherical_harmonics(cue.SO3(1), [2])
        [out] = cuex.segmented_polynomial(
            e.polynomial,
            [x],
            [jax.ShapeDtypeStruct((5,), jnp.float32)],
        )
        return out

    assert jax.jacobian(h)(jnp.array([1.0, 0.0, 0.0])).shape == (5, 3)


# def test_broadcasting():
#     e = cue.descriptors.full_tensor_product(
#         cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1")
#     )

#     x = jnp.ones((2, 1, 3))
#     y = jnp.ones((1, 2, 3))
#     [out] = cuex.tensor_product(
#         [(cue.Operation([0, 1, 2]), e.ds[0])],
#         [x, y],
#         [jax.ShapeDtypeStruct((2, 2, 3), jnp.float32)],
#     )
#     assert out.shape == (2, 2, 3)


def test_vmap():
    e = cue.descriptors.full_tensor_product(
        cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1")
    )
    d = e.polynomial.operations[0][1]

    def f(x1, x2, i1):
        return cuex.segmented_polynomial(
            cue.SegmentedPolynomial(
                d.operands[:2],
                [d.operands[2], d.operands[2]],
                [
                    (cue.Operation([0, 1, 2]), d),
                    (cue.Operation([0, 1, 3]), d),
                ],
            ),
            [x1, x2],
            [
                jax.ShapeDtypeStruct((2, 3), jnp.float32),
                jax.ShapeDtypeStruct((1, 3), jnp.float32),
            ],
            indices=[i1, None, None, None],
        )

    def g(outs):
        return jax.tree.map(jnp.shape, outs)

    x1 = jnp.ones((3, 3))
    x2 = jnp.ones((2, 3))
    i1 = jnp.array([0, 2])
    assert g(f(x1, x2, i1)) == [(2, 3), (1, 3)]

    bx2 = jnp.ones((4, 2, 3))
    bi1 = jnp.array([[0, 2], [1, 2], [0, 0], [1, 1]])
    assert g(jax.vmap(f, (None, 0, None))(x1, bx2, i1)) == [(4, 2, 3), (4, 1, 3)]
    assert g(jax.vmap(f, (None, None, 0))(x1, x2, bi1)) == [(4, 2, 3), (4, 1, 3)]
    assert g(jax.vmap(f, (None, 0, 0))(x1, bx2, bi1)) == [(4, 2, 3), (4, 1, 3)]
