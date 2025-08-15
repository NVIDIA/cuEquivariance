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
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cuequivariance as cue
import cuequivariance_jax as cuex

jax.config.update("jax_enable_x64", True)


def test_one_operand():
    d = cue.SegmentedTensorProduct.empty_segments([1])
    poly = cue.SegmentedPolynomial([], [cue.SegmentedOperand([()])], [([0], d)])
    [out] = cuex.segmented_polynomial(
        poly, [], [jax.ShapeDtypeStruct((2, 1), jnp.float32)], method="naive"
    )
    np.testing.assert_array_equal(out, np.array([[0.0], [0.0]]))

    d.add_path(0, c=123)
    poly = cue.SegmentedPolynomial([], [cue.SegmentedOperand([()])], [([0], d)])
    [out] = cuex.segmented_polynomial(
        poly, [], [jax.ShapeDtypeStruct((2, 1), jnp.float32)], method="naive"
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
            e.polynomial,
            [w, x],
            [jax.ShapeDtypeStruct((2, 1), jnp.float32)],
            method="naive",
        )
        return jnp.sum(out)

    jax.jit(jax.grad(f, 0))(w, x)


def test_multiple_operand_shape_bug():
    # This was causing an issue in the past.
    # Before, it was not possible to have an input
    # with a different shape than the output of the same operand.
    def h(x):
        poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [2]).polynomial
        [out] = cuex.segmented_polynomial(
            poly, [x], [jax.ShapeDtypeStruct((5,), jnp.float32)], method="naive"
        )
        return out

    assert jax.jacobian(h)(jnp.array([1.0, 0.0, 0.0])).shape == (5, 3)


def test_broadcasting():
    poly = cue.descriptors.full_tensor_product(
        cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1")
    ).polynomial

    x = jnp.ones((2, 1, 3))
    y = jnp.ones((1, 2, 3))
    [out] = cuex.segmented_polynomial(
        poly, [x, y], [jax.ShapeDtypeStruct((2, 2, 3), jnp.float32)], method="naive"
    )
    assert out.shape == (2, 2, 3)


@pytest.mark.parametrize("mul", [10, 32])
@pytest.mark.parametrize("method", ["naive", "uniform_1d"])
def test_empty_input(mul: int, method: str):
    poly = cue.descriptors.channelwise_tensor_product(
        mul * cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1")
    ).polynomial

    w = jnp.ones((poly.inputs[0].size,))
    x = jnp.ones((2, 0, mul * 3))
    y = jnp.ones((1, 0, 3))
    [out] = cuex.segmented_polynomial(
        poly,
        [w, x, y],
        [jax.ShapeDtypeStruct((2, 0, mul * 3), jnp.float32)],
        method=method,
    )
    assert out.shape == (2, 0, mul * 3)


@pytest.mark.parametrize("method", ["naive", "uniform_1d"])
def test_no_batch(method: str):
    poly = cue.descriptors.channelwise_tensor_product(
        32 * cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1")
    ).polynomial

    w, x, y = jnp.ones((poly.inputs[0].size,)), jnp.ones((96,)), jnp.ones((3,))
    [out] = cuex.segmented_polynomial(
        poly, [w, x, y], [jax.ShapeDtypeStruct((96,), jnp.float32)], method=method
    )
    assert out.shape == (96,)


def test_vmap():
    e = cue.descriptors.full_tensor_product(
        cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1")
    )
    ((_, d),) = e.polynomial.operations

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
            method="naive",
        )

    def g(outs):
        return jax.tree.map(jnp.shape, outs)

    x1 = jnp.ones((3, 3))
    x2 = jnp.ones((2, 3))
    i1 = jnp.array([0, 2])
    assert g(f(x1, x2, i1)) == [(2, 3), (1, 3)]

    bx1 = jnp.ones((4, 3, 3))
    bx2 = jnp.ones((4, 2, 3))
    bi1 = jnp.array([[0, 2], [1, 2], [0, 0], [1, 1]])
    assert g(jax.vmap(f, (None, 0, None))(x1, bx2, i1)) == [(4, 2, 3), (4, 1, 3)]
    assert g(jax.vmap(f, (None, None, 0))(x1, x2, bi1)) == [(4, 2, 3), (4, 1, 3)]
    assert g(jax.vmap(f, (None, 0, 0))(x1, bx2, bi1)) == [(4, 2, 3), (4, 1, 3)]
    assert g(jax.vmap(f, (0, 0, 0))(bx1, bx2, bi1)) == [(4, 2, 3), (4, 1, 3)]
    assert g(jax.vmap(f, (0, None, None))(bx1, x2, i1)) == [(4, 2, 3), (4, 1, 3)]


@pytest.mark.skipif(
    not importlib.util.find_spec("cuequivariance_ops_jax"),
    reason="cuequivariance_ops_jax is not installed",
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64])
def test_compare_uniform_1d_with_naive(dtype):
    poly = cue.descriptors.channelwise_tensor_product(
        32 * cue.Irreps("SO3", "0 + 1 + 2"),
        cue.Irreps("SO3", "0 + 1 + 2"),
        cue.Irreps("SO3", "0 + 1 + 2"),
    ).polynomial

    operands = [
        jax.random.normal(jax.random.key(i), (10, 10, ope.size), dtype=dtype)
        for i, ope in enumerate(poly.operands)
    ]
    indices = [
        jnp.s_[jax.random.randint(jax.random.key(100), (10, 10), 0, 10), :],
        jnp.s_[:, jax.random.randint(jax.random.key(101), (1, 10), 0, 10)],
        None,
        jnp.s_[
            jax.random.randint(jax.random.key(102), (10, 1), 0, 10),
            jax.random.randint(jax.random.key(103), (10, 10), 0, 10),
        ],
    ]

    [jax_out] = cuex.segmented_polynomial(
        poly,
        operands[: poly.num_inputs],
        operands[poly.num_inputs :],
        indices,
        method="naive",
    )
    [cud_out] = cuex.segmented_polynomial(
        poly,
        operands[: poly.num_inputs],
        operands[poly.num_inputs :],
        indices,
        method="uniform_1d",
    )
    assert jax_out.shape == cud_out.shape
    assert jax_out.dtype == cud_out.dtype
    jax_out = np.asarray(jax_out, dtype=np.float64)
    cud_out = np.asarray(cud_out, dtype=np.float64)
    np.testing.assert_allclose(
        jax_out,
        cud_out,
        atol={
            jnp.bfloat16: 1,
            jnp.float16: 1e-1,
            jnp.float32: 1e-4,
            jnp.float64: 1e-12,
        }[dtype],
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16, jnp.float64])
def test_indexed_linear_method(dtype):
    jax.config.update("jax_enable_x64", True)
    method = "indexed_linear" if jax.default_backend() == "gpu" else "naive"

    num_species_total = 3
    batch_size = 10
    input_dim = 8
    output_dim = 16
    num_species = jnp.array([3, 4, 3], dtype=jnp.int32)
    input_array = jax.random.normal(jax.random.key(0), (batch_size, input_dim), dtype)
    input_irreps = cue.Irreps(cue.O3, f"{input_dim}x0e")
    output_irreps = cue.Irreps(cue.O3, f"{output_dim}x0e")
    e = cue.descriptors.linear(input_irreps, output_irreps)
    w = jax.random.normal(
        jax.random.key(1), (num_species_total, e.inputs[0].dim), dtype
    )

    [result] = cuex.segmented_polynomial(
        e.polynomial,
        [w, input_array],
        [jax.ShapeDtypeStruct((batch_size, output_dim), dtype)],
        [cuex.Repeats(num_species), None, None],
        method=method,
    )
    assert result.shape == (batch_size, output_dim)

    [ref] = cuex.segmented_polynomial(
        e.polynomial,
        [w, input_array],
        [jax.ShapeDtypeStruct((batch_size, output_dim), dtype)],
        [jnp.repeat(jnp.arange(num_species_total), num_species), None, None],
        method="naive",
    )

    result = np.asarray(result, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    match dtype:
        case jnp.float16 | jnp.bfloat16:
            atol, rtol = 1e-2, 1e-2
        case jnp.float32:
            atol, rtol = 1e-3, 1e-3
        case jnp.float64:
            atol, rtol = 1e-6, 1e-6
    np.testing.assert_allclose(result, ref, rtol=rtol, atol=atol)


def test_math_dtype_backward_compatibility():
    """Test that jnp.dtype objects work for math_dtype (backward compatibility)."""
    poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1]).polynomial
    x = jnp.array([0.0, 1.0, 0.0])

    # Test with jnp.dtype object (backward compatibility)
    [result_dtype] = cuex.segmented_polynomial(
        poly,
        [x],
        [jax.ShapeDtypeStruct((-1,), jnp.float32)],
        method="naive",
        math_dtype=jnp.float32,
    )

    # Test with string (official API)
    [result_str] = cuex.segmented_polynomial(
        poly,
        [x],
        [jax.ShapeDtypeStruct((-1,), jnp.float32)],
        method="naive",
        math_dtype="float32",
    )

    np.testing.assert_allclose(result_dtype, result_str, rtol=1e-12, atol=1e-12)
