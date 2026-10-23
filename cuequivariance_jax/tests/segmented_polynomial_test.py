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
import builtins
import importlib
import sys
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cuequivariance_jax.segmented_polynomials.indexing_mode import IndexingMode
from cuequivariance_jax.segmented_polynomials.segmented_polynomial_indexed_linear import (
    Buffer,
    ein,
    gather,
    scatter,
)
from cuequivariance_jax.segmented_polynomials.segmented_polynomial_naive import (
    sum_cat_list_list,
    tp_list_list,
)
from cuequivariance_jax.segmented_polynomials.segmented_polynomial_uniform_1d import (
    execute_uniform_1d,
    sanitize_string,
)

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


def _indexed_linear_test_methods():
    methods = ["naive"]
    if jax.default_backend() == "gpu" and importlib.util.find_spec(
        "cuequivariance_ops_jax"
    ):
        methods.append("indexed_linear")
    return methods


def test_indexed_linear_repeated_input_with_multiple_paths():
    d = cue.SegmentedTensorProduct.from_subscripts("u,uv,v")
    d.add_segment(0, (3,))
    d.add_segment(0, (3,))
    d.add_segment(1, (3, 4))
    d.add_segment(2, (4,))
    d.add_path(0, 0, 0, c=1.0)
    d.add_path(1, 0, 0, c=-0.25)
    polynomial = cue.SegmentedPolynomial.eval_last_operand(d)

    counts = jnp.array([2, 1], dtype=jnp.int32)
    categories = jnp.repeat(jnp.arange(2), counts)
    features = jnp.arange(18, dtype=jnp.float32).reshape(3, 6) / 10
    weights = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10
    weights_by_row = weights[categories]

    expected = jnp.einsum("zu,zuv->zv", features[:, :3], weights_by_row)
    expected -= 0.25 * jnp.einsum("zu,zuv->zv", features[:, 3:], weights_by_row)
    for method in _indexed_linear_test_methods():
        [actual] = cuex.segmented_polynomial(
            polynomial,
            [features, weights.reshape(2, -1)],
            [jax.ShapeDtypeStruct((3, 4), jnp.float32)],
            [None, cuex.Repeats(counts), None],
            method=method,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_indexed_linear_repeated_matrix_input():
    d = cue.SegmentedTensorProduct.from_subscripts("uv,wv,uw")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (4, 3))
    d.add_segment(2, (2, 4))
    d.add_path(0, 0, 0, c=1.0)
    polynomial = cue.SegmentedPolynomial.eval_last_operand(d)

    counts = jnp.array([2, 1], dtype=jnp.int32)
    categories = jnp.repeat(jnp.arange(2), counts)
    features = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3) / 10
    weights = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 10

    expected = jnp.einsum("zuv,zwv->zuw", features, weights[categories])
    for method in _indexed_linear_test_methods():
        [actual] = cuex.segmented_polynomial(
            polynomial,
            [features.reshape(3, -1), weights.reshape(2, -1)],
            [jax.ShapeDtypeStruct((3, 8), jnp.float32)],
            [None, cuex.Repeats(counts), None],
            method=method,
        )
        np.testing.assert_allclose(actual, expected.reshape(3, 8), rtol=1e-6, atol=1e-6)


def test_indexed_linear_repeated_output():
    d = cue.SegmentedTensorProduct.from_subscripts("uv,uw,wv")
    d.add_segment(0, (3, 2))
    d.add_segment(1, (3, 4))
    d.add_segment(2, (4, 2))
    d.add_path(0, 0, 0, c=1.0)
    polynomial = cue.SegmentedPolynomial.eval_last_operand(d)

    counts = jnp.array([2, 1], dtype=jnp.int32)
    left = jnp.arange(18, dtype=jnp.float32).reshape(3, 3, 2) / 10
    right = jnp.arange(36, dtype=jnp.float32).reshape(3, 3, 4) / 10

    per_row = jnp.einsum("zuv,zuw->zwv", left, right)
    expected = jnp.stack(
        [jnp.sum(per_row[:2], axis=0), per_row[2]],
    )
    for method in _indexed_linear_test_methods():
        [actual] = cuex.segmented_polynomial(
            polynomial,
            [left.reshape(3, -1), right.reshape(3, -1)],
            [jax.ShapeDtypeStruct((2, 8), jnp.float32)],
            [None, None, cuex.Repeats(counts)],
            method=method,
        )
        np.testing.assert_allclose(actual, expected.reshape(2, 8), rtol=1e-6, atol=1e-6)


def test_indexed_linear_indexed_scatter_and_gather():
    indices = [jnp.array([2, 0, 2], dtype=jnp.int32)]
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    scattered = scatter(
        x,
        (0,),
        (IndexingMode.INDEXED,),
        indices,
        batch_sizes=[3],
    )
    np.testing.assert_array_equal(scattered, x[indices[0]])

    initial = jnp.ones((4, 4), dtype=jnp.float32)
    updates = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    actual = gather(initial, updates, (0,), (IndexingMode.INDEXED,), indices)
    expected = initial.at[indices[0]].add(updates)
    np.testing.assert_array_equal(actual, expected)


def test_indexed_linear_rejects_unmatched_kernel_formula():
    d = cue.SegmentedTensorProduct.from_subscripts("uv,vw,uw")
    buffers = [
        Buffer(
            jnp.ones((2, 3, 4), dtype=jnp.float32),
            (-1,),
            (IndexingMode.BATCHED_OR_SHARED,),
        ),
        Buffer(
            jnp.ones((2, 4, 5), dtype=jnp.float32),
            (-1,),
            (IndexingMode.BATCHED_OR_SHARED,),
        ),
    ]
    output = Buffer(
        jnp.zeros((2, 3, 5), dtype=jnp.float32),
        (-1,),
        (IndexingMode.BATCHED_OR_SHARED,),
    )

    with pytest.raises(
        ValueError, match="not possible to execute the method indexed_linear"
    ):
        ein(
            np.asarray(1.0),
            buffers,
            output,
            [],
            list(d.subscripts.operands),
            d.coefficient_subscripts,
            [2],
            math_dtype=None,
            run_kernel=True,
        )


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


def _uniform_naive_tp_case():
    d = cue.SegmentedTensorProduct.from_subscripts("uv,vw,uw")
    for _ in range(2):
        d.add_segment(0, (2, 3))
        d.add_segment(1, (3, 2))
    d.add_segment(2, (2, 2))
    d.add_path(0, 0, 0, c=1.5)
    d.add_path(1, 1, 0, c=-0.25)

    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 12) / 10
    y = jnp.arange(24, 48, dtype=jnp.float32).reshape(2, 12) / 10
    x_segments = x.reshape(2, 2, 2, 3)
    y_segments = y.reshape(2, 2, 3, 2)
    expected = (
        1.5 * jnp.einsum("buv,bvw->buw", x_segments[:, 0], y_segments[:, 0])
        - 0.25 * jnp.einsum("buv,bvw->buw", x_segments[:, 1], y_segments[:, 1])
    ).reshape(2, 4)
    return d, x, y, expected


def _uniform_1d_single_mode_case():
    d = cue.SegmentedTensorProduct.from_subscripts("u,u,u")
    for operand in range(3):
        d.add_segment(operand, (2,))
    d.add_path(0, 0, 0, c=1.0)

    polynomial = cue.SegmentedPolynomial(
        d.operands[:2], d.operands[2:], [(cue.Operation([0, 1, 2]), d)]
    )
    inputs = [jnp.ones((2, 2), dtype=jnp.float32) for _ in range(2)]
    outputs = [jax.ShapeDtypeStruct((2, 2), jnp.float32)]
    return polynomial, inputs, outputs


def _execute_uniform_1d_for_test(
    polynomial,
    inputs,
    outputs_shape_dtype,
    *,
    indices=None,
    index_configuration=None,
    options=None,
):
    indices = [] if indices is None else indices
    if index_configuration is None:
        num_batch_axes = (
            len(inputs[0].shape) - 1
            if inputs
            else len(outputs_shape_dtype[0].shape) - 1
        )
        index_configuration = tuple(
            (-1,) * num_batch_axes
            for _ in range(len(inputs) + len(outputs_shape_dtype) + len(indices))
        )

    return execute_uniform_1d(
        inputs,
        tuple(outputs_shape_dtype),
        indices,
        index_configuration,
        polynomial,
        {} if options is None else options,
        "uniform_1d_test",
    )


def test_uniform_1d_sanitizes_kernel_name():
    assert sanitize_string("3 bad/name") == "_3badname"
    assert sanitize_string("safe_name") == "safe_name"


def test_uniform_1d_rejects_multiple_modes_and_nonuniform_segments():
    d, x, y, expected = _uniform_naive_tp_case()
    multi_mode_polynomial = cue.SegmentedPolynomial(
        d.operands[:2], d.operands[2:], [(cue.Operation([0, 1, 2]), d)]
    )
    with pytest.raises(ValueError, match="requires exactly one mode"):
        _execute_uniform_1d_for_test(
            multi_mode_polynomial,
            [x, y],
            [jax.ShapeDtypeStruct(expected.shape, expected.dtype)],
        )

    d = cue.SegmentedTensorProduct.from_subscripts("u,u,u")
    for size in (2, 3):
        for operand in range(3):
            d.add_segment(operand, (size,))
    d.add_path(0, 0, 0, c=1.0)
    d.add_path(1, 1, 1, c=1.0)
    polynomial = cue.SegmentedPolynomial(
        d.operands[:2], d.operands[2:], [(cue.Operation([0, 1, 2]), d)]
    )
    inputs = [jnp.ones((2, 5), dtype=jnp.float32) for _ in range(2)]
    outputs = [jax.ShapeDtypeStruct((2, 5), jnp.float32)]
    with pytest.raises(
        ValueError, match="requires all segments to have uniform shapes"
    ):
        _execute_uniform_1d_for_test(polynomial, inputs, outputs)


def test_uniform_1d_wraps_coefficient_flattening_error():
    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,jv+ij")
    d.add_path(
        None,
        None,
        None,
        c=np.ones((2, 3)),
        dims={"u": 4, "v": 5, "i": 2, "j": 3},
    )
    d.add_path(
        None,
        None,
        None,
        c=np.ones((4, 5)),
        dims={"u": 2, "v": 3, "i": 4, "j": 5},
    )
    d = d.add_or_transpose_modes("uv,ui,jv+ij")
    polynomial = cue.SegmentedPolynomial(
        d.operands[:2], d.operands[2:], [(cue.Operation([0, 1, 2]), d)]
    )
    inputs = [
        jnp.ones((1, operand.size), dtype=jnp.float32) for operand in polynomial.inputs
    ]
    outputs = [jax.ShapeDtypeStruct((1, polynomial.outputs[0].size), jnp.float32)]

    with pytest.raises(
        ValueError, match="This method does not support coefficient modes"
    ):
        _execute_uniform_1d_for_test(polynomial, inputs, outputs)


def test_uniform_1d_validates_buffer_index_and_math_dtypes():
    polynomial, inputs, outputs = _uniform_1d_single_mode_case()

    bad_inputs = [inputs[0].astype(jnp.int32), inputs[1]]
    with pytest.raises(ValueError, match="Unsupported buffer type"):
        _execute_uniform_1d_for_test(polynomial, bad_inputs, outputs)

    indices = [jnp.array([0, 1], dtype=jnp.float32)]
    with pytest.raises(ValueError, match="Unsupported index type"):
        _execute_uniform_1d_for_test(polynomial, inputs, outputs, indices=indices)

    with pytest.raises(ValueError, match="only supports math_dtype"):
        _execute_uniform_1d_for_test(
            polynomial, inputs, outputs, options={"math_dtype": "int32"}
        )


def test_uniform_1d_handles_no_mode_and_rejects_mixed_extents():
    d = cue.SegmentedTensorProduct.empty_segments([1])
    d.add_path(0, c=1.0)
    scalar_polynomial = cue.SegmentedPolynomial(
        [],
        [cue.SegmentedOperand([()])],
        [(cue.Operation([0]), d)],
    )
    with pytest.raises(ValueError, match="only supports math_dtype"):
        _execute_uniform_1d_for_test(
            scalar_polynomial,
            [],
            [jax.ShapeDtypeStruct((1,), jnp.float32)],
            index_configuration=((),),
            options={"math_dtype": "int32"},
        )

    descriptors = []
    for size in (2, 3):
        stp = cue.SegmentedTensorProduct.from_subscripts("u,u")
        stp.add_segment(0, (size,))
        stp.add_segment(1, (size,))
        stp.add_path(0, 0, c=1.0)
        descriptors.append(stp)
    d2, d3 = descriptors
    polynomial = cue.SegmentedPolynomial(
        [d2.operands[0], d3.operands[0]],
        [d2.operands[1], d3.operands[1]],
        [(cue.Operation([0, 2]), d2), (cue.Operation([1, 3]), d3)],
    )
    inputs = [jnp.ones((2, 2), dtype=jnp.float32), jnp.ones((2, 3), dtype=jnp.float32)]
    outputs = [
        jax.ShapeDtypeStruct((2, 2), jnp.float32),
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
    ]
    with pytest.raises(ValueError, match="requires a single uniform mode"):
        _execute_uniform_1d_for_test(polynomial, inputs, outputs)


def test_uniform_1d_reports_missing_kernel_dependency(monkeypatch):
    polynomial, inputs, outputs = _uniform_1d_single_mode_case()
    original_import = builtins.__import__

    def reject_ops_jax(module_name, *args, **kwargs):
        if module_name == "cuequivariance_ops_jax":
            raise ImportError("forced missing optional dependency")
        return original_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ops_jax)
    with pytest.raises(ValueError, match="cuequivariance_ops_jax is not installed"):
        _execute_uniform_1d_for_test(
            polynomial,
            inputs,
            outputs,
            options={"math_dtype": "float32"},
        )


def test_uniform_1d_warns_for_old_kernel_dependency(monkeypatch):
    polynomial, inputs, outputs = _uniform_1d_single_mode_case()
    old_ops_jax = ModuleType("cuequivariance_ops_jax")
    old_ops_jax.Operation = object
    old_ops_jax.Path = object
    old_ops_jax.__version__ = "0.3.0"
    old_ops_jax.uniform_1d = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "cuequivariance_ops_jax", old_ops_jax)

    with pytest.warns(UserWarning, match="too old"):
        with pytest.raises(ValueError, match="too old"):
            _execute_uniform_1d_for_test(
                polynomial,
                inputs,
                outputs,
                options={"math_dtype": "float32"},
            )


def test_segmented_polynomial_validates_public_arguments():
    polynomial, _, outputs = _uniform_1d_single_mode_case()
    left = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    right = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
    inputs = [left, right]

    with pytest.raises(ValueError, match="`method` argument is now required"):
        cuex.segmented_polynomial(polynomial, inputs, outputs, method="")

    with pytest.raises(ValueError, match="math_dtype provided both"):
        cuex.segmented_polynomial(
            polynomial,
            inputs,
            outputs,
            method="naive",
            math_dtype="float32",
            options={"math_dtype": "float32"},
        )

    with pytest.raises(ValueError, match="precision is not anymore supported"):
        cuex.segmented_polynomial(
            polynomial,
            inputs,
            outputs,
            method="naive",
            precision=jax.lax.Precision.HIGHEST,
        )

    with pytest.raises(ValueError, match="Input 0 has no dimensions"):
        cuex.segmented_polynomial(
            polynomial,
            [jnp.array(1.0), right],
            outputs,
            method="naive",
        )

    with pytest.raises(ValueError, match="expected shape 2"):
        cuex.segmented_polynomial(
            polynomial,
            [left[:, :1], right],
            outputs,
            method="naive",
        )

    with pytest.raises(ValueError, match="Output has no dimensions"):
        cuex.segmented_polynomial(
            polynomial,
            inputs,
            [jax.ShapeDtypeStruct((), jnp.float32)],
            method="naive",
        )

    with pytest.warns(UserWarning, match="expected the last dimension to be 2"):
        [reshaped_output] = cuex.segmented_polynomial(
            polynomial,
            inputs,
            [jax.ShapeDtypeStruct((2, 3), jnp.float32)],
            method="naive",
        )
    np.testing.assert_allclose(reshaped_output, left * right)

    with pytest.raises(ValueError, match="Expected 3 indices, got 1"):
        cuex.segmented_polynomial(
            polynomial,
            inputs,
            outputs,
            indices=[None],
            method="naive",
        )

    with pytest.raises(ValueError, match="Expected index to be a jax.Array"):
        cuex.segmented_polynomial(
            polynomial,
            inputs,
            outputs,
            indices=[(object(),), None, None],
            method="naive",
        )

    options = {"math_dtype": jnp.float32}
    [actual] = cuex.segmented_polynomial(
        polynomial, inputs, outputs, method="naive", options=options
    )
    np.testing.assert_allclose(actual, left * right)
    assert options == {"math_dtype": jnp.float32}


def test_segmented_polynomial_prints_stats(monkeypatch, capsys):
    polynomial, _, outputs = _uniform_1d_single_mode_case()
    inputs = [jnp.ones((2, 2), dtype=jnp.float32) for _ in range(2)]
    monkeypatch.setenv("CUE_PRINT_STATS", "1")

    # Stats are printed while lowering: use a distinct primitive name to avoid
    # reusing a lowering cached before CUE_PRINT_STATS was enabled.
    [actual] = cuex.segmented_polynomial(
        polynomial, inputs, outputs, method="naive", name="stats_reporting_test"
    )

    np.testing.assert_array_equal(actual, jnp.ones((2, 2), dtype=jnp.float32))
    printed = capsys.readouterr().out
    assert "stats_reporting_test:" in printed
    assert "GFLOP" in printed and "GB" in printed


def test_segmented_polynomial_continues_when_unsymmetrize_is_unsupported(monkeypatch):
    polynomial, _, outputs = _uniform_1d_single_mode_case()
    left = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    right = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

    calls = []

    def raise_not_implemented(self):
        calls.append(self)
        raise NotImplementedError("forced unsupported symmetrization")

    monkeypatch.setattr(
        cue.SegmentedPolynomial,
        "unsymmetrize_for_identical_operands",
        raise_not_implemented,
    )
    [actual] = cuex.segmented_polynomial(
        polynomial,
        [left, right],
        outputs,
        method="naive",
        name="unsupported_unsymmetrize_test",
    )
    np.testing.assert_allclose(actual, left * right)
    assert calls, "the patched unsymmetrization branch must execute"


def test_segmented_polynomial_rejects_repeats_for_uniform_dispatch():
    polynomial, inputs, outputs = _uniform_1d_single_mode_case()
    module = importlib.import_module(
        "cuequivariance_jax.segmented_polynomials.segmented_polynomial"
    )
    repeats = jnp.array([1, 1], dtype=jnp.int32)

    with pytest.raises(ValueError, match="REPEATED is only supported"):
        module.segmented_polynomial_impl(
            "cuda",
            *inputs,
            repeats,
            index_configuration=((0,), (-1,), (-1,)),
            index_mode=(
                (IndexingMode.REPEATED,),
                (IndexingMode.BATCHED_OR_SHARED,),
                (IndexingMode.BATCHED_OR_SHARED,),
            ),
            outputs_shape_dtype=tuple(outputs),
            polynomial=polynomial,
            options=(),
            name="repeated_uniform_test",
            method="uniform_1d",
        )


def test_segmented_polynomial_dispatches_indexed_linear(monkeypatch):
    polynomial, inputs, outputs = _uniform_1d_single_mode_case()
    module = importlib.import_module(
        "cuequivariance_jax.segmented_polynomials.segmented_polynomial"
    )

    def fake_indexed_linear(**kwargs):
        assert kwargs["polynomial"].num_inputs == 2
        assert kwargs["name"] == "indexed_dispatch_test"
        assert kwargs["options"] == {"math_dtype": "float32"}
        assert kwargs["index_mode"] == (
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.BATCHED_OR_SHARED,),
        )
        return (jnp.full((2, 2), 9.0, dtype=jnp.float32),)

    monkeypatch.setattr(module, "execute_indexed_linear", fake_indexed_linear)
    actual = module.segmented_polynomial_impl(
        "cuda",
        *inputs,
        index_configuration=((-1,), (-1,), (-1,)),
        index_mode=(
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.BATCHED_OR_SHARED,),
        ),
        outputs_shape_dtype=tuple(outputs),
        polynomial=polynomial,
        options=(("math_dtype", "float32"),),
        name="indexed_dispatch_test",
        method="indexed_linear",
    )
    np.testing.assert_array_equal(actual[0], jnp.full((2, 2), 9.0, jnp.float32))


def test_uniform_1d_rejects_unused_buffer():
    stp = cue.SegmentedTensorProduct.from_subscripts("u,u")
    stp.add_segment(0, (2,))
    stp.add_segment(1, (2,))
    stp.add_path(0, 0, c=1.0)
    polynomial = cue.SegmentedPolynomial(
        [stp.operands[0], cue.SegmentedOperand.empty_segments(2)],
        [stp.operands[1]],
        [(cue.Operation([0, 2]), stp)],
    )
    inputs = [jnp.ones((2, 2), dtype=jnp.float32) for _ in range(2)]
    outputs = [jax.ShapeDtypeStruct((2, 2), jnp.float32)]

    with pytest.raises(ValueError, match="All buffers must be used"):
        _execute_uniform_1d_for_test(polynomial, inputs, outputs)


@pytest.mark.parametrize(
    "algorithm",
    [
        "stacked",
        "compact_stacked",
        "indexed_vmap",
        "indexed_compact",
        "indexed_for_loop",
    ],
)
def test_tp_list_list_naive_algorithms(algorithm):
    d, x, y, expected = _uniform_naive_tp_case()
    output_segments = tp_list_list(
        x,
        y,
        out_batch_shape=(2,),
        d=d,
        output_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        algorithm=algorithm,
    )

    actual = sum_cat_list_list(d.operands[-1], output_segments, (2,), jnp.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_tp_list_list_naive_no_op():
    d, x, y, _ = _uniform_naive_tp_case()
    with pytest.warns(UserWarning, match="skipping computation"):
        output_segments = tp_list_list(
            x,
            y,
            out_batch_shape=(2,),
            d=d,
            output_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            algorithm="no-op",
        )

    actual = sum_cat_list_list(d.operands[-1], output_segments, (2,), jnp.float32)
    dummy = jnp.sum(x) + jnp.sum(y)
    np.testing.assert_allclose(actual, jnp.full((2, 4), 2 * dummy))


def test_tp_list_list_naive_dispatch_and_sliced_shapes(monkeypatch):
    monkeypatch.setenv("CUEQUIVARIANCE_MAX_PATH_UNROLL", "1")
    d, x, y, expected = _uniform_naive_tp_case()
    with pytest.warns(UserWarning, match="switching algorithm from stacked"):
        output_segments = tp_list_list(
            x,
            y,
            out_batch_shape=(2,),
            d=d,
            output_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            algorithm="stacked",
        )
    actual = sum_cat_list_list(d.operands[-1], output_segments, (2,), jnp.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    d = cue.SegmentedTensorProduct.from_subscripts("uv,vw,uw")
    d.add_path(None, None, None, c=1.0, dims={"u": 2, "v": 3, "w": 2})
    d.add_path(None, None, None, c=-0.25, dims={"u": 1, "v": 2, "w": 1})
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 10
    y = jnp.arange(16, 32, dtype=jnp.float32).reshape(2, 8) / 10

    with pytest.warns(UserWarning, match="more than 1 paths"):
        with pytest.raises(
            ValueError, match="all operands must have the same segment shape"
        ):
            tp_list_list(
                x,
                y,
                out_batch_shape=(2,),
                d=d,
                output_dtype=jnp.float32,
                compute_dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
                algorithm="stacked",
            )

    with pytest.warns(UserWarning, match="more than 1 paths"):
        output_segments = tp_list_list(
            x,
            y,
            out_batch_shape=(2,),
            d=d,
            output_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            algorithm="sliced",
        )

    expected = jnp.concatenate(
        [
            jnp.einsum(
                "buv,bvw->buw",
                x[:, :6].reshape(2, 2, 3),
                y[:, :6].reshape(2, 3, 2),
            ).reshape(2, 4),
            -0.25
            * jnp.einsum(
                "buv,bvw->buw",
                x[:, 6:].reshape(2, 1, 2),
                y[:, 6:].reshape(2, 2, 1),
            ).reshape(2, 1),
        ],
        axis=-1,
    )
    actual = sum_cat_list_list(d.operands[-1], output_segments, (2,), jnp.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
