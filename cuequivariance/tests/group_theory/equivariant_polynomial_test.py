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
import numpy as np
import pytest

import cuequivariance as cue


def test_transpose():
    cue.descriptors.transpose(cue.Irreps(cue.O3, "3x1e"), cue.mul_ir, cue.ir_mul)


def test_commutativity_squeeze_flatten():
    irreps1 = cue.Irreps("O3", "32x0e + 32x1o")
    irreps2 = cue.Irreps("O3", "1x0e + 1x1o")
    irreps3 = cue.Irreps("O3", "32x0e + 32x1o")

    poly = cue.descriptors.fully_connected_tensor_product(irreps1, irreps2, irreps3)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )

    poly = cue.descriptors.full_tensor_product(irreps1, irreps2, irreps3)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )

    poly = cue.descriptors.channelwise_tensor_product(irreps1, irreps2, irreps3)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )

    poly = cue.descriptors.linear(irreps1, irreps2)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )


@pytest.mark.parametrize("ell", [1, 2, 3, 4])
def test_spherical_harmonics(ell: int):
    poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [ell])

    vec = np.random.randn(3)
    axis = np.random.randn(3)
    angle = np.random.rand()

    [yl] = poly(vec)

    R = cue.SO3(1).rotation(axis, angle)
    Rl = cue.SO3(ell).rotation(axis, angle)

    [yl1] = poly(R @ vec)
    yl2 = Rl @ yl

    np.testing.assert_allclose(yl1, yl2, rtol=1e-06, atol=1e-13)
    np.testing.assert_allclose(np.sum(yl**2), (2 * ell + 1) * np.sum(vec**2) ** ell)


@pytest.mark.parametrize("ell", [0, 1, 2, 3, 4])
def test_yxy_rotation(ell: int):
    alpha = 0.3
    beta = 0.4
    gamma = -0.5

    irrep = cue.SO3(ell)
    poly = cue.descriptors.yxy_rotation(cue.Irreps("SO3", [irrep]))

    def enc(th: float):
        m = np.arange(1, ell + 1)
        c = np.cos(m * th)
        s = np.sin(m * th)
        return np.concatenate([c[::-1], [1.0], s])

    x = np.random.randn(irrep.dim)
    [y1] = poly(enc(gamma), enc(beta), enc(alpha), x)

    A = irrep.rotation(np.array([0.0, 1.0, 0.0]), alpha)
    B = irrep.rotation(np.array([1.0, 0.0, 0.0]), beta)
    C = irrep.rotation(np.array([0.0, 1.0, 0.0]), gamma)
    y2 = A @ B @ C @ x

    np.testing.assert_allclose(y1, y2)


@pytest.mark.parametrize("ell", [0, 1, 2, 3, 4])
def test_yx_rotation(ell: int):
    phi = 0.3
    theta = 0.4

    irrep = cue.SO3(ell)
    poly = cue.descriptors.yx_rotation(cue.Irreps("SO3", [irrep]))

    def enc(th: float):
        m = np.arange(1, ell + 1)
        c = np.cos(m * th)
        s = np.sin(m * th)
        return np.concatenate([c[::-1], [1.0], s])

    x = np.random.randn(irrep.dim)
    [y1] = poly(enc(phi), enc(theta), x)

    # Rotation around x-axis followed by rotation around y-axis
    Rx = irrep.rotation(np.array([1.0, 0.0, 0.0]), phi)
    Ry = irrep.rotation(np.array([0.0, 1.0, 0.0]), theta)
    y2 = Ry @ Rx @ x

    np.testing.assert_allclose(y1, y2)


def test_elementwise_tensor_product():
    irreps1 = cue.Irreps("O3", "8x0e + 7x1o + 4x2e")
    irreps2 = cue.Irreps("O3", "8x0e + 7x1o + 4x0e")
    irreps3 = cue.Irreps("O3", "0e + 1o + 1e")
    poly = cue.descriptors.elementwise_tensor_product(irreps1, irreps2, irreps3)
    assert (
        poly.squeeze_modes().flatten_coefficient_modes()
        == poly.flatten_coefficient_modes().squeeze_modes()
    )
    assert poly.num_inputs == 2
    assert poly.num_outputs == 1


def test_symmetric_contraction():
    irreps_in = 16 * cue.Irreps("SO3", "0 + 1 + 2")
    irreps_out = 16 * cue.Irreps("SO3", "0 + 1")
    poly = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, [0, 1, 2, 3])
    assert poly.num_inputs == 2
    assert poly.num_outputs == 1

    [_, _, _, (_, d)] = poly.polynomial.operations
    assert d.num_paths == 437

    poly = poly.polynomial.unsymmetrize_for_identical_operands()
    [_, _, _, (_, d)] = poly.operations
    assert d.num_paths == 105


def _add_unused_scalar_input(
    poly: cue.EquivariantPolynomial,
) -> cue.EquivariantPolynomial:
    unused_rep = cue.IrrepsAndLayout(cue.Irreps("SO3", "1x0"), cue.ir_mul)
    operations = [
        (
            cue.Operation(
                [buffer + (buffer >= poly.num_inputs) for buffer in operation.buffers]
            ),
            stp,
        )
        for operation, stp in poly.polynomial.operations
    ]
    polynomial = cue.SegmentedPolynomial(
        [*poly.polynomial.inputs, cue.SegmentedOperand.empty_segments(unused_rep.dim)],
        poly.polynomial.outputs,
        operations,
    )
    return cue.EquivariantPolynomial(
        [*poly.inputs, unused_rep], poly.outputs, polynomial
    )


def test_equivariant_polynomial_wrappers_and_stack():
    irreps = cue.Irreps("SO3", "2x0")
    poly = cue.descriptors.linear(irreps, irreps)
    equivalent = cue.descriptors.linear(irreps, irreps)
    scaled = 2.0 * poly
    assert poly == equivalent
    assert hash(poly) == hash(equivalent)
    assert scaled != poly
    assert not poly < poly
    assert (poly < scaled) != (scaled < poly)
    assert "2x0" in repr(poly)
    assert poly.used_inputs() == [True, True]
    assert poly.used_outputs() == [True]
    assert poly.used_operands() == [True, True, True]
    # Cost convention: four multiply-adds (8) and two output rescalings (2).
    assert poly.flop() == 10
    assert poly.flop(batch_size=3) == 30
    assert poly.memory([2, 3, 5]) == 2 * 4 + 3 * 2 + 5 * 2

    weights = np.array([1.0, 2.0, 3.0, 4.0])
    other_weights = np.array([-2.0, 5.0, 1.0, -3.0])
    values = np.array([2.0, -0.5])
    expected = np.array([0.5, 2.0]) / np.sqrt(2)
    np.testing.assert_allclose(poly(weights, values)[0], expected)
    np.testing.assert_allclose(scaled(weights, values)[0], 2 * expected)
    np.testing.assert_allclose((poly * 2.0)(weights, values)[0], 2 * expected)

    stacked = cue.EquivariantPolynomial.stack([poly, poly], [True, False, False])
    assert stacked.inputs[0].dim == 8
    assert stacked.inputs[1] == poly.inputs[1]
    assert stacked.outputs == poly.outputs
    # Shared outputs add the two independent matrix products, not concatenate them.
    expected_other = np.array([-4.5, 11.5]) / np.sqrt(2)
    [actual] = stacked(np.concatenate([weights, other_weights]), values)
    assert actual.shape == (2,)
    np.testing.assert_allclose(actual, expected + expected_other)


def test_equivariant_polynomial_filtering_and_derivatives():
    scalar = cue.Irreps("SO3", "1x0")
    poly = cue.descriptors.linear(scalar, scalar)
    weights = np.array([3.0])
    values = np.array([-2.0])
    unused_value = np.array([17.0])

    with_unused = _add_unused_scalar_input(poly)
    assert with_unused.used_inputs() == [True, True, False]
    assert with_unused.used_outputs() == [True]
    assert with_unused.used_operands() == [True, True, False, True]

    np.testing.assert_allclose(poly(weights, values)[0], [-6.0])
    for unused in (unused_value, np.array([-23.0])):
        np.testing.assert_allclose(with_unused(weights, values, unused)[0], [-6.0])
    dropped = with_unused.filter_drop_unused_operands()
    assert dropped.inputs == poly.inputs
    np.testing.assert_allclose(dropped(weights, values)[0], [-6.0])

    kept = with_unused.filter_keep_operands([True, True, False, True])
    np.testing.assert_allclose(kept(weights, values)[0], [-6.0])
    assert with_unused.filter_keep_outputs([False]).num_outputs == 0

    compute_none = with_unused.compute_only([False])
    [zero_output] = compute_none(weights, values, unused_value)
    assert zero_output.shape == (1,)
    np.testing.assert_array_equal(zero_output, [0.0])

    weight_tangent = np.array([0.5])
    value_tangent = np.array([4.0])
    jvp, _ = poly.jvp([True, True])
    [jvp_result] = jvp(weights, values, weight_tangent, value_tangent)
    expected_jvp = np.array([11.0])
    np.testing.assert_allclose(jvp_result, expected_jvp)

    cotangent = np.array([1.5])
    transpose_weight, _ = poly.transpose([True, False], [True])
    [weight_gradient] = transpose_weight(values, cotangent)
    expected_weight_gradient = np.array([-3.0])
    np.testing.assert_allclose(weight_gradient, expected_weight_gradient)

    transpose_value, _ = poly.transpose([False, True], [True])
    [value_gradient] = transpose_value(weights, cotangent)
    expected_value_gradient = np.array([4.5])
    np.testing.assert_allclose(value_gradient, expected_value_gradient)

    backward, _ = poly.backward([True, True], [True])
    weight_gradient, value_gradient = backward(weights, values, cotangent)
    np.testing.assert_allclose(weight_gradient, expected_weight_gradient)
    np.testing.assert_allclose(value_gradient, expected_value_gradient)


def test_equivariant_polynomial_rejects_invalid_operands():
    irreps = cue.Irreps("SO3", "2x0")
    poly = cue.descriptors.linear(irreps, irreps)
    with pytest.raises(ValueError, match="Number of inputs"):
        cue.EquivariantPolynomial(poly.inputs[:-1], poly.outputs, poly.polynomial)
    with pytest.raises(ValueError, match="Number of outputs"):
        cue.EquivariantPolynomial(poly.inputs, [], poly.polynomial)

    vector_poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [1])
    rep_only_poly = cue.EquivariantPolynomial(
        [cue.SO3(1)], [cue.SO3(1)], vector_poly.polynomial
    )
    with pytest.raises(ValueError, match="Cannot stack operand 0"):
        cue.EquivariantPolynomial.stack(
            [rep_only_poly, rep_only_poly], stacked=[True, False]
        )

    ordered = cue.Irreps("SO3", "1x0 + 1x1")
    reversed_order = cue.Irreps("SO3", "1x1 + 1x0")
    same_inputs_different_output = cue.descriptors.linear(ordered, reversed_order)
    first = cue.descriptors.linear(ordered, ordered)
    with pytest.raises(ValueError, match="Operand 2 must be the same"):
        cue.EquivariantPolynomial.stack(
            [first, same_inputs_different_output], stacked=[False, False, False]
        )


@pytest.mark.parametrize(
    "method",
    [
        "fuse_stps",
        "consolidate",
        "flatten_modes",
        "canonicalize_subscripts",
        "squeeze_modes",
        "split_mode",
        "apply_fn",
    ],
)
def test_equivariant_polynomial_transforms_change_structure(method):
    irreps = cue.Irreps("SO3", "2x0")
    base = cue.descriptors.linear(irreps, irreps)
    poly = cue.EquivariantPolynomial(
        base.inputs,
        base.outputs,
        cue.SegmentedPolynomial(
            base.polynomial.inputs,
            base.polynomial.outputs,
            list(base.polynomial.operations) * 2,
        ),
    )
    expected = 2 * np.array([0.5, 2.0]) / np.sqrt(2)
    if method == "split_mode":
        poly = poly.squeeze_modes()
        transformed = poly.split_mode("u", 1)
        assert sum(d.num_paths for _, d in transformed.polynomial.operations) == 4
    elif method == "flatten_modes":
        transformed = poly.flatten_modes(["i"])
        assert all(
            "i" not in d.subscripts for _, d in transformed.polynomial.operations
        )
    elif method == "apply_fn":
        transformed = poly.apply_fn(lambda operation, stp: (operation, stp * 3))
        expected *= 3
    else:
        transformed = getattr(poly, method)()
    assert transformed.polynomial != poly.polynomial
    if method in ("fuse_stps", "consolidate"):
        assert len(transformed.polynomial.operations) == 1
    np.testing.assert_allclose(
        transformed(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, -0.5]))[0], expected
    )


def test_equivariant_polynomial_flattens_coefficient_modes():
    d = cue.SegmentedTensorProduct.from_subscripts("u,v+uv")
    coefficients = np.array([[1.0, 2.0], [3.0, -4.0]])
    d.add_path(None, None, c=coefficients)
    rep = cue.IrrepsAndLayout(cue.Irreps("SO3", "2x0"), cue.ir_mul)
    poly = cue.EquivariantPolynomial(
        [rep], [rep], cue.SegmentedPolynomial.eval_last_operand(d)
    )
    flattened = poly.flatten_coefficient_modes()
    assert flattened.polynomial != poly.polynomial
    assert all(
        not stp.coefficient_subscripts for _, stp in flattened.polynomial.operations
    )
    np.testing.assert_allclose(flattened(np.array([2.0, -0.5]))[0], [0.5, 6.0])


@pytest.mark.parametrize(
    "method",
    ["symmetrize_for_identical_operands", "unsymmetrize_for_identical_operands"],
)
def test_equivariant_polynomial_repeated_operands_symmetry(method):
    d = cue.SegmentedTensorProduct.from_subscripts("i,j,k+ijk")
    d.add_path(None, None, None, c=np.arange(1.0, 5.0).reshape(2, 2, 1))
    d = d.flatten_coefficient_modes()
    rep = cue.IrrepsAndLayout(cue.Irreps("SO3", "2x0"), cue.ir_mul)
    scalar = cue.IrrepsAndLayout(cue.Irreps("SO3", "0"), cue.ir_mul)
    poly = cue.EquivariantPolynomial(
        [rep],
        [scalar],
        cue.SegmentedPolynomial(
            [d.operands[0]], [d.operands[-1]], [(cue.Operation([0, 0, 1]), d)]
        ),
    )
    transformed = getattr(poly, method)()
    assert transformed.polynomial != poly.polynomial
    # x.T @ [[1,2],[3,4]] @ x at x=[2, .5] is 4+2+3+1=10.
    np.testing.assert_allclose(poly(np.array([2.0, 0.5]))[0], [10.0])
    np.testing.assert_allclose(transformed(np.array([2.0, 0.5]))[0], [10.0])


@pytest.mark.parametrize(
    "mask,tangent,expected", [([True, False], 0.5, -1.0), ([False, True], 4.0, 12.0)]
)
def test_equivariant_polynomial_partial_tangents(mask, tangent, expected):
    scalar = cue.Irreps("SO3", "0")
    poly = cue.descriptors.linear(scalar, scalar)
    derivative, mapping = poly.jvp(mask)
    mapped_inputs, mapped_outputs = mapping((poly.inputs, poly.outputs))
    assert tuple(mapped_inputs) == derivative.inputs
    assert tuple(mapped_outputs) == derivative.outputs
    [actual] = derivative(np.array([3.0]), np.array([-2.0]), np.array([tangent]))
    np.testing.assert_allclose(actual, [expected])


def test_equivariant_polynomial_cubic_derivative():
    d = cue.SegmentedTensorProduct.from_subscripts("i,i,i,i")
    d.add_path(None, None, None, None, c=1.0, dims={"i": 1})
    scalar = cue.IrrepsAndLayout(cue.Irreps("SO3", "0"), cue.ir_mul)
    poly = cue.EquivariantPolynomial(
        [scalar],
        [scalar],
        cue.SegmentedPolynomial(
            [d.operands[0]], [d.operands[-1]], [(cue.Operation([0, 0, 0, 1]), d)]
        ),
    )
    np.testing.assert_allclose(poly(np.array([2.0]))[0], [8.0])
    derivative, _ = poly.jvp([True])
    np.testing.assert_allclose(derivative(np.array([2.0]), np.array([0.5]))[0], [6.0])
    backward, _ = poly.backward([True], [True])
    np.testing.assert_allclose(backward(np.array([2.0]), np.array([1.5]))[0], [18.0])
