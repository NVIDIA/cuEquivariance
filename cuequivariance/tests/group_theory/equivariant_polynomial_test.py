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
    assert repr(poly)
    assert poly < scaled or scaled < poly
    assert scaled == poly * 2.0
    assert poly.used_inputs() == [True, True]
    assert poly.used_outputs() == [True]
    assert poly.used_operands() == [True, True, True]
    assert poly.flop(batch_size=3) > 0
    assert poly.memory([1, 1, 1]) == sum(rep.dim for rep in poly.operands)

    weights = np.array([1.0, 2.0, 3.0, 4.0])
    values = np.array([2.0, -1.0])
    expected = poly(weights, values)[0]
    transformed = [
        poly.apply_fn(lambda operation, stp: (operation, stp)),
        poly.fuse_stps(),
        poly.consolidate(),
        poly.flatten_modes(["i"]),
        poly.canonicalize_subscripts(),
        poly.squeeze_modes(),
        poly.split_mode("i", 1),
        poly.flatten_coefficient_modes(),
        poly.symmetrize_for_identical_operands(),
        poly.unsymmetrize_for_identical_operands(),
    ]
    for equivalent_poly in transformed:
        np.testing.assert_allclose(equivalent_poly(weights, values)[0], expected)

    stacked = cue.EquivariantPolynomial.stack(
        [poly, poly], stacked=[True, False, False]
    )
    assert stacked.num_operands == poly.num_operands
    assert stacked.inputs[0].dim == 2 * poly.inputs[0].dim
    assert stacked.inputs[1] == poly.inputs[1]

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

    dropped = with_unused.filter_drop_unused_operands()
    assert dropped.inputs == poly.inputs
    np.testing.assert_allclose(
        dropped(weights, values)[0], with_unused(weights, values, unused_value)[0]
    )

    kept = with_unused.filter_keep_operands([True, True, False, True])
    np.testing.assert_allclose(
        kept(weights, values)[0], with_unused(weights, values, unused_value)[0]
    )
    assert with_unused.filter_keep_outputs([False]).num_outputs == 0

    compute_none = with_unused.compute_only([False])
    [zero_output] = compute_none(weights, values, unused_value)
    np.testing.assert_array_equal(zero_output, np.zeros_like(zero_output))

    weight_tangent = np.array([0.5])
    value_tangent = np.array([4.0])
    jvp, _ = poly.jvp([True, True])
    [jvp_result] = jvp(weights, values, weight_tangent, value_tangent)
    expected_jvp = poly(weight_tangent, values)[0] + poly(weights, value_tangent)[0]
    np.testing.assert_allclose(jvp_result, expected_jvp)

    cotangent = np.array([1.5])
    transpose_weight, _ = poly.transpose([True, False], [True])
    [weight_gradient] = transpose_weight(values, cotangent)
    expected_weight_gradient = poly(np.ones_like(weights), values)[0] * cotangent[0]
    np.testing.assert_allclose(weight_gradient, expected_weight_gradient)

    transpose_value, _ = poly.transpose([False, True], [True])
    [value_gradient] = transpose_value(weights, cotangent)
    expected_value_gradient = poly(weights, np.ones_like(values))[0] * cotangent[0]
    np.testing.assert_allclose(value_gradient, expected_value_gradient)

    backward, _ = poly.backward([True, True], [True])
    weight_gradient, value_gradient = backward(weights, values, cotangent)
    np.testing.assert_allclose(weight_gradient, expected_weight_gradient)
    np.testing.assert_allclose(value_gradient, expected_value_gradient)
