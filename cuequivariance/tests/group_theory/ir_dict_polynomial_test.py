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


# --------------------------------------------------------------------------
# split_polynomial_by_irreps
# --------------------------------------------------------------------------


def test_split_polynomial_by_irreps_matches_split_operand_by_irrep():
    """The new standalone function should produce the same result as
    EquivariantPolynomial.split_operand_by_irrep."""
    irreps_in = cue.Irreps(cue.O3, "64x0e + 32x1o")
    irreps_sh = cue.Irreps(cue.O3, "0e + 1o")
    irreps_out = cue.Irreps(cue.O3, "0e + 1o + 2e")

    e = cue.descriptors.channelwise_tensor_product(irreps_in, irreps_sh, irreps_out, True)
    old = (
        e.split_operand_by_irrep(2)
        .split_operand_by_irrep(1)
        .split_operand_by_irrep(-1)
        .polynomial
    )

    new = e.polynomial
    new = cue.split_polynomial_by_irreps(new, 2, irreps_sh)
    new = cue.split_polynomial_by_irreps(new, 1, irreps_in)
    new = cue.split_polynomial_by_irreps(new, -1, e.outputs[0].irreps)

    assert old == new


# --------------------------------------------------------------------------
# IrDictPolynomial validation
# --------------------------------------------------------------------------


def test_ir_dict_polynomial_rejects_wrong_operand_count():
    irreps_in = cue.Irreps(cue.O3, "4x0e + 2x1o")
    irreps_out = cue.Irreps(cue.O3, "3x0e")

    result = cue.descriptors.linear_ir_dict(irreps_in, irreps_out)

    with pytest.raises(ValueError, match="input_irreps describe"):
        cue.IrDictPolynomial(
            polynomial=result.polynomial,
            input_irreps=(irreps_in,),  # wrong: should include weight group
            output_irreps=result.output_irreps,
        )


def test_ir_dict_polynomial_rejects_wrong_operand_size():
    irreps_in = cue.Irreps(cue.O3, "4x0e + 2x1o")
    irreps_out = cue.Irreps(cue.O3, "3x0e")

    result = cue.descriptors.linear_ir_dict(irreps_in, irreps_out)

    with pytest.raises(ValueError, match="expected size"):
        cue.IrDictPolynomial(
            polynomial=result.polynomial,
            input_irreps=(
                result.input_irreps[0],
                cue.Irreps(cue.O3, "3x0e + 2x1o"),  # wrong mul for 0e
            ),
            output_irreps=result.output_irreps,
        )


# --------------------------------------------------------------------------
# _ir_dict descriptor variants match the old EquivariantPolynomial path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "irreps1, irreps2, irreps3_filter",
    [
        (
            cue.Irreps(cue.O3, "64x0e + 32x1o"),
            cue.Irreps(cue.O3, "0e + 1o"),
            cue.Irreps(cue.O3, "0e + 1o + 2e"),
        ),
        (
            cue.Irreps(cue.O3, "16x0e + 8x1o + 4x2e"),
            cue.Irreps(cue.O3, "0e + 1o"),
            None,
        ),
        (
            cue.Irreps(cue.SO3, "8x0 + 4x1 + 2x2"),
            cue.Irreps(cue.SO3, "0 + 1"),
            None,
        ),
    ],
)
def test_channelwise_tensor_product_ir_dict(irreps1, irreps2, irreps3_filter):
    # channelwise_tensor_product_ir_dict always simplifies output irreps
    e = cue.descriptors.channelwise_tensor_product(
        irreps1, irreps2, irreps3_filter, simplify_irreps3=True
    )
    old_poly = (
        e.split_operand_by_irrep(2)
        .split_operand_by_irrep(1)
        .split_operand_by_irrep(-1)
        .polynomial
    )

    result = cue.descriptors.channelwise_tensor_product_ir_dict(
        irreps1, irreps2, irreps3_filter
    )

    assert result.polynomial == old_poly
    assert result.output_irreps[0] == e.outputs[0].irreps
    assert result.input_irreps[1] == irreps1
    assert result.input_irreps[2] == irreps2


@pytest.mark.parametrize(
    "irreps1, irreps2, irreps3",
    [
        (
            cue.Irreps(cue.O3, "4x0e + 2x1o"),
            cue.Irreps(cue.O3, "0e + 1o"),
            cue.Irreps(cue.O3, "4x0e + 2x1o"),
        ),
        (
            cue.Irreps(cue.SO3, "8x0 + 4x1"),
            cue.Irreps(cue.SO3, "0 + 1 + 2"),
            cue.Irreps(cue.SO3, "8x0 + 4x1 + 2x2"),
        ),
    ],
)
def test_fully_connected_tensor_product_ir_dict(irreps1, irreps2, irreps3):
    e = cue.descriptors.fully_connected_tensor_product(irreps1, irreps2, irreps3)
    old_poly = (
        e.split_operand_by_irrep(2)
        .split_operand_by_irrep(1)
        .split_operand_by_irrep(-1)
        .polynomial
    )

    result = cue.descriptors.fully_connected_tensor_product_ir_dict(
        irreps1, irreps2, irreps3
    )

    assert result.polynomial == old_poly
    assert result.output_irreps[0] == irreps3


@pytest.mark.parametrize(
    "irreps_in, irreps_out",
    [
        (
            cue.Irreps(cue.O3, "4x0e + 2x1o"),
            cue.Irreps(cue.O3, "3x0e + 5x1o"),
        ),
        (
            cue.Irreps(cue.SO3, "16x0 + 8x1 + 4x2"),
            cue.Irreps(cue.SO3, "8x0 + 4x1"),
        ),
    ],
)
def test_linear_ir_dict(irreps_in, irreps_out):
    e = cue.descriptors.linear(irreps_in, irreps_out)
    old_poly = (
        e.split_operand_by_irrep(1).split_operand_by_irrep(-1).polynomial
    )

    result = cue.descriptors.linear_ir_dict(irreps_in, irreps_out)

    assert result.polynomial == old_poly
    assert result.output_irreps[0] == irreps_out
    assert result.input_irreps[1] == irreps_in


def test_full_tensor_product_ir_dict():
    irreps1 = cue.Irreps(cue.O3, "2x0e + 1x1o")
    irreps2 = cue.Irreps(cue.O3, "0e + 1o")

    e = cue.descriptors.full_tensor_product(irreps1, irreps2)
    old_poly = (
        e.split_operand_by_irrep(1)
        .split_operand_by_irrep(0)
        .split_operand_by_irrep(-1)
        .polynomial
    )

    result = cue.descriptors.full_tensor_product_ir_dict(irreps1, irreps2)

    assert result.polynomial == old_poly
    assert result.input_irreps[0] == irreps1
    assert result.input_irreps[1] == irreps2


def test_elementwise_tensor_product_ir_dict():
    irreps1 = cue.Irreps(cue.O3, "4x0e + 4x1o")
    irreps2 = cue.Irreps(cue.O3, "4x0e + 4x1o")

    e = cue.descriptors.elementwise_tensor_product(irreps1, irreps2)
    old_poly = (
        e.split_operand_by_irrep(1)
        .split_operand_by_irrep(0)
        .split_operand_by_irrep(-1)
        .polynomial
    )

    result = cue.descriptors.elementwise_tensor_product_ir_dict(irreps1, irreps2)

    assert result.polynomial == old_poly


def test_symmetric_contraction_ir_dict():
    irreps_in = 16 * cue.Irreps("SO3", "0 + 1 + 2")
    irreps_out = 16 * cue.Irreps("SO3", "0 + 1")

    e = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, (1, 2, 3))
    old_poly = (
        e.split_operand_by_irrep(1).split_operand_by_irrep(-1).polynomial
    )

    result = cue.descriptors.symmetric_contraction_ir_dict(
        irreps_in, irreps_out, (1, 2, 3)
    )

    assert result.polynomial == old_poly
    (output_irreps,) = result.output_irreps
    assert output_irreps == irreps_out


def test_mace_symmetric_contraction_ir_dict():
    from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
        symmetric_contraction as mace_sc,
        symmetric_contraction_ir_dict as mace_sc_ir_dict,
    )

    irreps_in = 4 * cue.Irreps("SO3", "0 + 1 + 2")
    irreps_out = 4 * cue.Irreps("SO3", "0 + 1")

    e, projection_old = mace_sc(irreps_in, irreps_out, [1, 2, 3])
    old_poly = (
        e.split_operand_by_irrep(1).split_operand_by_irrep(-1).polynomial
    )

    result, projection_new = mace_sc_ir_dict(irreps_in, irreps_out, [1, 2, 3])

    assert result.polynomial == old_poly
    np.testing.assert_array_equal(projection_old, projection_new)
    (output_irreps,) = result.output_irreps
    assert output_irreps == irreps_out


@pytest.mark.parametrize("max_degree", [1, 2, 3, 4])
def test_spherical_harmonics_ir_dict(max_degree):
    ir_vec = cue.O3(1, -1)
    ls = list(range(max_degree + 1))

    e = cue.descriptors.spherical_harmonics(ir_vec, ls)
    old_poly = e.split_operand_by_irrep(-1).polynomial

    result = cue.descriptors.spherical_harmonics_ir_dict(ir_vec, ls)

    assert result.polynomial == old_poly
    (output_irreps,) = result.output_irreps
    assert output_irreps == e.outputs[0].irreps

    # Numpy evaluation: verify output matches unsplit
    vec = np.array([0.3, -0.5, 0.8])
    [out_flat] = e.polynomial(vec)

    out_parts = result.polynomial(vec)
    out_concat = np.concatenate(out_parts)
    np.testing.assert_allclose(out_flat, out_concat, atol=1e-12)


# --------------------------------------------------------------------------
# Numpy evaluation: ir_dict variant produces same results as original
# --------------------------------------------------------------------------


def test_channelwise_numpy_evaluation():
    """Evaluate both paths with numpy and compare outputs."""
    irreps1 = cue.Irreps(cue.O3, "4x0e + 2x1o")
    irreps_sh = cue.Irreps(cue.O3, "0e + 1o")

    e = cue.descriptors.channelwise_tensor_product(irreps1, irreps_sh, simplify_irreps3=True)
    result = cue.descriptors.channelwise_tensor_product_ir_dict(irreps1, irreps_sh)

    # Generate random inputs matching the unsplit polynomial
    np.random.seed(42)
    inputs_orig = [np.random.randn(op.size) for op in e.polynomial.inputs]
    [out_orig] = e.polynomial(*inputs_orig)

    # The split polynomial has more operands — reconstruct matching inputs
    inputs_split = [np.random.randn(op.size) for op in result.polynomial.inputs]

    # Use the same flat data for both
    # Unsplit: [weights, input1_flat, input2_flat]
    # Split:   [weights, input1_ir0, input1_ir1, input2_ir0, input2_ir1]
    w = np.random.randn(result.polynomial.inputs[0].size)
    x1 = np.random.randn(e.polynomial.inputs[1].size)
    x2 = np.random.randn(e.polynomial.inputs[2].size)

    [out_orig] = e.polynomial(w, x1, x2)

    # Split x1 and x2 by irrep boundaries
    x1_parts = []
    offset = 0
    for mul, ir in irreps1:
        size = mul * ir.dim
        x1_parts.append(x1[offset : offset + size])
        offset += size

    x2_parts = []
    offset = 0
    for mul, ir in irreps_sh:
        size = mul * ir.dim
        x2_parts.append(x2[offset : offset + size])
        offset += size

    out_split = result.polynomial(w, *x1_parts, *x2_parts)
    out_split_concat = np.concatenate(out_split)

    np.testing.assert_allclose(out_orig, out_split_concat, atol=1e-12)
