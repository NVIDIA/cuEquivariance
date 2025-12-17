# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cuequivariance as cue


def make_simple_stp() -> cue.SegmentedTensorProduct:
    d = cue.SegmentedTensorProduct.empty_segments([2, 2, 2])
    d.add_path(0, 0, 0, c=1.0)
    d.add_path(1, 1, 1, c=-2.0)
    return d


def make_simple_dot_product_stp() -> cue.SegmentedTensorProduct:
    d = cue.SegmentedTensorProduct.from_subscripts("i,j,k+ijk")
    i0 = d.add_segment(0, (3,))
    i1 = d.add_segment(1, (3,))
    i2 = d.add_segment(2, (1,))
    d.add_path(i0, i1, i2, c=np.eye(3).reshape(3, 3, 1))
    return d


def test_init_segmented_polynomial():
    """Test initialization of SegmentedPolynomial."""
    stp = make_simple_stp()
    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    assert poly.num_inputs == 2 and poly.num_outputs == 1 and poly.num_operands == 3
    assert len(poly.operations) == 1
    assert poly.operations[0] == (cue.Operation((0, 1, 2)), stp)


def test_polynomial_equality():
    """Test equality comparison of polynomials."""
    stp1 = make_simple_stp()
    stp2 = make_simple_stp()

    poly1 = cue.SegmentedPolynomial.eval_last_operand(stp1)
    poly2 = cue.SegmentedPolynomial.eval_last_operand(stp2)
    poly3 = cue.SegmentedPolynomial.eval_last_operand(2 * stp2)

    assert poly1 == poly2 and poly1 != poly3 and poly1 < poly3


def test_call_function():
    """Test calling the polynomial as a function."""
    stp = make_simple_dot_product_stp()
    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    [result] = poly(a, b)
    assert np.allclose(result, np.array([a.dot(b)]))


def test_buffer_properties():
    """Test properties related to buffer sizes and usage."""
    stp1 = make_simple_stp()
    op1 = cue.Operation((0, 1, 2))

    stp2 = cue.SegmentedTensorProduct.empty_segments([2, 1])
    stp2.add_path(0, 0, c=1.0)
    op2 = cue.Operation((0, 3))

    poly = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),
        ],
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(1),
        ],
        [(op1, stp1), (op2, stp2)],
    )

    assert [ope.size for ope in poly.operands] == [2, 2, 2, 1]
    assert poly.used_operands() == [True, True, True, True]


def test_remove_unused_buffers():
    """Test removing unused buffers from the polynomial."""
    stp = make_simple_stp()
    op = cue.Operation((0, 2, 3))  # Buffer 1 is not used

    poly = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),
        ],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op, stp)],
    )

    assert poly.used_operands() == [True, False, True, True]

    cleaned = poly.filter_drop_unsued_operands()
    assert cleaned.num_inputs == 2 and cleaned.num_outputs == 1
    assert cleaned.used_operands() == [True, True, True]


def test_consolidate():
    """Test consolidating tensor products."""
    stp1 = make_simple_stp()
    stp2 = make_simple_stp()
    op = cue.Operation((0, 1, 2))

    poly = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),
        ],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op, stp1), (op, stp2)],
    )

    consolidated = poly.consolidate()
    assert len(consolidated.operations) == 1
    assert len(consolidated.operations[0][1].paths) == 2
    assert consolidated.operations[0][1].paths[0].coefficients == 2.0
    assert consolidated.operations[0][1].paths[1].coefficients == -4.0


def test_stack():
    """Test stacking polynomials."""
    stp1 = cue.SegmentedTensorProduct.empty_segments([2, 2, 1])
    stp1.add_path(0, 0, 0, c=1.0)
    poly1 = cue.SegmentedPolynomial.eval_last_operand(stp1)

    stp2 = cue.SegmentedTensorProduct.empty_segments([2, 2, 1])
    stp2.add_path(0, 0, 0, c=2.0)
    poly2 = cue.SegmentedPolynomial.eval_last_operand(stp2)

    stacked = cue.SegmentedPolynomial.stack([poly1, poly2], [False, False, True])
    assert stacked.num_inputs == 2 and stacked.num_outputs == 1
    assert [ope.size for ope in stacked.operands] == [2, 2, 2]


def test_flops_and_memory():
    """Test computation of FLOPS and memory usage."""
    stp = make_simple_stp()
    op = cue.Operation((0, 1, 2))
    poly = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),
        ],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op, stp)],
    )

    assert poly.flop(batch_size=100) > 0
    assert poly.memory([100, 100, 100]) == 100 * (2 + 2 + 2)


def test_jvp():
    """Test Jacobian-vector product computation."""
    stp = make_simple_dot_product_stp()
    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    x_tangent = np.array([0.1, 0.2, 0.3])
    y_tangent = np.array([0.4, 0.5, 0.6])

    # Test with both inputs having tangents
    jvp_poly, map = poly.jvp([True, True])
    assert map(([0, 1], [2])) == ([0, 1, 0, 1], [2])

    jvp_result = jvp_poly(x, y, x_tangent, y_tangent)
    expected_jvp = np.array([y.dot(x_tangent) + x.dot(y_tangent)])
    assert np.allclose(jvp_result[0], expected_jvp)

    # Test with only x having tangent
    jvp_x_only, _ = poly.jvp([True, False])
    x_only_result = jvp_x_only(x, y, x_tangent)
    assert np.allclose(x_only_result[0], np.array([y.dot(x_tangent)]))

    # Test with only y having tangent
    jvp_y_only, _ = poly.jvp([False, True])
    y_only_result = jvp_y_only(x, y, y_tangent)
    assert np.allclose(y_only_result[0], np.array([x.dot(y_tangent)]))


def test_transpose_linear():
    """Test transposing a linear polynomial."""
    stp = make_simple_dot_product_stp()
    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    cotangent = np.array([2.0])

    # Test transpose w.r.t. x
    transpose_x, _ = poly.transpose(
        is_undefined_primal=[True, False], has_cotangent=[True]
    )
    x_result = transpose_x(y, cotangent)
    assert np.allclose(x_result[0], y * cotangent[0])

    # Test transpose w.r.t. y
    transpose_y, _ = poly.transpose(
        is_undefined_primal=[False, True], has_cotangent=[True]
    )
    y_result = transpose_y(x, cotangent)
    assert np.allclose(y_result[0], x * cotangent[0])


def test_transpose_nonlinear():
    """Test transposing a non-linear polynomial raises an error."""
    stp = make_simple_stp()
    op = cue.Operation((0, 0, 1))  # Using same buffer twice (x^2)
    poly = cue.SegmentedPolynomial(
        [cue.SegmentedOperand.empty_segments(2)],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op, stp)],
    )

    with np.testing.assert_raises(ValueError):
        poly.transpose(is_undefined_primal=[True], has_cotangent=[True])


def test_backward():
    """Test the backward method for gradient computation."""
    stp = make_simple_dot_product_stp()
    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    cotangent = np.array([2.0])

    # Test backward for both inputs
    backward_both, _ = poly.backward(
        requires_gradient=[True, True], has_cotangent=[True]
    )
    grad_x, grad_y = backward_both(x, y, cotangent)
    assert np.allclose(grad_x, y * cotangent[0]) and np.allclose(
        grad_y, x * cotangent[0]
    )

    # Test backward for x only
    backward_x, _ = poly.backward(requires_gradient=[True, False], has_cotangent=[True])
    [grad_x_only] = backward_x(x, y, cotangent)
    assert np.allclose(grad_x_only, y * cotangent[0])

    # Test backward for y only
    backward_y, _ = poly.backward(requires_gradient=[False, True], has_cotangent=[True])
    [grad_y_only] = backward_y(x, y, cotangent)
    assert np.allclose(grad_y_only, x * cotangent[0])

    # Test with zero cotangent
    grad_x_zero, grad_y_zero = backward_both(x, y, np.array([0.0]))
    assert np.allclose(grad_x_zero, np.zeros_like(x)) and np.allclose(
        grad_y_zero, np.zeros_like(y)
    )


def test_symmetrize_identical_operands():
    """Test symmetrization and unsymmetrization of polynomials with identical operands."""
    stp = cue.SegmentedTensorProduct.empty_segments([2, 2, 1])
    stp.add_path(0, 1, 0, c=1.0)

    op = cue.Operation((0, 0, 1))
    poly = cue.SegmentedPolynomial(
        [cue.SegmentedOperand.empty_segments(2)],
        [cue.SegmentedOperand.empty_segments(1)],
        [(op, stp)],
    )

    sym_poly = poly.symmetrize_for_identical_operands()
    [(_, sym_stp)] = sym_poly.operations

    assert len(sym_stp.paths) == 2
    assert sym_stp.paths[0].coefficients == sym_stp.paths[1].coefficients == 0.5
    assert sym_stp.paths[0].indices == (0, 1, 0) and sym_stp.paths[1].indices == (
        1,
        0,
        0,
    )

    unsym_poly = sym_poly.unsymmetrize_for_identical_operands()
    [(_, unsym_stp)] = unsym_poly.operations
    assert len(unsym_stp.paths) == 1 and unsym_stp.paths[0].coefficients == 1.0

    x = np.array([1.0, 2.0])
    assert np.allclose(poly(x)[0], sym_poly(x)[0])


def test_stack_tensor_products():
    """Test stacking tensor products together."""
    stp1 = cue.SegmentedTensorProduct.empty_segments([2, 2, 2])
    stp1.add_path(0, 0, 0, c=1.0)

    stp2 = cue.SegmentedTensorProduct.empty_segments([2, 2, 1])
    stp2.add_path(0, 1, 0, c=2.0)

    in1 = cue.SegmentedOperand.empty_segments(2)
    in2 = cue.SegmentedOperand.empty_segments(2)
    poly = cue.SegmentedPolynomial.stack_tensor_products(
        [in1, in2], [None], [([0, 1, 2], stp1), ([0, 1, 2], stp2)]
    )

    assert poly.num_inputs == 2 and poly.num_outputs == 1 and poly.num_operands == 3
    assert len(poly.operations) == 1
    assert poly.outputs[0].num_segments == 3


def test_concatenate():
    """Test concatenating segmented polynomials."""
    stp1 = cue.SegmentedTensorProduct.empty_segments([2, 1])
    stp1.add_path(0, 0, c=1.0)
    poly1 = cue.SegmentedPolynomial(
        [cue.SegmentedOperand.empty_segments(2)],
        [cue.SegmentedOperand.empty_segments(1)],
        [(cue.Operation((0, 1)), stp1)],
    )

    stp2 = cue.SegmentedTensorProduct.empty_segments([2, 1])
    stp2.add_path(0, 0, c=2.0)
    poly2 = cue.SegmentedPolynomial(
        [cue.SegmentedOperand.empty_segments(2)],
        [cue.SegmentedOperand.empty_segments(1)],
        [(cue.Operation((0, 1)), stp2)],
    )

    [in1] = poly1.inputs
    [out1] = poly1.outputs
    [out2] = poly2.outputs

    combined = cue.SegmentedPolynomial.concatenate(
        [in1], [out1, out2], [(poly1, [0, 1, None]), (poly2, [0, None, 1])]
    )

    assert combined.num_inputs == 1 and combined.num_outputs == 2
    assert len(combined.operations) == 2

    x = np.array([1.0, 2.0])
    y1, y2 = combined(x)
    assert np.isclose(y1[0], x[0]) and np.isclose(y2[0], 2.0 * x[0])


def test_filter_keep_outputs():
    """Test filtering to keep only selected outputs."""
    in_op = cue.SegmentedOperand.empty_segments(3)
    out1 = cue.SegmentedOperand.empty_segments(2)
    out2 = cue.SegmentedOperand.empty_segments(1)

    stp1 = cue.SegmentedTensorProduct.empty_segments([3, 2])
    stp1.add_path(0, 0, c=1.0)
    stp1.add_path(1, 1, c=1.0)
    op1 = cue.Operation((0, 1))

    stp2 = cue.SegmentedTensorProduct.empty_segments([3, 1])
    stp2.add_path(2, 0, c=2.0)
    op2 = cue.Operation((0, 2))

    poly = cue.SegmentedPolynomial([in_op], [out1, out2], [(op1, stp1), (op2, stp2)])
    filtered = poly.filter_keep_outputs([True, False])

    assert filtered.num_inputs == 1 and filtered.num_outputs == 1
    assert len(filtered.operations) == 1

    test_input = np.array([1.0, 2.0, 3.0])
    [result] = filtered(test_input)
    assert (
        result.shape == (2,)
        and np.isclose(result[0], test_input[0])
        and np.isclose(result[1], test_input[1])
    )


def test_fuse_stps():
    """Test fusing segmented tensor products with identical operations."""
    input_op = cue.SegmentedOperand.empty_segments(2)
    output_op = cue.SegmentedOperand.empty_segments(1)

    stp1 = cue.SegmentedTensorProduct.empty_segments([2, 1])
    stp1.add_path(0, 0, c=1.0)

    stp2 = cue.SegmentedTensorProduct.empty_segments([2, 1])
    stp2.add_path(0, 0, c=2.0)

    op = cue.Operation((0, 1))
    poly = cue.SegmentedPolynomial([input_op], [output_op], [(op, stp1), (op, stp2)])

    assert len(poly.operations) == 2

    fused = poly.fuse_stps()
    assert len(fused.operations) == 1

    fused_op, fused_stp = fused.operations[0]
    assert fused_stp.paths[0].coefficients == 3.0

    test_input = np.array([1.0, 2.0])
    assert np.allclose(poly(test_input), fused(test_input))


def test_compute_only():
    """Test creating a polynomial that only computes selected outputs."""
    input_op = cue.SegmentedOperand.empty_segments(2)
    output1 = cue.SegmentedOperand.empty_segments(1)
    output2 = cue.SegmentedOperand.empty_segments(1)

    stp1 = cue.SegmentedTensorProduct.empty_segments([2, 1])
    stp1.add_path(0, 0, c=1.0)
    op1 = cue.Operation((0, 1))

    stp2 = cue.SegmentedTensorProduct.empty_segments([2, 1])
    stp2.add_path(1, 0, c=2.0)
    op2 = cue.Operation((0, 2))

    poly = cue.SegmentedPolynomial(
        [input_op], [output1, output2], [(op1, stp1), (op2, stp2)]
    )
    filtered = poly.compute_only([False, True])

    assert (
        filtered.num_inputs == poly.num_inputs
        and filtered.num_outputs == poly.num_outputs
    )
    assert len(filtered.operations) == 1 and filtered.operations[0][0] == op2

    test_input = np.array([1.0, 2.0])
    full_output = poly(test_input)
    filtered_output = filtered(test_input)

    assert np.all(filtered_output[0] == 0)
    assert np.array_equal(filtered_output[1], full_output[1])


def test_permute_inputs_and_outputs():
    """Test permuting inputs and outputs of a SegmentedPolynomial."""
    # Create 3 inputs and 3 outputs
    inputs = [
        cue.SegmentedOperand.empty_segments(2),  # in1
        cue.SegmentedOperand.empty_segments(3),  # in2
        cue.SegmentedOperand.empty_segments(4),  # in3
    ]
    outputs = [
        cue.SegmentedOperand.empty_segments(1),  # out1
        cue.SegmentedOperand.empty_segments(1),  # out2
        cue.SegmentedOperand.empty_segments(1),  # out3
    ]

    # Create operations: each input i maps to output i
    operations = []
    for i in range(3):
        stp = cue.SegmentedTensorProduct.empty_segments([inputs[i].size, 1])
        stp.add_path(i, 0, c=(i + 1.0))
        operations.append((cue.Operation((i, i + 3)), stp))

    poly = cue.SegmentedPolynomial(inputs, outputs, operations)

    # Test data
    x1 = np.array([1.0, 2.0])
    x2 = np.array([3.0, 4.0, 5.0])
    x3 = np.array([6.0, 7.0, 8.0, 9.0])

    # Original results
    y1, y2, y3 = poly(x1, x2, x3)

    # Test input permutation [2,0,1]
    perm_in = poly.permute_inputs([2, 0, 1])
    y1_in, y2_in, y3_in = perm_in(x3, x1, x2)  # Inputs permuted

    # Structure checks
    assert [op.size for op in perm_in.inputs] == [4, 2, 3]
    assert perm_in.operations[0][0].buffers[0] == 1  # Buffer references updated
    assert perm_in.operations[1][0].buffers[0] == 2
    assert perm_in.operations[2][0].buffers[0] == 0

    # Output checks - results should match
    assert np.isclose(y1_in, y1)
    assert np.isclose(y2_in, y2)
    assert np.isclose(y3_in, y3)

    # Test output permutation [1,2,0]
    perm_out = poly.permute_outputs([1, 2, 0])
    y2_out, y3_out, y1_out = perm_out(x1, x2, x3)  # Outputs permuted

    # Structure checks
    assert [op.size for op in perm_out.outputs] == [1, 1, 1]
    assert perm_out.operations[0][0].buffers[1] == 3  # Buffer indices preserved
    assert perm_out.operations[1][0].buffers[1] == 4
    assert perm_out.operations[2][0].buffers[1] == 5

    # Output checks - results should match but in different order
    assert np.isclose(y1_out, y1)
    assert np.isclose(y2_out, y2)
    assert np.isclose(y3_out, y3)


def test_split_operand_by():
    """Test splitting operands by size and segment."""

    # ========================================
    # Example 1: Linear polynomial with contractions
    # ========================================

    # Create a polynomial with 4 segments in first input, 3 in second input
    stp = cue.SegmentedTensorProduct.from_subscripts("u,u,")

    # First input: 4 segments of shape (16,) each
    stp.add_segment(0, (16,))  # segment 0
    stp.add_segment(0, (16,))  # segment 1
    stp.add_segment(0, (16,))  # segment 2
    stp.add_segment(0, (16,))  # segment 3

    # Second input: 3 segments of shape (16,) each
    stp.add_segment(1, (16,))  # segment 0
    stp.add_segment(1, (16,))  # segment 1
    stp.add_segment(1, (16,))  # segment 2

    # Output: 1 segment (scalar)
    stp.add_segment(2, ())

    # Add paths (contractions between segments)
    stp.add_path(0, 0, 0, c=1.0)  # input0[seg0] * input1[seg0]
    stp.add_path(1, 0, 0, c=2.0)  # input0[seg1] * input1[seg0]
    stp.add_path(2, 1, 0, c=3.0)  # input0[seg2] * input1[seg1]
    stp.add_path(3, 2, 0, c=4.0)  # input0[seg3] * input1[seg2]

    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    # Test data
    x = np.random.rand(64)  # 4 segments × 16 = 64
    y = np.random.rand(48)  # 3 segments × 16 = 48

    # Original evaluation
    [result_original] = poly(x, y)

    # ========================================
    # Test 1a: Split by size
    # ========================================

    # Split first input at every 16 elements (segment boundaries)
    poly_split_size = poly.split_operand_by_size(0, [0, 16, 32, 48, 64])

    # Now we have 5 inputs: 4 split parts + original second input
    assert poly_split_size.num_inputs == 5

    # Evaluate with split inputs
    [result_split_size] = poly_split_size(
        x[0:16],  # first segment
        x[16:32],  # second segment
        x[32:48],  # third segment
        x[48:64],  # fourth segment
        y,  # second input unchanged
    )

    # Results should match
    np.testing.assert_allclose(result_original, result_split_size)

    # ========================================
    # Test 1b: Split by segment
    # ========================================

    # Split first input into two groups: segments [0,1] and segments [2,3]
    poly_split_seg = poly.split_operand_by_segment(0, [0, 2, 4])

    # Now we have 3 inputs: 2 split parts + original second input
    assert poly_split_seg.num_inputs == 3

    # Evaluate with split inputs
    [result_split_seg] = poly_split_seg(
        x[0:32],  # segments 0 and 1 (2 × 16 = 32)
        x[32:64],  # segments 2 and 3 (2 × 16 = 32)
        y,  # second input unchanged
    )

    # Results should match
    np.testing.assert_allclose(result_original, result_split_seg)

    # ========================================
    # Example 2: Quadratic polynomial (x² terms)
    # ========================================

    # Create a polynomial that computes quadratic terms
    stp_quad = cue.SegmentedTensorProduct.from_subscripts("u,u,")

    # Input has 2 segments of shape (8,) each
    stp_quad.add_segment(0, (8,))  # segment 0
    stp_quad.add_segment(0, (8,))  # segment 1

    # For quadratic terms, we need the same segments again
    stp_quad.add_segment(1, (8,))  # segment 0
    stp_quad.add_segment(1, (8,))  # segment 1

    # Output: 1 segment (scalar)
    stp_quad.add_segment(2, ())

    # Add paths for quadratic terms
    stp_quad.add_path(0, 0, 0, c=1.0)  # x[0] * x[0]
    stp_quad.add_path(1, 1, 0, c=2.0)  # x[1] * x[1]
    stp_quad.add_path(0, 1, 0, c=3.0)  # x[0] * x[1]
    stp_quad.add_path(1, 0, 0, c=4.0)  # x[1] * x[0]

    # Create polynomial where both inputs are the same operand
    poly_quad = cue.SegmentedPolynomial(
        [stp_quad.operands[0]],  # single input
        [stp_quad.operands[-1]],  # single output
        [((0, 0, 1), stp_quad)],  # use input 0 for both positions
    )

    # Test data
    x_quad = np.random.rand(16)  # 2 segments × 8 = 16

    # Original evaluation
    [result_quad_original] = poly_quad(x_quad)

    # ========================================
    # Test 2a: Split quadratic by size
    # ========================================

    # Split at segment boundary (size 8)
    poly_quad_split_size = poly_quad.split_operand_by_size(0, [0, 8, 16])

    # Now we have 2 inputs (the split parts)
    assert poly_quad_split_size.num_inputs == 2

    # Evaluate with split inputs
    [result_quad_split_size] = poly_quad_split_size(
        x_quad[0:8],  # first segment
        x_quad[8:16],  # second segment
    )

    # Results should match
    np.testing.assert_allclose(result_quad_original, result_quad_split_size)

    # ========================================
    # Test 2b: Split quadratic by segment
    # ========================================

    # Split into individual segments
    poly_quad_split_seg = poly_quad.split_operand_by_segment(0, [0, 1, 2])

    # Now we have 2 inputs (one per segment)
    assert poly_quad_split_seg.num_inputs == 2

    # Evaluate with split inputs
    [result_quad_split_seg] = poly_quad_split_seg(
        x_quad[0:8],  # segment 0
        x_quad[8:16],  # segment 1
    )

    # Results should match
    np.testing.assert_allclose(result_quad_original, result_quad_split_seg)

    # ========================================
    # Example 3: Demonstrating alignment of size and segment splits
    # ========================================

    # When we split at segment boundaries, size and segment methods give same result
    x_test = np.random.rand(64)
    y_test = np.random.rand(48)

    # Split at segment boundary (32 = 2 segments × 16)
    poly_by_size = poly.split_operand_by_size(0, [0, 32, 64])
    poly_by_segment = poly.split_operand_by_segment(0, [0, 2, 4])

    # Both should produce same structure
    assert poly_by_size.num_inputs == 3
    assert poly_by_segment.num_inputs == 3

    # Evaluate both
    [result_by_size] = poly_by_size(x_test[:32], x_test[32:], y_test)
    [result_by_segment] = poly_by_segment(x_test[:32], x_test[32:], y_test)

    # Results should be identical
    np.testing.assert_allclose(result_by_size, result_by_segment)
