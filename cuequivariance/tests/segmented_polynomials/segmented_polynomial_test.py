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

    assert poly.num_inputs == 2
    assert poly.num_outputs == 1
    assert poly.num_operands == 3
    assert len(poly.operations) == 1
    assert poly.operations[0] == (cue.Operation((0, 1, 2)), stp)


def test_polynomial_equality():
    """Test equality comparison of polynomials."""
    stp1 = make_simple_stp()
    stp2 = make_simple_stp()

    poly1 = cue.SegmentedPolynomial.eval_last_operand(stp1)
    poly2 = cue.SegmentedPolynomial.eval_last_operand(stp2)
    poly3 = cue.SegmentedPolynomial.eval_last_operand(2 * stp2)

    assert poly1 == poly2
    assert poly1 != poly3
    assert poly1 < poly3  # Test less than operator


def test_call_function():
    """Test calling the polynomial as a function."""
    # Create a simple bilinear form: f(a, b) = a^T * b
    # For this specific test, we need a particular structure
    stp = cue.SegmentedTensorProduct.from_subscripts("i,j,k+ijk")
    i0 = stp.add_segment(0, (3,))
    i1 = stp.add_segment(1, (3,))
    i2 = stp.add_segment(2, (1,))
    stp.add_path(i0, i1, i2, c=np.eye(3).reshape(3, 3, 1))

    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    # Test evaluation
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    [result] = poly(a, b)
    expected = np.array([a.dot(b)])  # Dot product

    assert np.allclose(result, expected)


def test_buffer_properties():
    """Test properties related to buffer sizes and usage."""
    stp1 = make_simple_stp()
    op1 = cue.Operation((0, 1, 2))

    # Create a second STP with different structure for testing multiple buffers
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

    # Test buffer properties
    assert [ope.size for ope in poly.operands] == [2, 2, 2, 1]

    assert poly.used_buffers() == [True, True, True, True]


def test_remove_unused_buffers():
    """Test removing unused buffers from the polynomial."""
    stp = make_simple_stp()
    # Use operation that doesn't use buffer 1
    op = cue.Operation((0, 2, 3))  # Note: buffer 1 is not used

    poly = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),  # unused
            cue.SegmentedOperand.empty_segments(2),
        ],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op, stp)],
    )

    # Buffer 1 is not used
    assert poly.used_buffers() == [True, False, True, True]

    # Remove unused buffer
    cleaned_poly = poly.remove_unused_buffers()

    assert cleaned_poly.num_inputs == 2
    assert cleaned_poly.num_outputs == 1
    assert cleaned_poly.used_buffers() == [True, True, True]


def test_consolidate():
    """Test consolidating tensor products."""
    stp1 = make_simple_stp()
    stp2 = make_simple_stp()

    op = cue.Operation((0, 1, 2))

    # Create a polynomial with duplicate operations
    poly = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),
        ],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op, stp1), (op, stp2)],
    )

    # Consolidate the polynomial
    consolidated = poly.consolidate()

    # Should have fused the two tensor products
    assert len(consolidated.operations) == 1
    # Coefficients should have been combined for each path
    assert len(consolidated.operations[0][1].paths) == 2
    # The coefficients should have been added
    assert consolidated.operations[0][1].paths[0].coefficients == 2.0
    assert consolidated.operations[0][1].paths[1].coefficients == -4.0


def test_stack():
    """Test stacking polynomials."""
    # Create two simple polynomials using make_simple_stp
    stp = make_simple_stp()
    op1 = cue.Operation((0, 1, 2))
    poly1 = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),
        ],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op1, stp)],
    )

    stp2 = make_simple_stp()
    op2 = cue.Operation((0, 1, 2))
    poly2 = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
            cue.SegmentedOperand.empty_segments(2),
        ],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op2, stp2)],
    )

    # Stack the polynomials with the output being stacked
    stacked = cue.SegmentedPolynomial.stack([poly1, poly2], [False, False, True])

    assert stacked.num_inputs == 2
    assert stacked.num_outputs == 1

    assert [ope.size for ope in stacked.operands] == [2, 2, 4]

    [(_, stp)] = stacked.operations
    assert stp.operands[0].num_segments == 2
    assert stp.operands[1].num_segments == 2
    assert stp.operands[2].num_segments == 4
    assert stp.num_paths == 4
    assert stp.paths[0].indices == (0, 0, 0)
    assert stp.paths[1].indices == (0, 0, 2 + 0)
    assert stp.paths[2].indices == (1, 1, 1)
    assert stp.paths[3].indices == (1, 1, 2 + 1)


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
    # Test FLOPS calculation
    flops = poly.flop(batch_size=100)
    assert flops > 0

    # Test memory calculation
    memory = poly.memory([100, 100, 100])
    assert memory == 100 * (2 + 2 + 2)  # All operands have size 2


def test_jvp():
    """Test Jacobian-vector product computation."""
    # Create a simple polynomial for testing: f(x,y) = x^T * y (dot product)
    stp = make_simple_dot_product_stp()
    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    # Input values
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    # Tangent vectors (directions for differentiation)
    x_tangent = np.array([0.1, 0.2, 0.3])
    y_tangent = np.array([0.4, 0.5, 0.6])

    # Create the JVP polynomial for both inputs having tangents
    jvp_poly = poly.jvp([True, True])

    # When both inputs have tangents, we need to concatenate inputs and tangents
    # The JVP polynomial expects inputs followed by their respective tangents
    jvp_result = jvp_poly(x, y, x_tangent, y_tangent)

    # For the dot product function f(x,y) = x^T * y:
    # The Jacobian w.r.t x is y^T, and the Jacobian w.r.t y is x^T
    # So Jvp = y^T * x_tangent + x^T * y_tangent
    expected_jvp = np.array([y.dot(x_tangent) + x.dot(y_tangent)])

    assert np.allclose(jvp_result[0], expected_jvp)

    # Test with only x having a tangent
    jvp_x_only = poly.jvp([True, False])
    x_only_result = jvp_x_only(x, y, x_tangent)
    expected_x_only = np.array([y.dot(x_tangent)])
    assert np.allclose(x_only_result[0], expected_x_only)

    # Test with only y having a tangent
    jvp_y_only = poly.jvp([False, True])
    y_only_result = jvp_y_only(x, y, y_tangent)
    expected_y_only = np.array([x.dot(y_tangent)])
    assert np.allclose(y_only_result[0], expected_y_only)


def test_transpose_linear():
    """Test transposing a linear polynomial."""
    # Create a linear polynomial f(x, y) = Ax where A is a matrix
    # Here we use f(x, y) = x^T * y (dot product)
    # This is linear in both x and y
    stp = make_simple_dot_product_stp()
    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    # Input values
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    # x dot y = 1*4 + 2*5 + 3*6 = 32

    # Cotangent for the output
    cotangent = np.array([2.0])

    # Test transposing with respect to x (x is undefined primal)
    # is_undefined_primal = [True, False] means x is undefined, y is defined
    # has_cotangent = [True] means the output has a cotangent
    transpose_x = poly.transpose(
        is_undefined_primal=[True, False], has_cotangent=[True]
    )

    # The transpose polynomial should compute the gradient of the output w.r.t x
    # For f(x, y) = x^T * y, the gradient w.r.t x is y
    # So transpose_x(y, cotangent) should be y * cotangent
    x_result = transpose_x(y, cotangent)
    expected_x_result = y * cotangent[0]
    assert np.allclose(x_result[0], expected_x_result)

    # Test transposing with respect to y (y is undefined primal)
    transpose_y = poly.transpose(
        is_undefined_primal=[False, True], has_cotangent=[True]
    )

    # For f(x, y) = x^T * y, the gradient w.r.t y is x
    # So transpose_y(x, cotangent) should be x * cotangent
    y_result = transpose_y(x, cotangent)
    expected_y_result = x * cotangent[0]
    assert np.allclose(y_result[0], expected_y_result)


def test_transpose_nonlinear():
    """Test transposing a non-linear polynomial raises an error."""
    # Create a non-linear polynomial
    stp = make_simple_stp()
    op = cue.Operation((0, 0, 1))  # Note: using the same buffer twice (x^2)
    poly = cue.SegmentedPolynomial(
        [
            cue.SegmentedOperand.empty_segments(2),
        ],
        [cue.SegmentedOperand.empty_segments(2)],
        [(op, stp)],
    )

    # Try to transpose the non-linear polynomial
    # This should raise a ValueError since there are multiple undefined primals
    # (the same input buffer is used twice)
    with np.testing.assert_raises(ValueError):
        poly.transpose(is_undefined_primal=[True], has_cotangent=[True])


def test_backward():
    """Test the backward method for gradient computation."""
    # Create a linear polynomial for testing: f(x,y) = x^T * y (dot product)
    stp = make_simple_dot_product_stp()
    poly = cue.SegmentedPolynomial.eval_last_operand(stp)

    # Input values
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    # Cotangent for the output (upstream gradient)
    cotangent = np.array([2.0])

    # Test backward with respect to both x and y
    backward_both = poly.backward(requires_gradient=[True, True], has_cotangent=[True])

    # The backward polynomial computes gradients for all inputs that require gradients
    # For f(x,y) = x^T * y:
    # - gradient w.r.t x is y * cotangent
    # - gradient w.r.t y is x * cotangent
    grad_x, grad_y = backward_both(x, y, cotangent)
    expected_grad_x = y * cotangent[0]
    expected_grad_y = x * cotangent[0]

    assert np.allclose(grad_x, expected_grad_x)
    assert np.allclose(grad_y, expected_grad_y)

    # Test backward with respect to only x
    backward_x = poly.backward(requires_gradient=[True, False], has_cotangent=[True])

    # Should only compute gradient for x
    [grad_x_only] = backward_x(x, y, cotangent)
    assert np.allclose(grad_x_only, expected_grad_x)

    # Test backward with respect to only y
    backward_y = poly.backward(requires_gradient=[False, True], has_cotangent=[True])

    # Should only compute gradient for y
    [grad_y_only] = backward_y(x, y, cotangent)
    assert np.allclose(grad_y_only, expected_grad_y)

    # Test with zero cotangent
    zero_cotangent = np.array([0.0])
    grad_x_zero, grad_y_zero = backward_both(x, y, zero_cotangent)

    # With zero cotangent, gradients should be zero
    assert np.allclose(grad_x_zero, np.zeros_like(x))
    assert np.allclose(grad_y_zero, np.zeros_like(y))


def test_symmetrize_identical_operands():
    """Test symmetrization and unsymmetrization of polynomials with identical operands."""
    stp = cue.SegmentedTensorProduct.empty_segments([2, 2, 1])
    stp.add_path(0, 1, 0, c=1.0)  # x0 * y1 path

    # Create operation that uses the same input buffer twice
    op = cue.Operation((0, 0, 1))  # Use buffer 0 twice, write to buffer 1
    poly = cue.SegmentedPolynomial(
        [cue.SegmentedOperand.empty_segments(2)],
        [cue.SegmentedOperand.empty_segments(1)],
        [(op, stp)],
    )
    # Symmetrize the polynomial
    sym_poly = poly.symmetrize_for_identical_operands()

    # Check that we get 0.5 x0*y1 + 0.5 x1*y0
    # This means we should have two paths with coefficient 0.5
    [(_, sym_stp)] = sym_poly.operations
    assert len(sym_stp.paths) == 2
    # Check that we get 0.5 x0*y1 + 0.5 x1*y0
    assert sym_stp.paths[0].coefficients == 0.5
    assert sym_stp.paths[1].coefficients == 0.5
    # Check that the paths have different indices (operands swapped)
    assert sym_stp.paths[0].indices == (0, 1, 0)
    assert sym_stp.paths[1].indices == (1, 0, 0)

    # Test that unsymmetrize returns to original form
    unsym_poly = sym_poly.unsymmetrize_for_identical_operands()
    [(_, unsym_stp)] = unsym_poly.operations
    assert len(unsym_stp.paths) == 1
    assert unsym_stp.paths[0].coefficients == 1.0
    assert unsym_stp.paths[0].indices == (0, 1, 0)

    # Test evaluation to verify the symmetrization works correctly
    x = np.array([1.0, 2.0])
    [result] = poly(x)  # Original polynomial
    [sym_result] = sym_poly(x)  # Symmetrized polynomial
    assert np.allclose(result, sym_result)  # Results should be identical
