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


def test_init_segmented_polynomial():
    """Test initialization of SegmentedPolynomial."""
    stp = make_simple_stp()
    op = cue.Operation((0, 1, 2))
    poly = cue.SegmentedPolynomial(2, 1, [(op, stp)])

    assert poly.num_inputs == 2
    assert poly.num_outputs == 1
    assert poly.num_operands == 3
    assert len(poly.tensor_products) == 1
    assert poly.tensor_products[0] == (op, stp)


def test_polynomial_equality():
    """Test equality comparison of polynomials."""
    stp1 = make_simple_stp()
    stp2 = make_simple_stp()
    op1 = cue.Operation((0, 1, 2))
    op2 = cue.Operation((0, 1, 2))

    poly1 = cue.SegmentedPolynomial(2, 1, [(op1, stp1)])
    poly2 = cue.SegmentedPolynomial(2, 1, [(op2, stp2)])
    poly3 = cue.SegmentedPolynomial(2, 1, [(op2, 2 * stp2)])

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

    op = cue.Operation((0, 1, 2))
    poly = cue.SegmentedPolynomial(2, 1, [(op, stp)])

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

    poly = cue.SegmentedPolynomial(2, 2, [(op1, stp1), (op2, stp2)])

    # Test buffer properties
    assert poly.buffer_sizes == [2, 2, 2, 1]
    assert poly.input_sizes == [2, 2]
    assert poly.output_sizes == [2, 1]

    assert poly.used_buffers() == [0, 1, 2, 3]
    assert poly.buffer_used() == [True, True, True, True]


def test_remove_unused_buffers():
    """Test removing unused buffers from the polynomial."""
    stp = make_simple_stp()
    # Use operation that doesn't use buffer 1
    op = cue.Operation((0, 2, 3))  # Note: buffer 1 is not used

    poly = cue.SegmentedPolynomial(3, 1, [(op, stp)])

    # Buffer 1 is not used
    assert poly.buffer_used() == [True, False, True, True]

    # Remove unused buffer
    cleaned_poly = poly.remove_unused_buffers()

    assert cleaned_poly.num_inputs == 2
    assert cleaned_poly.num_outputs == 1
    assert cleaned_poly.buffer_used() == [True, True, True]


def test_consolidate():
    """Test consolidating tensor products."""
    stp1 = make_simple_stp()
    stp2 = make_simple_stp()

    op = cue.Operation((0, 1, 2))

    # Create a polynomial with duplicate operations
    poly = cue.SegmentedPolynomial(2, 1, [(op, stp1), (op, stp2)])

    # Consolidate the polynomial
    consolidated = poly.consolidate()

    # Should have fused the two tensor products
    assert len(consolidated.tensor_products) == 1
    # Coefficients should have been combined for each path
    assert len(consolidated.tensor_products[0][1].paths) == 2
    # The coefficients should have been added
    assert consolidated.tensor_products[0][1].paths[0].coefficients == 2.0
    assert consolidated.tensor_products[0][1].paths[1].coefficients == -4.0


def test_stack():
    """Test stacking polynomials."""
    # Create two simple polynomials using make_simple_stp
    stp = make_simple_stp()
    op1 = cue.Operation((0, 1, 2))
    poly1 = cue.SegmentedPolynomial(2, 1, [(op1, stp)])

    stp2 = make_simple_stp()
    op2 = cue.Operation((0, 1, 2))
    poly2 = cue.SegmentedPolynomial(2, 1, [(op2, stp2)])

    # Stack the polynomials with the output being stacked
    stacked = cue.SegmentedPolynomial.stack([poly1, poly2], [False, False, True])

    assert stacked.num_inputs == 2
    assert stacked.num_outputs == 1

    assert stacked.buffer_sizes == [2, 2, 4]

    [(_, stp)] = stacked.tensor_products
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
    poly = cue.SegmentedPolynomial(2, 1, [(op, stp)])

    # Test FLOPS calculation
    flops = poly.flops(batch_size=100)
    assert flops > 0

    # Test memory calculation
    memory = poly.memory([100, 100, 100])
    assert memory == 100 * (2 + 2 + 2)  # All operands have size 2
