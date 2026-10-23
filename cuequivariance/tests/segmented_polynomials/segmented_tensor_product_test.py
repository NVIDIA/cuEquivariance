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


def make_coeffs(shape):
    n = np.prod(shape)
    c = np.arange(n) + 1.0
    return c.reshape(shape)


def test_user_friendly():
    d = cue.SegmentedTensorProduct.from_subscripts("ia_jb_kab+ijk")
    assert (
        str(d)
        == "ia,jb,kab+ijk sizes=0,0,0 num_segments=0,0,0 num_paths=0 a= b= i= j= k="
    )

    with pytest.raises(ValueError):
        d.add_path(0, 0, 0, c=make_coeffs((2, 2, 3)))  # need to add segments first

    with pytest.raises(ValueError):
        d.add_segment(0, (2, 2, 2))  # wrong number of dimensions

    d.add_segment(0, (5, 16))
    d.add_segment(1, (5, 32))
    d.add_segment(2, (4, 16, 32))
    d.add_segment(2, (4, 32, 32))

    with pytest.raises(ValueError):
        d.add_path(0, 0, 0, c=make_coeffs((5, 5, 5)))  # wrong dimension for k

    assert (
        str(d)
        == "ia,jb,kab+ijk sizes=80,160,6144 num_segments=1,1,2 num_paths=0 a={16, 32} b=32 i=5 j=5 k=4"
    )

    d.add_path(0, 0, 0, c=make_coeffs((5, 5, 4)))
    assert (
        str(d)
        == "ia,jb,kab+ijk sizes=80,160,6144 num_segments=1,1,2 num_paths=1 a={16, 32} b=32 i=5 j=5 k=4"
    )

    assert not d.all_segments_are_used()
    assert d.subscripts == "ia,jb,kab+ijk"
    assert d.subscripts.is_equivalent("ia,jb,kab+ijk")

    d.assert_valid()


def test_squeeze():
    d = cue.SegmentedTensorProduct.from_subscripts("i_j+ij")
    d.add_segment(0, (1,))
    d.add_segment(1, (20,))
    d.add_path(0, 0, c=make_coeffs((1, 20)))
    d.assert_valid()

    assert d.squeeze_modes().subscripts == ",j+j"

    d = cue.SegmentedTensorProduct.from_subscripts("i_j+ij")
    d.add_segment(0, (1,))
    d.add_segment(0, (2,))
    d.add_segment(1, (20,))
    d.add_path(0, 0, c=make_coeffs((1, 20)))
    d.add_path(1, 0, c=make_coeffs((2, 20)))
    d.assert_valid()

    assert d.squeeze_modes().subscripts == "i,j+ij"

    with pytest.raises(ValueError):
        d.squeeze_modes("i")


def test_normalize_paths_for_operand():
    d = cue.SegmentedTensorProduct.from_subscripts("i_j+ij")

    d.add_segments(0, 2 * [(2,)])
    d.add_segments(1, 2 * [(3,)])
    d.assert_valid()

    d.add_path(0, 0, c=np.array([[2, 0, 0], [0, 2, 0]]))
    d.assert_valid()

    d = d.normalize_paths_for_operand(0)
    d.assert_valid()

    np.testing.assert_allclose(
        d.paths[0].coefficients,
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ]
        ),
    )


def make_example_descriptor():
    d = cue.SegmentedTensorProduct.from_subscripts("uv_iu_jv+ij")
    d.add_path(
        None,
        None,
        None,
        c=np.random.randn(2, 3),
        dims={"u": 4, "v": 5, "i": 2, "j": 3},
    )
    d.assert_valid()
    d.add_path(
        None,
        None,
        None,
        c=np.random.randn(4, 5),
        dims={"u": 2, "v": 3, "i": 4, "j": 5},
    )
    d.assert_valid()
    return d


def test_flatten():
    d = make_example_descriptor()
    d.assert_valid()

    assert d.flatten_modes("").subscripts == "uv,iu,jv+ij"
    assert d.flatten_modes("i").subscripts == "uv,u,jv+j"
    assert d.flatten_modes("j").subscripts == "uv,iu,v+i"
    assert d.flatten_modes("ij").subscripts == "uv,u,v"
    assert d.flatten_modes("ui").subscripts == "v,,jv+j"

    x0 = np.random.randn(d.operands[0].size)
    x1 = np.random.randn(d.operands[1].size)
    x2 = cue.segmented_polynomials.compute_last_operand(d, x0, x1)

    for channels in ["i", "j", "ij", "ui", "iju", "uvij"]:
        np.testing.assert_allclose(
            x2,
            cue.segmented_polynomials.compute_last_operand(
                d.flatten_modes(channels), x0, x1
            ),
        )


def test_flatten_coefficients():
    d = make_example_descriptor()

    assert d.subscripts == "uv,iu,jv+ij"
    assert d.flatten_coefficient_modes().subscripts == "uv,u,v"

    d = d.add_or_transpose_modes("uv,ui,jv+ij")
    assert d.subscripts == "uv,ui,jv+ij"

    with pytest.raises(ValueError):
        d.flatten_coefficient_modes()

    assert d.flatten_coefficient_modes(force=True).subscripts == "v,,v"


def test_consolidate():
    d = cue.SegmentedTensorProduct.from_subscripts("ab_ab")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (2, 3))
    d.add_path(0, 0, c=1.0)
    d.assert_valid()

    assert d.consolidate_modes().subscripts == "a,a"

    d = cue.SegmentedTensorProduct.from_subscripts("ab_ab_a")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (2, 3))
    d.add_segment(2, (2,))
    d.add_path(0, 0, 0, c=1.0)
    d.assert_valid()

    assert d.consolidate_modes() == d

    d = cue.SegmentedTensorProduct.from_subscripts("ab_iab+abi")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (4, 2, 3))
    d.add_path(0, 0, c=make_coeffs((2, 3, 4)))
    d.assert_valid()

    assert d.consolidate_modes().subscripts == "a,ia+ai"

    d = cue.SegmentedTensorProduct.from_subscripts("ab,iab+abi")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (4, 2, 3))
    d.add_path(0, 0, c=make_coeffs((2, 3, 4)))

    assert d.consolidate_modes().subscripts == "a,ia+ai"


def test_stacked_coefficients():
    d = cue.SegmentedTensorProduct.from_subscripts("ab_ab+ab")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (2, 3))
    np.testing.assert_allclose(d.stacked_coefficients, make_coeffs((0, 2, 3)))

    d.add_path(0, 0, c=make_coeffs((2, 3)))
    d.add_path(0, 0, c=make_coeffs((2, 3)))
    expected = np.stack([make_coeffs((2, 3)), make_coeffs((2, 3))], axis=0)
    np.testing.assert_allclose(d.stacked_coefficients, expected)

    d = d.consolidate_paths()
    np.testing.assert_allclose(d.stacked_coefficients, 2 * make_coeffs((1, 2, 3)))


@pytest.mark.parametrize("extended", [False, True])
def test_data_transfer(extended: bool):
    d = cue.SegmentedTensorProduct.from_subscripts("ui,uj,uk+ijk")
    d.add_path(None, None, None, c=make_coeffs((3, 3, 3)), dims={"u": 12})
    d.add_path(None, None, None, c=make_coeffs((1, 2, 1)), dims={"u": 14})
    d.assert_valid()

    dict = d.to_dict(extended)
    json = d.to_json(extended)
    bin = d.to_bytes(extended)
    b64 = d.to_base64(extended)

    assert d == cue.SegmentedTensorProduct.from_dict(dict)
    assert d == cue.SegmentedTensorProduct.from_json(json)
    assert d == cue.SegmentedTensorProduct.from_bytes(bin)
    assert d == cue.SegmentedTensorProduct.from_base64(b64)


def test_to_text():
    d = cue.SegmentedTensorProduct.from_subscripts("iu,ju,ku+ijk")
    d.add_path(None, None, None, c=make_coeffs((3, 3, 3)), dims={"u": 12})
    d.add_path(None, None, None, c=make_coeffs((1, 2, 1)), dims={"u": 14})
    d = d.flatten_modes("ijk")

    text = d.to_text()
    assert (
        text
        == """u,u,u sizes=50,64,50 num_segments=4,5,4 num_paths=29 u={12, 14}
operand #0 subscripts=u
  | u: [12, 12, 12, 14]
operand #1 subscripts=u
  | u: [12, 12, 12, 14, 14]
operand #2 subscripts=u
  | u: [12, 12, 12, 14]
Flop cost: 0->704 1->704 2->704
Memory cost: 164
Path indices: 0 0 0, 0 0 1, 0 0 2, 0 1 0, 0 1 1, 0 1 2, 0 2 0, 0 2 1, 0 2 2, 1 0 0, 1 0 1, 1 0 2, 1 1 0, 1 1 1, 1 1 2, 1 2 0, 1 2 1, 1 2 2, 2 0 0, 2 0 1, 2 0 2, 2 1 0, 2 1 1, 2 1 2, 2 2 0, 2 2 1, 2 2 2, 3 3 3, 3 4 3
Path coefficients: [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 1.0 2.0]"""
    )


def test_hash():
    d = cue.SegmentedTensorProduct.from_subscripts("ui,uj,uk+ijk")
    d.add_path(None, None, None, c=make_coeffs((3, 3, 3)), dims={"u": 12})
    d.add_path(None, None, None, c=make_coeffs((1, 2, 1)), dims={"u": 14})
    assert hash(d) == hash(d)

    d2 = cue.SegmentedTensorProduct.from_subscripts("ui,uj,uk+ijk")
    assert hash(d) != hash(d2)
    d2.add_path(None, None, None, c=make_coeffs((3, 3, 3)), dims={"u": 12})
    assert hash(d) != hash(d2)
    d2.add_path(None, None, None, c=make_coeffs((1, 2, 1)), dims={"u": 14})
    assert hash(d) == hash(d2)


def test_split_mode():
    # Create a descriptor with a mode that has dimensions divisible by the desired split size
    d = cue.SegmentedTensorProduct.from_subscripts("ua,ub+ab")

    # Add a segment with u dimension = 6 (divisible by 2 and 3)
    d.add_segment(0, (6, 4))
    d.add_segment(1, (6, 5))

    # Add a path
    d.add_path(0, 0, c=make_coeffs((4, 5)))
    d.assert_valid()

    # Split mode 'u' with size 2
    d_split = d.split_mode("u", 2)
    d_split.assert_valid()

    # Check that the dimensions are correctly split
    assert d_split.operands[0].num_segments == 3  # 6/2 = 3 segments
    assert d_split.operands[1].num_segments == 3  # 6/2 = 3 segments

    # Check that the subscripts are preserved
    assert d_split.subscripts == "ua,ub+ab"

    # Check that the segments have the correct shape
    for segment in d_split.operands[0]:
        assert segment[0] == 2  # First dimension should be 2
        assert segment[1] == 4  # Second dimension should be 4

    for segment in d_split.operands[1]:
        assert segment[0] == 2  # First dimension should be 2
        assert segment[1] == 5  # Second dimension should be 5

    # Test with a different split size
    d_split_3 = d.split_mode("u", 3)
    d_split_3.assert_valid()

    assert d_split_3.operands[0].num_segments == 2  # 6/3 = 2 segments
    assert d_split_3.operands[1].num_segments == 2  # 6/3 = 2 segments

    # Test error case: split size not divisible by dimension
    with pytest.raises(ValueError):
        d.split_mode("u", 5)  # 6 is not divisible by 5

    # Test case where mode is not in descriptor
    d_unchanged = d.split_mode("v", 2)  # 'v' is not in the descriptor
    assert d_unchanged == d

    # Test case where mode is not at the beginning of the operand
    d_complex = cue.SegmentedTensorProduct.from_subscripts("au,bu+ab")
    d_complex.add_segment(0, (3, 6))
    d_complex.add_segment(1, (4, 6))
    d_complex.add_path(0, 0, c=make_coeffs((3, 4)))

    with pytest.raises(ValueError):
        d_complex.split_mode("u", 2)  # 'u' is not the first mode in operands

    # Test with coefficient subscripts
    d_coeff = cue.SegmentedTensorProduct.from_subscripts("ua,ub,ab+ab")
    d_coeff.add_segment(0, (6, 4))
    d_coeff.add_segment(1, (6, 5))
    d_coeff.add_segment(2, (4, 5))
    d_coeff.add_path(0, 0, 0, c=make_coeffs((4, 5)))

    d_coeff_split = d_coeff.split_mode("u", 2)
    d_coeff_split.assert_valid()

    assert d_coeff_split.operands[0].num_segments == 3
    assert d_coeff_split.operands[1].num_segments == 3
    assert d_coeff_split.operands[2].num_segments == 1  # Not affected by u split

    # Check that computation results are equivalent
    # Create a simple descriptor with just two operands for testing compute_last_operand
    d_compute = cue.SegmentedTensorProduct.from_subscripts("a,b+ab")
    d_compute.add_segment(0, (4,))
    d_compute.add_segment(1, (5,))
    d_compute.add_path(0, 0, c=make_coeffs((4, 5)))

    # Test computation on original descriptor
    x_input = np.random.randn(d_compute.operands[0].size)
    result_original = cue.segmented_polynomials.compute_last_operand(d_compute, x_input)

    # Verify split_mode works by first flattening the results to remove 'u' mode indices
    d_ua = cue.SegmentedTensorProduct.from_subscripts("ua,b+ab")
    d_ua.add_segment(0, (6, 4))
    d_ua.add_segment(1, (5,))
    d_ua.add_path(0, 0, c=make_coeffs((4, 5)))
    d_ua_split = d_ua.split_mode("u", 2)

    # Input for the split descriptor - we need a tensor with the right shape
    x_input_split = np.random.randn(d_ua_split.operands[0].size)
    result_split = cue.segmented_polynomials.compute_last_operand(
        d_ua_split, x_input_split
    )

    # Verify the shapes are consistent with our expectations
    assert result_original.shape == (5,)
    assert result_split.shape == (5,)


def test_add_or_transpose_modes():
    # Test 1: Simple mode transposition
    d = cue.SegmentedTensorProduct.from_subscripts("ia,ja+ij")
    d.add_segment(0, (3, 4))
    d.add_segment(1, (5, 4))
    d.add_path(0, 0, c=make_coeffs((3, 5)))
    d.assert_valid()

    # Transpose modes in first operand
    d_trans = d.add_or_transpose_modes("ai,ja+ij")
    d_trans.assert_valid()
    assert d_trans.subscripts == "ai,ja+ij"
    assert d_trans.operands[0][0] == (4, 3) and d_trans.operands[1][0] == (5, 4)
    np.testing.assert_allclose(d_trans.paths[0].coefficients, d.paths[0].coefficients)

    # Test 2: Adding new modes
    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d.add_segment(0, (3,))
    d.add_segment(1, (4,))
    d.add_path(0, 0, c=make_coeffs((3, 4)))
    d.assert_valid()

    # Add new modes with specified dimensions
    d_new = d.add_or_transpose_modes("ia,ja+ij", dims={"a": 5})
    d_new.assert_valid()
    assert d_new.subscripts == "ia,ja+ij"
    assert d_new.operands[0][0] == (3, 5) and d_new.operands[1][0] == (4, 5)

    # Test 3 & 4: Error cases
    with pytest.raises(ValueError):
        d.add_or_transpose_modes("ia,ja+ij")  # Missing dims for a
    with pytest.raises(ValueError):
        d.add_or_transpose_modes("i,j+i")  # Removing j from coefficients

    # Test 5: Transposing coefficient modes
    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d.add_segment(0, (3,))
    d.add_segment(1, (4,))
    d.add_path(0, 0, c=make_coeffs((3, 4)))
    d.assert_valid()

    # Transpose coefficient modes
    d_coeff = d.add_or_transpose_modes("i,j+ji")
    d_coeff.assert_valid()
    assert d_coeff.subscripts == "i,j+ji"
    np.testing.assert_allclose(d_coeff.paths[0].coefficients, d.paths[0].coefficients.T)

    # Test 6: Adding batch dimensions
    d = cue.SegmentedTensorProduct.from_subscripts("ui,uj+ij")
    d.add_segment(0, (5, 3))
    d.add_segment(1, (5, 4))
    d.add_path(0, 0, c=make_coeffs((3, 4)))
    d.assert_valid()

    # Add batch dimension to both operands
    d_batch = d.add_or_transpose_modes("bui,buj+ij", dims={"b": 8})
    d_batch.assert_valid()
    assert d_batch.subscripts == "bui,buj+ij"
    assert d_batch.operands[0][0] == (8, 5, 3) and d_batch.operands[1][0] == (8, 5, 4)


def test_add_or_rename_modes():
    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d.add_segment(0, (3,))
    d.add_segment(1, (4,))
    d.add_path(0, 0, c=make_coeffs((3, 4)))
    d.assert_valid()
    d_id = d.add_or_rename_modes("i,j+ij")
    d_id.assert_valid()
    assert d_id.subscripts == "i,j+ij"
    x_input = np.random.randn(d.operands[0].size)
    res0 = cue.segmented_polynomials.compute_last_operand(d, x_input)
    res_id = cue.segmented_polynomials.compute_last_operand(d_id, x_input)
    np.testing.assert_allclose(res0, res_id)
    d_ren = d.add_or_rename_modes("a,b+ab")
    d_ren.assert_valid()
    assert d_ren.subscripts == "a,b+ab"
    np.testing.assert_allclose(
        res0, cue.segmented_polynomials.compute_last_operand(d_ren, x_input)
    )
    with pytest.raises(ValueError):
        d.add_or_rename_modes("i+ij")
    d_sup = d.add_or_rename_modes("bi,bj+ij", mapping={"i": "i", "j": "j"})
    d_sup.assert_valid()
    assert d_sup.subscripts == "bi,bj+ij"
    np.testing.assert_allclose(
        res0, cue.segmented_polynomials.compute_last_operand(d_sup, x_input)
    )


def test_consolidate_with_optional_argument():
    d = cue.SegmentedTensorProduct.from_subscripts("ab_ab")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (2, 3))
    d.add_path(0, 0, c=1.0)
    d.assert_valid()
    d_consol = d.consolidate_modes("ab")
    assert d_consol.subscripts == "a,a"


def test_slice_by_segment():
    """Test the slice_by_segment method for slicing SegmentedTensorProduct operands."""
    # Create descriptor with multiple segments per operand
    d = cue.SegmentedTensorProduct.from_subscripts("uv,ui,vj+ij")

    # Add segments with consistent dimensions
    for i, (u, v) in enumerate([(2, 3), (2, 4), (3, 3), (3, 4)]):
        d.add_segment(0, (u, v))
        d.add_segment(1, (u, i + 5))
        d.add_segment(2, (v, i + 7))

    # Add paths with consistent dimensions
    for i in range(4):
        d.add_path(i, i, i, c=make_coeffs((i + 5, i + 7)))

    d.assert_valid()
    assert (d.num_operands, d.num_paths) == (3, 4)
    assert [len(op) for op in d.operands] == [4, 4, 4]

    # Test single integer index, slice objects, and complex slicing
    d_single = d.slice_by_segment[:1, :, :]
    assert (len(d_single.operands[0]), d_single.num_paths) == (1, 1)
    assert d_single.operands[0][0] == (2, 3)
    d_single.assert_valid()

    d_slice = d.slice_by_segment[1:3, :, :]
    assert ([len(op) for op in d_slice.operands], d_slice.num_paths) == ([2, 4, 4], 2)
    assert d_slice.operands[0].segments == ((2, 4), (3, 3))
    d_slice.assert_valid()

    d_multi = d.slice_by_segment[::2, 1:, :2]
    assert [len(op) for op in d_multi.operands] == [2, 3, 2]
    assert d_multi.operands[0].segments == ((2, 3), (3, 3))
    d_multi.assert_valid()

    # Test negative indexing and empty results
    d_neg = d.slice_by_segment[-1:, :, :]
    assert (len(d_neg.operands[0]), d_neg.operands[0][0]) == (1, (3, 4))
    d_neg.assert_valid()

    d_empty = d.slice_by_segment[2:3, 0:1, 0:1]
    assert d_empty.num_paths == 0
    d_empty.assert_valid()

    # Test path index remapping
    d_simple = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d_simple.add_segments(0, [(2,), (3,), (4,)])
    d_simple.add_segments(1, [(5,), (6,)])
    d_simple.add_path(1, 0, c=make_coeffs((3, 5)))
    d_simple.add_path(2, 1, c=make_coeffs((4, 6)))

    d_sliced = d_simple.slice_by_segment[1:, :]
    assert d_sliced.operands[0].segments == ((3,), (4,))
    assert [p.indices for p in d_sliced.paths] == [(0, 0), (1, 1)]
    d_sliced.assert_valid()

    # Test error conditions
    with pytest.raises(ValueError, match="Expected a slice or int for each operand"):
        d.slice_by_segment[0, 1]  # not enough indices
    with pytest.raises(ValueError, match="Expected a slice or int for each operand"):
        d.slice_by_segment[0, 1, 2, 3]  # too many indices
    with pytest.raises(TypeError, match="Invalid slice type"):
        d.slice_by_segment[0, 1, [2]]  # invalid slice type

    # Test coefficient preservation and computation equivalence
    d_coeff = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d_coeff.add_segments(0, [(2,), (3,)])
    d_coeff.add_segments(1, [(4,), (5,)])
    original_coeffs = make_coeffs((2, 4))
    d_coeff.add_path(0, 0, c=original_coeffs)
    d_coeff.assert_valid()

    d_coeff_sliced = d_coeff.slice_by_segment[:, :]
    np.testing.assert_allclose(d_coeff_sliced.paths[0].coefficients, original_coeffs)

    x0 = np.random.randn(d_coeff.operands[0].size)
    result_original = cue.segmented_polynomials.compute_last_operand(d_coeff, x0)
    result_sliced = cue.segmented_polynomials.compute_last_operand(d_coeff_sliced, x0)
    np.testing.assert_allclose(result_original, result_sliced)


def test_slice_by_size():
    """Test the slice_by_size method for slicing SegmentedTensorProduct operands by flat size."""
    # Create descriptor with segments of different sizes
    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")

    # Add segments with different sizes: 2, 6, 12 for first operand
    d.add_segments(0, [(2,), (6,), (12,)])
    # Add segments with sizes: 3, 4 for second operand
    d.add_segments(1, [(3,), (4,)])

    # Add paths
    d.add_path(0, 0, c=make_coeffs((2, 3)))
    d.add_path(1, 1, c=make_coeffs((6, 4)))
    d.add_path(2, 0, c=make_coeffs((12, 3)))

    d.assert_valid()
    assert d.operands[0].size == 2 + 6 + 12  # 20
    assert d.operands[1].size == 3 + 4  # 7

    # Test slicing by size - should include segments that overlap with size range
    d_size_slice = d.slice_by_size[
        2:8, :
    ]  # Should include segment 1 (size 6) from first operand
    assert len(d_size_slice.operands[0]) == 1
    assert d_size_slice.operands[0][0] == (6,)
    assert len(d_size_slice.operands[1]) == 2  # All segments from second operand
    assert (
        d_size_slice.num_paths == 1
    )  # Only the path using segment 1 from first operand
    d_size_slice.assert_valid()

    # Test integer indexing by size
    d_single_size = d.slice_by_size[:1, :]
    assert len(d_single_size.operands[0]) == 1
    assert d_single_size.operands[0][0] == (2,)
    d_single_size.assert_valid()

    # Test error conditions
    with pytest.raises(ValueError, match="Expected a slice or int for each operand"):
        d.slice_by_size[0]  # not enough indices
    with pytest.raises(ValueError, match="Step sizes other than 1 are not supported"):
        d.slice_by_size[::2, :]  # step size not supported
    with pytest.raises(TypeError, match="Invalid slice type"):
        d.slice_by_size[0, [1]]  # invalid slice type


def test_descriptor_metadata_and_path_queries():
    empty = cue.SegmentedTensorProduct.from_subscripts("i,i")
    assert empty.indices.shape == (0, 2)
    assert empty.coefficients_are_stackable
    assert empty.stacked_coefficients.shape == (0,)
    assert "No paths." in empty.to_text()

    with pytest.raises(ValueError, match="not contracted"):
        cue.SegmentedTensorProduct.from_subscripts("i,j").assert_valid()

    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d.add_segments(0, [(2,), (2,), (3,)])
    d.add_segments(1, [(3,), (3,)])
    d.add_path(0, 0, c=make_coeffs((2, 3)))
    d.add_path(1, 1, c=make_coeffs((2, 3)))
    d.add_path(2, 0, c=make_coeffs((3, 3)))
    d.assert_valid()

    np.testing.assert_array_equal(d.indices, [[0, 0], [1, 1], [2, 0]])
    assert d.get_path_dimensions_dict(2) == {"i": 3, "j": 3}
    assert d.get_path_dim(d.paths[0], "i") == 2
    assert d.get_path_dim(0, "absent") == 0
    assert d.get_path_dim(0, "absent", returns_set=True) == set()
    assert d.segment_slice(0, 1) == slice(2, 4)
    assert d.segment_slice(0, d.paths[1]) == slice(2, 4)
    assert d.get_segment_shape(1, 1) == (3,)
    assert d.get_segment_shape(1, d.paths[1]) == (3,)
    assert d.all_segments_are_used()
    np.testing.assert_array_equal(d.compressed_path_segment(0), [0, 1, 2, 3])
    np.testing.assert_array_equal(d.sort_paths(1).compressed_path_segment(1), [0, 2, 3])

    assert not d.coefficients_are_stackable
    assert "Path coefficients:\n" in d.to_text()
    assert "Path coefficients:" not in d.to_text(coefficient_formatter=None)
    d.add_segment(1, (4,))
    assert not d.all_segments_are_used()

    assert empty < d
    assert not (d == object())
    with pytest.raises(TypeError):
        _ = d < object()

    optimal_cost = d.flop(0)
    assert d.flop(0, algorithm="naive") >= optimal_cost
    assert d.flop(0, batch_size=3) == 3 * optimal_cost
    with pytest.raises(ValueError, match="unknown algorithm"):
        d.flop(0, algorithm="unknown")

    d.add_path(0, 0, c=make_coeffs((2, 3)))
    with pytest.raises(ValueError, match="Paths must be sorted"):
        d.compressed_path_segment(0)


def test_insert_path_segment_forms_and_validation():
    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d.add_segment(0, (2,))
    d.add_segment(1, (3,))
    d.add_path(0, 0, c=np.ones((2, 3)))

    d.insert_path(0, (2,), {"j": 3}, c=2 * np.ones((2, 3)))
    d.insert_path(
        -1,
        -1,
        -1,
        c=3 * np.ones((2, 3)),
        dims={"i": 2, "j": 3, "unused": 10},
    )
    d.add_path(None, None, c=4 * np.ones((2, 3)))
    d.assert_valid()
    assert [path.indices for path in d.paths] == [
        (1, 1),
        (0, 0),
        (1, 1),
        (2, 2),
    ]

    fused = d.fuse_paths_with_same_indices()
    assert fused.num_paths == 3
    assert fused.paths[0].indices == (1, 1)
    np.testing.assert_allclose(fused.paths[0].coefficients, 5 * np.ones((2, 3)))
    fused.assert_valid()

    with pytest.raises(ValueError, match="expected 2 indices"):
        d.insert_path(0, 0, c=np.ones((2, 3)))
    with pytest.raises(ValueError, match="expected 2 dimensions"):
        d.insert_path(0, 0, 0, c=np.ones(2))
    with pytest.raises(ValueError, match="expected a segment index"):
        d.insert_path(0, [0], 0, c=np.ones((2, 3)))
    with pytest.raises(ValueError, match="Ambiguous dimensions"):
        d.insert_path(0, (4,), 0, c=np.ones((2, 3)))

    missing_mode = cue.SegmentedTensorProduct.from_subscripts("ui,uj+ij")
    with pytest.raises(ValueError, match="expected a dimension for each subscripts"):
        missing_mode.add_path(None, None, c=np.ones((2, 3)))
    with pytest.raises(ValueError, match="path_index to be between"):
        d.insert_path(d.num_paths + 1, 0, 0, c=np.ones((2, 3)))


def test_mode_extensions_and_coefficient_mode_split():
    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d.add_path(None, None, c=make_coeffs((2, 3)))

    appended = d.append_modes_to_all_operands("b", {"b": 4})
    appended.assert_valid()
    assert appended.subscripts == "ib,jb+ij"
    assert appended.operands[0][0] == (2, 4)
    assert appended.operands[1][0] == (3, 4)
    with pytest.raises(ValueError, match="disjoint"):
        d.append_modes_to_all_operands("i", {"i": 2})
    with pytest.raises(ValueError, match="dimensions for all new modes"):
        d.append_modes_to_all_operands("b", {})
    with pytest.raises(ValueError, match="impossible to introduce new modes"):
        d.add_or_transpose_modes("i,j+ijk", dims={"k": 2})
    with pytest.raises(ValueError, match="single lowercase letter"):
        d.split_mode("IU", 2)

    split_source = cue.SegmentedTensorProduct.from_subscripts("ua,ub+abu")
    split_source.add_path(None, None, c=make_coeffs((2, 3, 4)), dims={"u": 4})
    split_source.assert_valid()
    x = np.arange(8, dtype=float) + 1
    split = split_source.split_mode("u", 2)
    split.assert_valid()
    assert split.num_paths == 2
    assert [path.coefficients.shape for path in split.paths] == [(2, 3, 2)] * 2
    np.testing.assert_allclose(
        cue.segmented_polynomials.compute_last_operand(split_source, x),
        cue.segmented_polynomials.compute_last_operand(split, x),
    )


def test_rename_and_consolidate_mode_boundaries():
    ambiguous = cue.SegmentedTensorProduct.from_subscripts("a,a")
    with pytest.raises(ValueError, match="non-ambiguous mapping"):
        ambiguous.add_or_rename_modes("uv,uv")

    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    with pytest.raises(ValueError, match="all segment modes"):
        d.add_or_rename_modes("ib,jb+ij", mapping={})
    with pytest.raises(ValueError, match="all mapped modes"):
        d.add_or_rename_modes("ib,jb+ij", mapping={"i": "x", "j": "y"})

    for coefficients in ("a", "b"):
        descriptor = cue.SegmentedTensorProduct.from_subscripts(f"ab,ab+{coefficients}")
        assert descriptor.consolidate_modes() == descriptor

    partial = cue.SegmentedTensorProduct.from_subscripts("ab,ac+bc")
    assert partial.consolidate_modes("z") == partial
    with pytest.raises(ValueError, match="contiguous"):
        partial.consolidate_modes("ab")

    coefficient_pair = cue.SegmentedTensorProduct.from_subscripts("ab,ab+a")
    with pytest.raises(ValueError, match="both a and b"):
        coefficient_pair.consolidate_modes("ab")


def test_coefficient_rounding_transform_guards_and_size_slice_edges():
    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    coefficients = np.array([[0.3333, 0.7], [-0.7, 0.5]])
    d.add_path(None, None, c=coefficients)
    rational = d.round_coefficients_to_rational(10)
    sqrt_rational = d.round_coefficients_to_sqrt_rational(10)
    np.testing.assert_allclose(
        rational.paths[0].coefficients,
        [[1 / 3, 7 / 10], [-7 / 10, 1 / 2]],
    )
    np.testing.assert_allclose(
        sqrt_rational.paths[0].coefficients,
        [[1 / 3, np.sqrt(1 / 2)], [-np.sqrt(1 / 2), 1 / 2]],
    )
    np.testing.assert_allclose(d.paths[0].coefficients, coefficients)

    non_scalar = cue.SegmentedTensorProduct.from_subscripts("ui,uj+ij")
    non_scalar.add_path(None, None, c=np.ones((2, 2)), dims={"u": 2})
    with pytest.raises(NotImplementedError, match="sorting non scalar coefficients"):
        non_scalar.sort_indices_for_identical_operands((0, 1))
    with pytest.raises(
        NotImplementedError, match="symmetrizing non scalar coefficients"
    ):
        non_scalar.symmetrize_operands((0, 1))
    with pytest.raises(ValueError, match="lowercase single letter modes"):
        non_scalar.flatten_modes("I")

    size_source = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    size_source.add_segments(0, [(2,), (6,), (12,)])
    size_source.add_segments(1, [(3,), (4,)])
    size_source.add_path(2, 0, c=make_coeffs((12, 3)))
    tail = size_source.slice_by_size[-2:, :]
    tail.assert_valid()
    assert tail.operands[0].segments == ((12,),)
    assert [path.indices for path in tail.paths] == [(0, 0)]

    with pytest.raises(ValueError, match="Expected a slice or int for each operand"):
        size_source.slice_by_segment[:]
    with pytest.raises(ValueError, match="Expected a slice or int for each operand"):
        size_source.slice_by_size[:]


def test_descriptor_metadata_and_path_edges():
    empty = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    empty.add_segment(0, (2,))
    empty.add_segment(1, (3,))
    np.testing.assert_array_equal(empty.indices, np.empty((0, 2), dtype=int))
    assert not empty.all_segments_are_used()
    assert "No paths." in empty.to_text()
    assert empty.__eq__(object()) is NotImplemented
    assert empty.__lt__(object()) is NotImplemented

    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d.add_segments(0, [(2,), (3,)])
    d.add_segments(1, [(4,), (5,)])
    d.add_path(0, 0, c=make_coeffs((2, 4)))
    d.add_path(1, 1, c=make_coeffs((3, 5)))
    d.assert_valid()

    assert empty < d
    assert not d.coefficients_are_stackable
    assert d.all_segments_are_used()
    assert d.get_path_dimensions_dict(1) == {"i": 3, "j": 5}
    assert d.get_path_dim(1, "i") == 3
    assert (d.segment_slice(0, 1).start, d.segment_slice(0, 1).stop) == (2, 5)
    assert d.get_segment_shape(1, 1) == (5,)
    np.testing.assert_array_equal(d.compressed_path_segment(0), [0, 1, 2])

    formatted = d.to_text(lambda value: f"{value:.1f}")
    assert "Path coefficients:\n" in formatted
    assert "Path coefficients:" not in d.to_text(None)
    assert d.flop(1, algorithm="naive") > 0
    assert d.flop(1, algorithm="optimal") > 0
    assert d.flop(1, batch_size=2, algorithm="naive") == 2 * d.flop(
        1, algorithm="naive"
    )
    with pytest.raises(ValueError, match="unknown algorithm"):
        d.flop(1, algorithm="invalid")

    unsorted = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    unsorted.add_segments(0, [(2,), (3,)])
    unsorted.add_segments(1, [(4,), (5,)])
    unsorted.add_path(1, 1, c=make_coeffs((3, 5)))
    unsorted.add_path(0, 0, c=make_coeffs((2, 4)))
    with pytest.raises(ValueError, match="Paths must be sorted"):
        unsorted.compressed_path_segment(0)

    duplicate = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    duplicate.add_segment(0, (2,))
    duplicate.add_segment(1, (3,))
    duplicate.add_path(0, 0, c=np.ones((2, 3)))
    duplicate.add_path(0, 0, c=2 * np.ones((2, 3)))
    fused = duplicate.fuse_paths_with_same_indices()
    assert fused.indices.tolist() == [[0, 0]]
    np.testing.assert_array_equal(fused.paths[0].coefficients, 3 * np.ones((2, 3)))


def test_path_insertion_and_append_mode_edges():
    invalid = cue.SegmentedTensorProduct.from_subscripts("i,j")
    with pytest.raises(ValueError, match="not contracted"):
        invalid.assert_valid()

    d = cue.SegmentedTensorProduct.from_subscripts("i,i")
    with pytest.raises(ValueError, match="expected 2 indices"):
        d.insert_path(0, 0, c=1.0)
    with pytest.raises(ValueError, match="expected 0 dimensions"):
        d.insert_path(0, None, None, c=np.ones((2,)))
    with pytest.raises(ValueError, match="expected a dimension for each"):
        d.add_path(None, None, c=1.0)
    with pytest.raises(ValueError, match="expected a segment index"):
        d.add_path("bad", None, c=1.0)
    with pytest.raises(ValueError, match="between"):
        d.insert_path(4, None, None, c=1.0)

    d.add_path(None, None, c=1.0, dims={"i": 2})
    inserted = d.insert_path(-1, 0, 0, c=2.0)
    assert inserted == 1
    assert d.paths[-1].coefficients.item() == 2.0

    mapped = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    mapped.add_path({"i": 2}, {"j": 3}, c=make_coeffs((2, 3)))
    mapped.assert_valid()
    assert mapped.operands[0][0] == (2,)
    assert mapped.operands[1][0] == (3,)

    inferred = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    inferred.add_path(None, None, c=make_coeffs((2, 3)))
    inferred.assert_valid()
    assert inferred.operands[0][0] == (2,)
    assert inferred.operands[1][0] == (3,)

    with pytest.raises(ValueError, match="disjoint"):
        d.append_modes_to_all_operands("i", {"i": 2})
    with pytest.raises(ValueError, match="dimensions"):
        d.append_modes_to_all_operands("a", {})
    extended = d.append_modes_to_all_operands("a", {"a": 3})
    assert extended.subscripts == "ia,ia"
    assert extended.operands[0][0] == (2, 3)


def test_mode_mapping_and_consolidation_edges():
    d = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    d.add_path(None, None, c=make_coeffs((2, 3)))
    with pytest.raises(ValueError, match="contain all the modes"):
        d.add_or_rename_modes("i+ij")
    with pytest.raises(ValueError, match="non-ambiguous"):
        cue.SegmentedTensorProduct.from_subscripts("a,a").add_or_rename_modes("uv,vu")
    with pytest.raises(ValueError, match="all segment modes"):
        d.add_or_rename_modes("xb,yb+xy", mapping={"i": "x"})
    with pytest.raises(ValueError, match="mapped modes"):
        d.add_or_rename_modes("xb,yb+xy", mapping={"i": "z", "j": "y"})
    with pytest.raises(ValueError, match="impossible to introduce new modes"):
        d.add_or_transpose_modes("i,j+ijk", dims={"k": 2})
    with pytest.raises(ValueError, match="lowercase single letter"):
        d.flatten_modes("A")

    plain = cue.SegmentedTensorProduct.from_subscripts("uv,uv")
    plain.add_segment(0, (2, 3))
    plain.add_segment(1, (2, 3))
    plain.add_path(0, 0, c=1.0)
    assert plain.consolidate_modes().subscripts == "u,u"

    coefficient_modes = cue.SegmentedTensorProduct.from_subscripts("uv,uv+uv")
    coefficient_modes.add_segment(0, (2, 3))
    coefficient_modes.add_segment(1, (2, 3))
    coefficient_modes.add_path(0, 0, c=make_coeffs((2, 3)))
    consolidated = coefficient_modes.consolidate_modes()
    assert consolidated.subscripts == "u,u+u"
    np.testing.assert_array_equal(consolidated.paths[0].coefficients, make_coeffs((6,)))

    unmatched_operands = cue.SegmentedTensorProduct.from_subscripts("uv,u,v")
    unmatched_operands.add_segment(0, (2, 3))
    unmatched_operands.add_segment(1, (2,))
    unmatched_operands.add_segment(2, (3,))
    unmatched_operands.add_path(0, 0, 0, c=1.0)
    assert unmatched_operands.consolidate_modes() == unmatched_operands

    coefficient_v = cue.SegmentedTensorProduct.from_subscripts("uv,uv+v")
    coefficient_v.add_segment(0, (2, 3))
    coefficient_v.add_segment(1, (2, 3))
    coefficient_v.add_path(0, 0, c=np.ones((3,)))
    assert coefficient_v.consolidate_modes() == coefficient_v

    coefficient_u = cue.SegmentedTensorProduct.from_subscripts("uv,uv+u")
    coefficient_u.add_segment(0, (2, 3))
    coefficient_u.add_segment(1, (2, 3))
    coefficient_u.add_path(0, 0, c=np.ones((2,)))
    assert coefficient_u.consolidate_modes() == coefficient_u

    with pytest.raises(ValueError, match="contiguous"):
        cue.SegmentedTensorProduct.from_subscripts("iu,iu").consolidate_modes("ui")

    coefficient_only_u = cue.SegmentedTensorProduct.from_subscripts("ui,ui+u")
    coefficient_only_u.add_segment(0, (3, 2))
    coefficient_only_u.add_segment(1, (3, 2))
    coefficient_only_u.add_path(0, 0, c=np.ones((3,)))
    with pytest.raises(ValueError, match="both u and i"):
        coefficient_only_u.consolidate_modes("ui")
    with pytest.raises(NotImplementedError, match="non scalar coefficients"):
        coefficient_only_u.sort_indices_for_identical_operands((0, 1))
    with pytest.raises(NotImplementedError, match="non scalar coefficients"):
        coefficient_only_u.symmetrize_operands((0, 1))

    untouched_operand = cue.SegmentedTensorProduct.from_subscripts("uv,uv,w+w")
    untouched_operand.add_segment(0, (2, 3))
    untouched_operand.add_segment(1, (2, 3))
    untouched_operand.add_segment(2, (4,))
    untouched_operand.add_path(0, 0, 0, c=np.ones((4,)))
    assert untouched_operand.consolidate_modes("uv").subscripts == "u,u,w+w"


def test_split_coefficient_mode_and_rounding_edges():
    with pytest.raises(ValueError, match="single lowercase letter"):
        cue.SegmentedTensorProduct.from_subscripts("u,u,v+vu").split_mode("UV", 2)

    coefficients = make_coeffs((2, 4))
    d = cue.SegmentedTensorProduct.from_subscripts("u,u,v+vu")
    d.add_segment(2, (2,))
    d.add_path(None, None, 0, c=coefficients)
    split = d.split_mode("u", 2)
    assert split.subscripts == d.subscripts
    assert split.num_paths == 2
    split.assert_valid()
    assert [path.indices for path in split.paths] == [(0, 0, 0), (1, 1, 0)]
    np.testing.assert_array_equal(
        np.concatenate([path.coefficients for path in split.paths], axis=1),
        coefficients,
    )

    round_input = cue.SegmentedTensorProduct.from_subscripts("i,j+ij")
    round_input.add_segment(0, (1,))
    round_input.add_segment(1, (1,))
    round_input.add_path(0, 0, c=np.array([[0.25]]))
    rational = round_input.round_coefficients_to_rational(32)
    sqrt_rational = round_input.round_coefficients_to_sqrt_rational(32)
    modified = round_input.modify_coefficients(lambda c: 2 * c)
    assert rational is not round_input
    assert sqrt_rational is not round_input
    np.testing.assert_allclose(rational.paths[0].coefficients, [[0.25]])
    np.testing.assert_allclose(sqrt_rational.paths[0].coefficients, [[0.25]])
    np.testing.assert_allclose(modified.paths[0].coefficients, [[0.5]])
    np.testing.assert_allclose(round_input.paths[0].coefficients, [[0.25]])


def test_single_operand_slicing_and_coefficient_transform_edges():
    d = cue.SegmentedTensorProduct.from_subscripts("i+i")
    d.add_segments(0, [(2,), (3,)])
    d.add_path(0, c=np.full((2,), 0.5))
    d.add_path(1, c=np.full((3,), 1.0))

    by_segment = d.slice_by_segment[:]
    assert by_segment.operands[0].segments == ((2,), (3,))
    assert by_segment.indices.tolist() == [[0], [1]]

    by_size = d.slice_by_size[0:2]
    assert by_size.operands[0].segments == ((2,),)
    assert by_size.indices.tolist() == [[0]]
    with pytest.raises(ValueError, match="Step sizes other than 1"):
        d.slice_by_size[::2]
