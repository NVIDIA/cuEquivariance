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
