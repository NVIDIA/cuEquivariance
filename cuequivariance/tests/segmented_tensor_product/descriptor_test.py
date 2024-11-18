# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cuequivariance.segmented_tensor_product as stp


def test_user_friendly():
    d = stp.SegmentedTensorProduct.from_subscripts("ia_jb_kab+ijk")
    assert str(d) == "ia,jb,kab+ijk sizes=0,0,0 num_segments=0,0,0 num_paths=0"

    with pytest.raises(ValueError):
        d.add_path(0, 0, 0, c=np.ones((2, 2, 3)))  # need to add segments first

    with pytest.raises(ValueError):
        d.add_segment(0, (2, 2, 2))  # wrong number of dimensions

    d.add_segment(0, (5, 16))
    d.add_segment(1, (5, 32))
    d.add_segment(2, (4, 16, 32))
    d.add_segment(2, (4, 32, 32))

    with pytest.raises(ValueError):
        d.add_path(0, 0, 0, c=np.ones((5, 5, 5)))  # wrong dimension for k

    assert (
        str(d)
        == "ia,jb,kab+ijk sizes=80,160,6144 num_segments=1,1,2 num_paths=0 a={16, 32} b=32 i=5 j=5 k=4"
    )

    d.add_path(0, 0, 0, c=np.ones((5, 5, 4)))
    assert (
        str(d)
        == "ia,jb,kab+ijk sizes=80,160,6144 num_segments=1,1,2 num_paths=1 a={16, 32} b=32 i=5 j=5 k=4"
    )

    assert d.all_segments_are_used() == False
    assert d.subscripts == "ia,jb,kab+ijk"
    assert d.subscripts.is_equivalent("ia,jb,kab+ijk")

    d.assert_valid()


def test_squeeze():
    d = stp.SegmentedTensorProduct.from_subscripts("i_j+ij")
    d.add_segment(0, (1,))
    d.add_segment(1, (20,))
    d.add_path(0, 0, c=np.ones((1, 20)))
    d.assert_valid()

    assert d.squeeze_modes().subscripts == ",j+j"

    d = stp.SegmentedTensorProduct.from_subscripts("i_j+ij")
    d.add_segment(0, (1,))
    d.add_segment(0, (2,))
    d.add_segment(1, (20,))
    d.add_path(0, 0, c=np.ones((1, 20)))
    d.add_path(1, 0, c=np.ones((2, 20)))
    d.assert_valid()

    assert d.squeeze_modes().subscripts == "i,j+ij"

    with pytest.raises(ValueError):
        d.squeeze_modes("i")


def test_normalize_paths_for_operand():
    d = stp.SegmentedTensorProduct.from_subscripts("i_j+ij")

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
    d = stp.SegmentedTensorProduct.from_subscripts("uv_iu_jv+ij")
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
    x2 = stp.compute_last_operand(d, x0, x1)

    for channels in ["i", "j", "ij", "ui", "iju", "uvij"]:
        np.testing.assert_allclose(
            x2, stp.compute_last_operand(d.flatten_modes(channels), x0, x1)
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
    d = stp.SegmentedTensorProduct.from_subscripts("ab_ab")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (2, 3))
    d.add_path(0, 0, c=1.0)
    d.assert_valid()

    assert d.consolidate_modes().subscripts == "a,a"

    d = stp.SegmentedTensorProduct.from_subscripts("ab_ab_a")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (2, 3))
    d.add_segment(2, (2,))
    d.add_path(0, 0, 0, c=1.0)
    d.assert_valid()

    assert d.consolidate_modes() == d

    d = stp.SegmentedTensorProduct.from_subscripts("ab_iab+abi")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (4, 2, 3))
    d.add_path(0, 0, c=np.ones((2, 3, 4)))
    d.assert_valid()

    assert d.consolidate_modes().subscripts == "a,ia+ai"

    d = stp.SegmentedTensorProduct.from_subscripts("ab,iab+abi")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (4, 2, 3))
    d.add_path(0, 0, c=np.ones((2, 3, 4)))

    assert d.consolidate_modes().subscripts == "a,ia+ai"


def test_stacked_coefficients():
    d = stp.SegmentedTensorProduct.from_subscripts("ab_ab+ab")
    d.add_segment(0, (2, 3))
    d.add_segment(1, (2, 3))
    np.testing.assert_allclose(d.stacked_coefficients, np.ones((0, 2, 3)))

    d.add_path(0, 0, c=np.ones((2, 3)))
    d.add_path(0, 0, c=np.ones((2, 3)))
    np.testing.assert_allclose(d.stacked_coefficients, np.ones((2, 2, 3)))

    d = d.consolidate_paths()
    np.testing.assert_allclose(d.stacked_coefficients, 2 * np.ones((1, 2, 3)))


@pytest.mark.parametrize("extended", [False, True])
def test_data_transfer(extended: bool):
    d = stp.SegmentedTensorProduct.from_subscripts("ui,uj,uk+ijk")
    d.add_path(None, None, None, c=np.ones((3, 3, 3)), dims={"u": 12})
    d.add_path(None, None, None, c=np.ones((1, 2, 1)), dims={"u": 14})
    d.assert_valid()

    dict = d.to_dict(extended)
    json = d.to_json(extended)
    bin = d.to_bytes(extended)
    b64 = d.to_base64(extended)

    assert d == stp.SegmentedTensorProduct.from_dict(dict)
    assert d == stp.SegmentedTensorProduct.from_json(json)
    assert d == stp.SegmentedTensorProduct.from_bytes(bin)
    assert d == stp.SegmentedTensorProduct.from_base64(b64)


def test_to_text():
    d = stp.SegmentedTensorProduct.from_subscripts("iu,ju,ku+ijk")
    d.add_path(None, None, None, c=np.ones((3, 3, 3)), dims={"u": 12})
    d.add_path(None, None, None, c=np.ones((1, 2, 1)), dims={"u": 14})
    d = d.flatten_modes("ijk")

    text = d.to_text()
    assert (
        text
        == """u,u,u sizes=50,64,50 num_segments=4,5,4 num_paths=29 u={12, 14}
operand #0 subscripts=u u: [12, 12, 12, 14]
operand #1 subscripts=u u: [12, 12, 12, 14, 14]
operand #2 subscripts=u u: [12, 12, 12, 14]
Flop cost: 0->704 1->704 2->704
Memory cost from 164 to 1056
Path indices: 0 0 0, 0 0 1, 0 0 2, 0 1 0, 0 1 1, 0 1 2, 0 2 0, 0 2 1, 0 2 2, 1 0 0, 1 0 1, 1 0 2, 1 1 0, 1 1 1, 1 1 2, 1 2 0, 1 2 1, 1 2 2, 2 0 0, 2 0 1, 2 0 2, 2 1 0, 2 1 1, 2 1 2, 2 2 0, 2 2 1, 2 2 2, 3 3 3, 3 4 3
Path coefficients: [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0]"""
    )


def test_hash():
    d = stp.SegmentedTensorProduct.from_subscripts("ui,uj,uk+ijk")
    d.add_path(None, None, None, c=np.ones((3, 3, 3)), dims={"u": 12})
    d.add_path(None, None, None, c=np.ones((1, 2, 1)), dims={"u": 14})
    assert hash(d) == hash(d)

    d2 = stp.SegmentedTensorProduct.from_subscripts("ui,uj,uk+ijk")
    assert hash(d) != hash(d2)
    d2.add_path(None, None, None, c=np.ones((3, 3, 3)), dims={"u": 12})
    assert hash(d) != hash(d2)
    d2.add_path(None, None, None, c=np.ones((1, 2, 1)), dims={"u": 14})
    assert hash(d) == hash(d2)
