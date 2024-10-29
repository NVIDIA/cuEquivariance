# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import numpy as np

import cuequivariance as cue
import cuequivariance.segmented_tensor_product as stp
import cuequivariance.equivariant_tensor_product as etp


def test_dot1():
    d1 = stp.SegmentedTensorProduct.from_subscripts("iab_jb_ak+ijk")
    d1.add_path(None, None, None, c=np.random.randn(2, 2, 2), dims={"a": 2, "b": 3})
    d1.add_path(None, None, None, c=np.random.randn(2, 3, 2), dims={"a": 4, "b": 3})
    d1.add_path(0, 1, 0, c=np.random.randn(2, 3, 2))

    d2 = stp.SegmentedTensorProduct.from_subscripts("jb_b_+j")
    d2.add_path(None, None, None, c=np.random.randn(2), dims={"b": 3})
    d2.add_path(None, 0, None, c=np.random.randn(3))

    d3 = stp.dot(d1, d2, (1, 0))
    assert d3.subscripts == "iab,ak,b,+ijk"

    x0, x2 = np.random.randn(d1.operands[0].size), np.random.randn(d1.operands[2].size)
    y1 = np.random.randn(d2.operands[1].size)

    tmp = stp.compute_last_operand(d1.move_operand_last(1), x0, x2)
    z0 = stp.compute_last_operand(d2, tmp, y1)

    z1 = stp.compute_last_operand(d3, x0, x2, y1)

    np.testing.assert_allclose(z0, z1)


def make_examples():
    irreps_middle = cue.Irreps("SO3", "2x0 + 3x1")
    dx = etp.fully_connected_tensor_product(
        cue.Irreps("SO3", "4x0 + 3x1"),
        cue.Irreps("SO3", "3x0 + 5x1"),
        irreps_middle,
    ).d
    assert dx.subscripts == "uvw,iu,jv,kw+ijk"
    dy = etp.channelwise_tensor_product(
        irreps_middle, cue.Irreps("SO3", "0 + 1 + 2"), cue.Irreps("SO3", "0 + 1")
    ).d
    dy = dy.squeeze_modes("v")
    assert dy.subscripts == "u,iu,j,ku+ijk"
    dy = dy.add_or_rename_modes("w_kw_l_mw+klm")
    return dx, dy


def test_dot2():
    dx, dy = make_examples()

    dxy = stp.dot(dx, dy, (3, 1))
    assert dxy.subscripts == "uvw,iu,jv,w,l,mw+ijklm"

    x0, x1, x2 = [np.random.randn(dx.operands[i].size) for i in range(3)]
    y0, y2 = [np.random.randn(dy.operands[i].size) for i in [0, 2]]

    tmp = stp.compute_last_operand(dx, x0, x1, x2)
    A = stp.compute_last_operand(dy, y0, tmp, y2)

    B = stp.compute_last_operand(dxy, x0, x1, x2, y0, y2)

    np.testing.assert_allclose(A, B)


def test_trace():
    dx, dy = make_examples()

    d1 = stp.dot(dx, dy, (3, 1))
    d1 = d1.canonicalize_subscripts()
    d1 = d1.sort_paths()

    assert dy.subscripts == "w,kw,l,mw+klm"
    dy = dy.add_or_rename_modes("a_xa_y_za+xyz")
    d2 = stp.trace(stp.dot(dx, dy), (3, 4 + 1))
    d2 = d2.canonicalize_subscripts()
    d2 = d2.sort_paths()

    assert d1.subscripts == d2.subscripts
    assert d1.operands == d2.operands
    np.testing.assert_allclose(d1.indices, d2.indices)
    assert len(d1.paths) == len(d2.paths)

    for path1, path2 in zip(d1.paths, d2.paths):
        assert path1.indices == path2.indices
        np.testing.assert_allclose(path1.coefficients, path2.coefficients)
