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
import pytest

import cuequivariance as cue
from cuequivariance import descriptors
import cuequivariance.segmented_tensor_product as stp


def test_commutativity_squeeze_flatten():
    irreps1 = cue.Irreps("O3", "32x0e + 32x1o")
    irreps2 = cue.Irreps("O3", "1x0e + 1x1o")
    irreps3 = cue.Irreps("O3", "32x0e + 32x1o")

    d = descriptors.fully_connected_tensor_product(irreps1, irreps2, irreps3).d
    assert (
        d.squeeze_modes().flatten_coefficient_modes()
        == d.flatten_coefficient_modes().squeeze_modes()
    )

    d = descriptors.channelwise_tensor_product(irreps1, irreps2, irreps3).d
    assert (
        d.squeeze_modes().flatten_coefficient_modes()
        == d.flatten_coefficient_modes().squeeze_modes()
    )

    d = descriptors.linear(irreps1, irreps2).d
    assert (
        d.squeeze_modes().flatten_coefficient_modes()
        == d.flatten_coefficient_modes().squeeze_modes()
    )


@pytest.mark.parametrize("l", [1, 2, 3, 4])
def test_spherical_harmonics(l: int):
    d = descriptors.spherical_harmonics(cue.SO3(1), [l]).d

    vec = np.random.randn(3)
    axis = np.random.randn(3)
    angle = np.random.rand()

    yl = stp.compute_last_operand(d, *(vec,) * l)

    R = cue.SO3(1).rotation(axis, angle)
    Rl = cue.SO3(l).rotation(axis, angle)

    yl1 = stp.compute_last_operand(d, *(R @ vec,) * l)
    yl2 = Rl @ yl

    np.testing.assert_allclose(yl1, yl2)
    np.testing.assert_allclose(np.sum(yl**2), (2 * l + 1) * np.sum(vec**2) ** l)


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_y_rotation(l: int):
    alpha = 0.3
    beta = 0.4
    gamma = -0.5

    irrep = cue.SO3(l)
    d = descriptors.yxy_rotation(cue.Irreps("SO3", [irrep])).d

    def enc(th: float):
        m = np.arange(1, l + 1)
        c = np.cos(m * th)
        s = np.sin(m * th)
        return np.concatenate([c[::-1], [1.0], s])

    x = np.random.randn(irrep.dim)
    y1 = stp.compute_last_operand(d, enc(gamma), enc(beta), enc(alpha), x)

    A = irrep.rotation(np.array([0.0, 1.0, 0.0]), alpha)
    B = irrep.rotation(np.array([1.0, 0.0, 0.0]), beta)
    C = irrep.rotation(np.array([0.0, 1.0, 0.0]), gamma)
    y2 = A @ B @ C @ x

    np.testing.assert_allclose(y1, y2)
