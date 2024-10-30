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
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


@pytest.mark.parametrize(
    "dtype, tol",
    [(torch.float64, 1e-6), (torch.float32, 1e-4)],
)
@pytest.mark.parametrize("l", [1, 2, 3])
def test_spherical_harmonics(l: int, dtype, tol):
    vec = torch.randn(3, dtype=dtype)
    axis = np.random.randn(3)
    angle = np.random.rand()
    scale = 1.3

    yl = cuet.spherical_harmonics([l], vec, False)

    R = torch.from_numpy(cue.SO3(1).rotation(axis, angle)).to(dtype)
    Rl = torch.from_numpy(cue.SO3(l).rotation(axis, angle)).to(dtype)

    yl1 = cuet.spherical_harmonics([l], scale * R @ vec, False)
    yl2 = scale**l * Rl @ yl

    torch.testing.assert_close(yl1, yl2, rtol=tol, atol=tol)


def test_spherical_harmonics_full():
    vec = torch.randn(3)
    ls = [0, 1, 2, 3]
    yl = cuet.spherical_harmonics(ls, vec, False)

    assert abs(yl[0] - 1.0) < 1e-6
