# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


def test_rotation():
    irreps = cue.Irreps("SO3", "3x0 + 1 + 0 + 4x2 + 4")
    alpha = torch.tensor(0.3).cuda()
    beta = torch.tensor(0.4).cuda()
    gamma = torch.tensor(-0.5).cuda()

    rot = cuet.Rotation(irreps, layout=cue.ir_mul).cuda()

    x = torch.randn(10, irreps.dim).cuda()

    rx = rot(gamma, beta, alpha, x)
    x_ = rot(-alpha, -beta, -gamma, rx)

    torch.testing.assert_close(x, x_)


def test_vector_to_euler_angles():
    A = torch.randn(4, 3)
    A = torch.nn.functional.normalize(A, dim=-1)

    beta, alpha = cuet.vector_to_euler_angles(A)
    ey = torch.tensor([0.0, 1.0, 0.0])
    B = cuet.Rotation(cue.Irreps("SO3", "1"), layout=cue.ir_mul)(0.0, beta, alpha, ey)

    assert torch.allclose(A, B)


def test_inversion():
    irreps = cue.Irreps("O3", "2x1e + 1o")
    torch.testing.assert_close(
        cuet.Inversion(irreps, layout=cue.ir_mul)(
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
    )
