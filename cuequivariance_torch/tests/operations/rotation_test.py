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
import pytest
import torch
from cuequivariance_torch._tests.utils import (
    module_with_mode,
)

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def test_rotation():
    irreps = cue.Irreps("SO3", "3x0 + 1 + 0 + 4x2 + 4")
    alpha = torch.tensor([0.3]).to(device)
    beta = torch.tensor([0.4]).to(device)
    gamma = torch.tensor([-0.5]).to(device)

    rot = cuet.Rotation(irreps, layout=cue.ir_mul).to(device)

    x = torch.randn(10, irreps.dim).to(device)

    rx = rot(gamma, beta, alpha, x)
    x_ = rot(-alpha, -beta, -gamma, rx)

    torch.testing.assert_close(x, x_)


def test_vector_to_euler_angles():
    A = torch.randn(4, 3)
    A = torch.nn.functional.normalize(A, dim=-1)

    beta, alpha = cuet.vector_to_euler_angles(A)
    ey = torch.tensor([[0.0, 1.0, 0.0]])
    B = cuet.Rotation(cue.Irreps("SO3", "1"), layout=cue.ir_mul)(
        torch.tensor([0.0]), beta, alpha, ey
    )

    assert torch.allclose(A, B, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_fallback", [False, True])
def test_inversion(use_fallback: bool):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    irreps = cue.Irreps("O3", "2x1e + 2x1o")
    torch.testing.assert_close(
        cuet.Inversion(
            irreps, layout=cue.ir_mul, device=device, use_fallback=use_fallback
        )(
            torch.tensor(
                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                device=device,
            )
        ),
        torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]],
            device=device,
        ),
    )


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("mode", export_modes)
def test_export(mode: str, tmp_path: str):
    # Skip all export tests for rotation - they are very slow (10+ seconds each)
    pytest.skip(
        "Skipping all rotation export tests for speed - they take 10+ seconds each"
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    irreps = cue.Irreps("SO3", "3x0 + 1 + 0 + 4x2 + 4")
    dtype = torch.float32
    alpha = torch.tensor([0.3]).to(device)
    beta = torch.tensor([0.4]).to(device)
    gamma = torch.tensor([-0.5]).to(device)

    m = cuet.Rotation(irreps, layout=cue.ir_mul).to(device)

    x = torch.randn(10, irreps.dim).to(device)
    inputs = (gamma, beta, alpha, x)

    out1 = m(*inputs)
    m = module_with_mode(mode, m, inputs, dtype, tmp_path)
    out2 = m(*inputs)
    torch.testing.assert_close(out1, out2)
