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
import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.parametrize(
    "irreps1, irreps2, irreps3",
    [
        (
            cue.Irreps("O3", "4x0e + 4x1o"),
            cue.Irreps("O3", "4x0e + 4x1o"),
            cue.Irreps("O3", "4x0e + 4x1o"),
        ),
        (
            cue.Irreps("O3", "2e + 0x0e + 0o + 0x1e + 1e"),
            cue.Irreps("O3", "4x0e + 4x1o"),
            cue.Irreps("O3", "2e + 0x0e + 0o + 0x1e + 1e"),
        ),
    ],
)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("use_fallback", [False, True])
def test_fully_connected(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    use_fallback: bool,
):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m1 = cuet.FullyConnectedTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=True,
        internal_weights=True,
        layout=layout,
        device=device,
        dtype=torch.float64,
        use_fallback=use_fallback,
    )

    x1 = torch.randn(32, irreps1.dim, dtype=torch.float64).to(device)
    x2 = torch.randn(32, irreps2.dim, dtype=torch.float64).to(device)

    out1 = m1(x1, x2)

    d = descriptors.fully_connected_tensor_product(irreps1, irreps2, irreps3).d
    if layout == cue.mul_ir:
        d = d.add_or_transpose_modes("uvw,ui,vj,wk+ijk")
    m2 = cuet.TensorProduct(d, math_dtype=torch.float64, use_fallback=True).to(device)
    out2 = m2(
        [m1.weight.to(torch.float64), x1.to(torch.float64), x2.to(torch.float64)],
    ).to(out1.dtype)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_fallback", [False, True])
def test_compile(use_fallback: bool):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.FullyConnectedTensorProduct(
        irreps_in1=cue.Irreps("O3", "32x0e + 32x1o"),
        irreps_in2=cue.Irreps("O3", "32x0e + 32x1o"),
        irreps_out=cue.Irreps("O3", "32x0e + 32x1o"),
        layout=cue.mul_ir,
        device=device,
        use_fallback=use_fallback,
    )

    m_compile = torch.compile(m, fullgraph=True)
    input1 = torch.randn(100, m.irreps_in1.dim, device=device)
    input2 = torch.randn(100, m.irreps_in2.dim, device=device)
    m_compile(input1, input2)
