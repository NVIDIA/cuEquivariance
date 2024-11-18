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

list_of_irreps = [
    cue.Irreps("O3", "4x0e + 4x1o"),
    cue.Irreps("O3", "2x1o + 5x0e + 2e + 1e + 1o"),
    cue.Irreps("O3", "2e + 0x0e + 0o + 0x1e + 1e"),
]


@pytest.mark.parametrize("irreps1", list_of_irreps)
@pytest.mark.parametrize("irreps2", [irreps.set_mul(1) for irreps in list_of_irreps])
@pytest.mark.parametrize("irreps3", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("use_fallback", [False, True])
def test_channel_wise(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    use_fallback: bool,
):
    m = cuet.ChannelWiseTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=True,
        internal_weights=True,
        layout=layout,
        device="cuda",
        dtype=torch.float64,
    )

    x1 = torch.randn(32, irreps1.dim, dtype=torch.float64).cuda()
    x2 = torch.randn(32, irreps2.dim, dtype=torch.float64).cuda()

    out1 = m(x1, x2, use_fallback=use_fallback)

    d = descriptors.channelwise_tensor_product(irreps1, irreps2, irreps3).d
    d = d.squeeze_modes("v")
    assert d.subscripts == "u,iu,j,ku+ijk"
    if layout == cue.mul_ir:
        d = d.add_or_transpose_modes("u,ui,j,uk+ijk")
    mfx = cuet.TensorProduct(d, math_dtype=torch.float64).cuda()
    out2 = mfx(m.weight, x1, x2, use_fallback=True)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


def test_channel_wise_bwd_bwd():
    irreps1 = cue.Irreps("SO3", "2x0 + 3x1")
    irreps2 = cue.Irreps("SO3", "0 + 1")
    irreps3 = cue.Irreps("SO3", "0 + 1")

    m = cuet.ChannelWiseTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=True,
        internal_weights=False,
        layout=cue.ir_mul,
        device="cuda",
        dtype=torch.float64,
    )

    x1 = torch.randn(
        32, irreps1.dim, device="cuda", requires_grad=True, dtype=torch.float64
    )
    x2 = torch.randn(
        32, irreps2.dim, device="cuda", requires_grad=True, dtype=torch.float64
    )
    w = torch.randn(
        m.weight_numel, device="cuda", requires_grad=True, dtype=torch.float64
    )

    outputs = {}
    for use_fallback in [True, False]:
        (grad1, grad2, grad3) = torch.autograd.grad(
            m(x1, x2, w).pow(2).sum(), (x1, x2, w), create_graph=True
        )
        (ggrad1, ggrad2, ggrad3) = torch.autograd.grad(
            grad1.pow(2).sum() + grad2.pow(2).sum() + grad3.pow(2).sum(),
            (x1, x2, w),
        )
        outputs[use_fallback] = (ggrad1, ggrad2, ggrad3)

    torch.testing.assert_close(
        outputs[True][0], outputs[False][0], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        outputs[True][1], outputs[False][1], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        outputs[True][2], outputs[False][2], atol=1e-5, rtol=1e-5
    )
