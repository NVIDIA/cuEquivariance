# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import pytest
import torch

import cuequivariance as cue
import cuequivariance.equivariant_tensor_product as etp
import cuequivariance_torch as cuet

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
    )

    x1 = torch.randn(32, irreps1.dim).cuda()
    x2 = torch.randn(32, irreps2.dim).cuda()

    out1 = m(x1, x2, use_fallback=use_fallback)

    d = etp.channelwise_tensor_product(irreps1, irreps2, irreps3).d
    d = d.squeeze_modes("v")
    assert d.subscripts == "u,iu,j,ku+ijk"
    if layout == cue.mul_ir:
        d = d.add_or_transpose_modes("u,ui,j,uk+ijk")
    mfx = cuet.TensorProduct(d).cuda()
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
    )

    x1 = torch.randn(32, irreps1.dim, device="cuda", requires_grad=True)
    x2 = torch.randn(32, irreps2.dim, device="cuda", requires_grad=True)
    w = torch.randn(m.weight_numel, device="cuda", requires_grad=True)

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
        outputs[True][0], outputs[False][0], atol=1e-3, rtol=1e-5
    )
    torch.testing.assert_close(
        outputs[True][1], outputs[False][1], atol=1e-3, rtol=1e-5
    )
    torch.testing.assert_close(
        outputs[True][2], outputs[False][2], atol=1e-3, rtol=1e-5
    )
