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
import cuequivariance.segmented_tensor_product as stp
import cuequivariance.equivariant_tensor_product as etp
import cuequivariance_torch as cuet
import itertools


def make_descriptors():
    yield etp.fully_connected_tensor_product(
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "6x0e + 6x1o"),
        cue.Irreps("O3", "5x0e + 5x1o + 5x2e + 5x1e"),
    ).d

    yield etp.spherical_harmonics(cue.SO3(1), [2]).d
    yield etp.spherical_harmonics(cue.SO3(1), [3]).d

    d = etp.channelwise_tensor_product(
        cue.Irreps("SU2", "3x1/2 + 4x1"),
        cue.Irreps("SU2", "1/2 + 1 + 3/2"),
        cue.Irreps("SU2", "1/2 + 1"),
    ).d
    yield d

    d = etp.channelwise_tensor_product(
        cue.Irreps("SO3", "32x1 + 32x2"),
        cue.Irreps("SO3", "0 + 1"),
        cue.Irreps("SO3", "0 + 1"),
    ).d
    yield d

    for subscripts in [
        "u__uw_w",
        "u_v_uv_u",
        "u_v_uv_v",
        "u_u_uw_w",
        "u_v_uvw_w",
        "_v_vw_w",
        "u_u_u",
        "u_v_uv",
        "u_uv_v",
        "u__u",
        "_v_v",
    ]:
        d = stp.SegmentedTensorProduct.from_subscripts(subscripts)
        for i in range(3):
            d.add_path(
                *[None] * d.num_operands,
                c=1.0,
                dims=dict(u=3 + i, v=6 - i, w=1 + 2 * i),
            )
        yield d
        yield d.move_operand_first(1)
        if d.num_operands == 4:
            yield d.move_operand_first(2)


@pytest.mark.parametrize("d", make_descriptors())
@pytest.mark.parametrize(
    "dtype, math_dtype, tol",
    [
        (torch.float16, torch.float32, 1.0),
        (torch.bfloat16, torch.float32, 1.0),
        (torch.float32, torch.float64, 1e-5),
        (torch.float64, torch.float32, 1e-5),
        (torch.float32, torch.float32, 1e-5),
        (torch.float64, torch.float64, 1e-12),
    ],
)
def test_primitive_tensor_product_cuda_vs_fx(
    d: stp.SegmentedTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    tol: float,
):
    # Make sure to run with CUDA_LAUNCH_BLOCKING=1 to catch the correct errors
    if (dtype, math_dtype) == (torch.float32, torch.float64):
        pytest.skip("no attribute 'fused_tensor_product_fwd_fp32_fp32_fp32_fp32_fp64'")

    device = torch.device("cuda:0")

    m = cuet.TensorProduct(
        d, device=device, math_dtype=math_dtype, optimize_fallback=False
    )

    for batches in itertools.product([(16,), (), (4, 1)], repeat=d.num_operands - 1):
        inputs = [
            torch.randn(
                batches[i] + (d.operands[i].size,),
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            for i in range(d.num_operands - 1)
        ]

        out1 = m(*inputs, use_fallback=False)
        out2 = m(*inputs, use_fallback=True)

        assert out1.shape[:-1] == torch.broadcast_shapes(*batches)
        assert out1.dtype == dtype
        assert out2.dtype == dtype

        torch.testing.assert_close(out1, out2, atol=tol, rtol=tol)

        grad1 = torch.autograd.grad(out1.sum(), inputs, create_graph=True)
        grad2 = torch.autograd.grad(out2.sum(), inputs, create_graph=True)

        for g1, g2 in zip(grad1, grad2):
            torch.testing.assert_close(g1, g2, atol=10 * tol, rtol=10 * tol)

        double_grad1 = torch.autograd.grad(sum(g.sum() for g in grad1), inputs)
        double_grad2 = torch.autograd.grad(sum(g.sum() for g in grad2), inputs)

        for g1, g2 in zip(double_grad1, double_grad2):
            torch.testing.assert_close(g1, g2, atol=100 * tol, rtol=100 * tol)
