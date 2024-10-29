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


def make_descriptors():
    yield etp.symmetric_contraction(
        cue.Irreps("SO3", "0 + 1 + 2"), cue.Irreps("SO3", "0"), [3]
    ).ds
    yield etp.symmetric_contraction(
        cue.Irreps("O3", "0e + 1o + 2e"), cue.Irreps("O3", "0e + 1o"), [4]
    ).ds
    yield etp.symmetric_contraction(
        cue.Irreps("SU2", "0 + 1/2"), cue.Irreps("SU2", "0 + 1/2"), [5]
    ).ds

    d1 = stp.SegmentedTensorProduct.from_subscripts(",,")
    d1.add_path(None, None, None, c=2.0)

    d3 = stp.SegmentedTensorProduct.from_subscripts(",,,,")
    d3.add_path(None, None, None, None, None, c=3.0)

    yield [d1, d3]


@pytest.mark.parametrize("ds", make_descriptors())
@pytest.mark.parametrize(
    "dtype, math_dtype, tol",
    [
        (torch.float64, torch.float64, 1e-12),
        (torch.float32, torch.float32, 1e-5),
        (torch.float32, torch.float64, 1e-5),
        (torch.float16, torch.float32, 1.0),
        (torch.float16, torch.float64, 0.1),
        (torch.bfloat16, torch.float32, 1.0),
        (torch.bfloat16, torch.float64, 0.5),
    ],
)
def test_primitive_indexed_symmetric_tensor_product_cuda_vs_fx(
    ds: list[stp.SegmentedTensorProduct], dtype, math_dtype, tol: float
):
    device = torch.device("cuda:0")

    m = cuet.IWeightedSymmetricTensorProduct(
        ds, math_dtype=math_dtype, device=device, optimize_fallback=False
    )

    x0 = torch.randn((2, m.x0_size), device=device, dtype=dtype, requires_grad=True)
    i0 = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
    x1 = torch.randn(
        (i0.size(0), m.x1_size), device=device, dtype=dtype, requires_grad=True
    )

    out1 = m(x0, i0, x1, use_fallback=False)
    out2 = m(x0, i0, x1, use_fallback=True)

    assert out1.dtype == dtype
    assert out2.dtype == dtype

    torch.testing.assert_close(out1, out2, atol=tol, rtol=tol)

    grad1 = torch.autograd.grad(out1.sum(), (x0, x1), create_graph=True)
    grad2 = torch.autograd.grad(out2.sum(), (x0, x1), create_graph=True)

    for g1, g2 in zip(grad1, grad2):
        torch.testing.assert_close(g1, g2, atol=10 * tol, rtol=10 * tol)

    double_grad1 = torch.autograd.grad(sum(g.sum() for g in grad1), (x0, x1))
    double_grad2 = torch.autograd.grad(sum(g.sum() for g in grad2), (x0, x1))

    for g1, g2 in zip(double_grad1, double_grad2):
        torch.testing.assert_close(g1, g2, atol=100 * tol, rtol=100 * tol)


@pytest.mark.parametrize(
    "dtype, math_dtype",
    [
        (torch.float64, torch.float64),
        (torch.float32, torch.float32),
        (torch.float16, torch.float32),
        (torch.float32, torch.float64),
        (torch.bfloat16, torch.float32),
    ],
)
def test_math_dtype(
    dtype: torch.dtype,
    math_dtype: torch.dtype,
):
    device = torch.device("cuda:0")

    ds = etp.symmetric_contraction(
        cue.Irreps("SO3", "0 + 1 + 2"), cue.Irreps("SO3", "0"), [1, 2, 3]
    ).ds
    m = cuet.IWeightedSymmetricTensorProduct(ds, math_dtype=math_dtype, device=device)
    x0 = torch.randn((20, m.x0_size), dtype=dtype, device=device)
    i0 = torch.randint(0, m.x0_size, (1000,), dtype=torch.int32, device=device)
    x1 = torch.randn((i0.size(0), m.x1_size), dtype=dtype, device=device)

    out1 = m(x0, i0, x1, use_fallback=False)

    # .to should have no effect
    for param in m.parameters():
        assert False  # no parameters
    m = m.to(torch.float16)

    out2 = m(x0, i0, x1, use_fallback=False)

    assert out1.dtype == dtype
    assert out2.dtype == dtype
    assert (out1 == out2).all()
