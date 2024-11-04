# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import timeit

import pytest
import torch

import cuequivariance as cue
import cuequivariance.equivariant_tensor_product as etp
import cuequivariance_torch as cuet


def make_descriptors():
    # This ETP will trigger the fusedTP kernel
    yield etp.fully_connected_tensor_product(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "0e + 1o + 2e"),
        cue.Irreps("O3", "32x0e + 32x1o"),
    ).flatten_coefficient_modes()

    # This ETP will trigger the uniform1dx4 kernel
    yield etp.channelwise_tensor_product(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "0e + 1o + 2e"),
        cue.Irreps("O3", "0e + 1o"),
    ).flatten_coefficient_modes().squeeze_modes()

    # These ETPs will trigger the symmetricContraction kernel
    yield etp.spherical_harmonics(cue.SO3(1), [1, 2, 3])
    yield etp.symmetric_contraction(
        cue.Irreps("O3", "32x0e + 32x1o"), cue.Irreps("O3", "32x0e + 32x1o"), [1, 2, 3]
    )


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize(
    "dtype, math_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_performance_cuda_vs_fx(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
):
    device = torch.device("cuda:0")

    m = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=math_dtype,
        optimize_fallback=True,
    )

    inputs = [
        torch.randn((1024, inp.irreps.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]

    for _ in range(10):
        m(*inputs, use_fallback=False)
        m(*inputs, use_fallback=True)

    def f(ufb: bool):
        m(*inputs, use_fallback=ufb)
        torch.cuda.synchronize()

    t0 = timeit.Timer(lambda: f(False)).timeit(number=10)
    t1 = timeit.Timer(lambda: f(True)).timeit(number=10)
    assert t0 < t1


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize(
    "dtype, math_dtype, atol, rtol",
    [
        (torch.float16, torch.float32, 1, 0.2),
        (torch.bfloat16, torch.float32, 1, 0.2),
        (torch.float32, torch.float32, 1e-4, 1e-6),
        (torch.float32, torch.float64, 1e-5, 1e-6),
        (torch.float64, torch.float32, 1e-5, 1e-6),
        (torch.float64, torch.float64, 1e-12, 0),
    ],
)
def test_precision_cuda_vs_fx(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    device = torch.device("cuda:0")

    m = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=math_dtype,
        optimize_fallback=True,
    )
    inputs = [
        torch.randn((1024, inp.irreps.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]

    y0 = m(*inputs, use_fallback=False)
    y1 = m(*inputs, use_fallback=True)

    torch.testing.assert_close(y0, y1, atol=atol, rtol=rtol)