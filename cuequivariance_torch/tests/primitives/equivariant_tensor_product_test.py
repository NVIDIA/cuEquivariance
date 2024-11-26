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
import timeit

import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


def make_descriptors():
    # This ETP will trigger the fusedTP kernel
    yield descriptors.fully_connected_tensor_product(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "0e + 1o + 2e"),
        cue.Irreps("O3", "32x0e + 32x1o"),
    ).flatten_coefficient_modes()

    # This ETP will trigger the uniform1dx4 kernel
    yield (
        descriptors.channelwise_tensor_product(
            cue.Irreps("O3", "32x0e + 32x1o"),
            cue.Irreps("O3", "0e + 1o + 2e"),
            cue.Irreps("O3", "0e + 1o"),
        )
        .flatten_coefficient_modes()
        .squeeze_modes()
    )

    # These ETPs will trigger the symmetricContraction kernel
    yield descriptors.spherical_harmonics(cue.SO3(1), [1, 2, 3])
    yield descriptors.symmetric_contraction(
        cue.Irreps("O3", "32x0e + 32x1o"), cue.Irreps("O3", "32x0e + 32x1o"), [1, 2, 3]
    )


settings1 = [
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
]
if torch.cuda.get_device_capability()[0] >= 8:
    settings1 += [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype", settings1)
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


settings2 = [
    (torch.float32, torch.float32, 1e-4, 1e-6),
    (torch.float32, torch.float64, 1e-5, 1e-6),
    (torch.float64, torch.float32, 1e-5, 1e-6),
    (torch.float64, torch.float64, 1e-12, 0),
]
if torch.cuda.get_device_capability()[0] >= 8:
    settings2 += [
        (torch.float16, torch.float32, 1, 0.2),
        (torch.bfloat16, torch.float32, 1, 0.2),
    ]


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype, atol, rtol", settings2)
def test_precision_cuda_vs_fx(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    device = torch.device("cuda:0")

    inputs = [
        torch.randn((1024, inp.irreps.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]
    m = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=math_dtype,
    )
    y0 = m(*inputs, use_fallback=False)

    m = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=torch.float64,
        optimize_fallback=True,
    )
    inputs = map(lambda x: x.to(torch.float64), inputs)
    y1 = m(*inputs, use_fallback=True).to(dtype)

    torch.testing.assert_close(y0, y1, atol=atol, rtol=rtol)


def test_compile():
    e = cue.descriptors.symmetric_contraction(
        cue.Irreps("O3", "32x0e + 32x1o"), cue.Irreps("O3", "32x0e + 32x1o"), [1, 2, 3]
    )
    m = cuet.EquivariantTensorProduct(e, layout=cue.mul_ir, optimize_fallback=False)
    m_compile = torch.compile(m, fullgraph=True)
    input1 = torch.randn(100, e.inputs[0].irreps.dim)
    input2 = torch.randn(100, e.inputs[1].irreps.dim)
    m_compile(input1, input2)
