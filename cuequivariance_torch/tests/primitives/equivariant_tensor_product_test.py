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
import torch._dynamo
from tests.utils import (
    module_with_mode,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
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
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=math_dtype,
        use_fallback=False,
    )

    m1 = cuet.EquivariantTensorProduct(
        e,
        layout=cue.ir_mul,
        device=device,
        math_dtype=math_dtype,
        use_fallback=True,
    )

    inputs = [
        torch.randn((1024, inp.dim), device=device, dtype=dtype) for inp in e.inputs
    ]

    for _ in range(10):
        m(inputs)
        m1(inputs)
    torch.cuda.synchronize()

    def f():
        ret = m(inputs)
        ret = torch.sum(ret)
        return ret

    def f1():
        ret = m1(inputs)
        ret = torch.sum(ret)
        return ret

    t0 = timeit.Timer(f).timeit(number=10)
    t1 = timeit.Timer(f1).timeit(number=10)
    assert t0 < t1


settings2 = [
    (torch.float32, torch.float32, 1e-4, 1e-6),
    (torch.float32, torch.float64, 1e-5, 1e-6),
    (torch.float64, torch.float32, 1e-5, 1e-6),
    (torch.float64, torch.float64, 1e-12, 0),
]
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    settings2 += [
        (torch.float16, torch.float32, 1, 0.2),
        (torch.bfloat16, torch.float32, 1, 0.2),
    ]


@pytest.mark.parametrize("batch_size", [0, 5])
@pytest.mark.parametrize("dtype, math_dtype, atol, rtol", settings2)
@pytest.mark.parametrize("e", make_descriptors())
def test_precision_cuda_vs_fx(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    atol: float,
    rtol: float,
    batch_size: int,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    inputs = [
        torch.randn((batch_size, inp.dim), device=device, dtype=dtype)
        for inp in e.inputs
    ]
    m = cuet.EquivariantTensorProduct(
        e, layout=cue.ir_mul, device=device, math_dtype=math_dtype, use_fallback=False
    )
    y0 = m(inputs)

    m = cuet.EquivariantTensorProduct(
        e, layout=cue.ir_mul, device=device, math_dtype=torch.float64, use_fallback=True
    )
    inputs = [x.to(torch.float64) for x in inputs]
    y1 = m(inputs).to(dtype)

    torch.testing.assert_close(y0, y1, atol=atol, rtol=rtol)


export_modes = ["compile", "script", "jit"]
# "export" does not support the change of batch size


@pytest.mark.parametrize("e", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype, atol, rtol", settings2)
@pytest.mark.parametrize("use_fallback", [True, False])
@pytest.mark.parametrize("mode", export_modes)
def test_export(
    e: cue.EquivariantTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    atol: float,
    rtol: float,
    mode: str,
    use_fallback: bool,
    tmp_path,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.EquivariantTensorProduct(
        e,
        layout=cue.mul_ir,
        math_dtype=math_dtype,
        use_fallback=use_fallback,
        device=device,
    )
    inputs = [
        torch.randn((512, inp.dim), device=device, dtype=dtype) for inp in e.inputs
    ]
    res = m(inputs)
    m = module_with_mode(mode, m, [inputs], math_dtype, tmp_path)
    res_script = m(inputs)
    torch.testing.assert_close(res, res_script, atol=atol, rtol=rtol)
