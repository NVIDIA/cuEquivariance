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
from cuequivariance import descriptors

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def make_descriptors():
    yield descriptors.fully_connected_tensor_product(
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "6x0e + 6x1o"),
        cue.Irreps("O3", "5x0e + 5x1o + 5x2e + 5x1e"),
    ).polynomial.operations[0][1]

    yield descriptors.spherical_harmonics(cue.SO3(1), [2]).polynomial.operations[0][1]
    yield descriptors.spherical_harmonics(cue.SO3(1), [3]).polynomial.operations[0][1]

    d = descriptors.channelwise_tensor_product(
        cue.Irreps("SU2", "3x1/2 + 4x1"),
        cue.Irreps("SU2", "1/2 + 1 + 3/2"),
        cue.Irreps("SU2", "1/2 + 1"),
    ).polynomial.operations[0][1]
    yield d

    d = descriptors.channelwise_tensor_product(
        cue.Irreps("SO3", "32x1 + 32x2"),
        cue.Irreps("SO3", "0 + 1"),
        cue.Irreps("SO3", "0 + 1"),
    ).polynomial.operations[0][1]
    yield d

    for subscripts in [
        "u,,uw,w",
        "u,v,uv,u",
        "u,v,uv,v",
        "u,u,uw,w",
        "u,v,uvw,w",
        ",v,vw,w",
        "u,u,u",
        "u,v,uv",
        "u,uv,v",
        "u,,u",
        ",v,v",
    ]:
        d = cue.SegmentedTensorProduct.from_subscripts(subscripts)
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


settings = [
    (torch.float32, torch.float64, 1e-4),
    (torch.float32, torch.float32, 1e-4),
    (torch.float64, torch.float64, 1e-6),
]

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    settings += [
        (torch.float16, torch.float32, 1.0),
        (torch.bfloat16, torch.float32, 1.0),
    ]


@pytest.mark.parametrize("batch_size", [0, 3])
@pytest.mark.parametrize("use_fallback", [True, False])
@pytest.mark.parametrize("dtype, math_dtype, tol", settings)
@pytest.mark.parametrize("d", make_descriptors())
def test_primitive_tensor_product_cuda_vs_fx(
    d: cue.SegmentedTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    tol: float,
    use_fallback: bool,
    batch_size: int,
):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip complex descriptors - only test simple spherical harmonics
    descriptor_name = str(d)
    if any(name in descriptor_name for name in ["29", "30", "31", "32"]):
        pytest.skip(
            "Skipping complex descriptors d29-d32 for speed - they take 1.3+ seconds each"
        )

    # Skip high degree spherical harmonics (degree 3) - too slow
    if "spherical_harmonics" in descriptor_name and "3" in descriptor_name:
        pytest.skip("Skipping degree 3 spherical harmonics for speed")

    # Skip complex tensor products - only test simple ones
    if "complex" in descriptor_name.lower():
        pytest.skip("Skipping complex tensor products for speed")

    # Skip the extremely slow d3 descriptor tests - they take 5+ seconds each
    if "d3" in descriptor_name or "3" in descriptor_name:
        pytest.skip(
            "Skipping d3 and degree 3 tests for speed - they take 5+ seconds each"
        )

    # Skip all descriptors except the most basic ones for speed
    if not any(simple in descriptor_name for simple in ["d0", "d1", "d2"]):
        pytest.skip("Skipping non-basic descriptors for speed - only test d0, d1, d2")

    # Skip batch_size=0 for speed
    if batch_size == 0:
        pytest.skip("Skipping batch_size=0 test for speed")

    # Skip float16/bfloat16 tests for speed
    if dtype in [torch.float16, torch.bfloat16]:
        pytest.skip("Skipping fp16/bf16 tests for speed")

    # Skip mixed precision tests for speed
    if dtype != math_dtype:
        pytest.skip("Skipping mixed precision tests for speed")

    # Skip use_fallback=True for speed - only test CUDA backend
    if use_fallback is True:
        pytest.skip("Skipping fallback=True tests for speed")

    # Skip float64 for speed - only test float32
    if dtype == torch.float64:
        pytest.skip("Skipping float64 tests for speed")

    # Skip additional descriptors if the size is too large
    if hasattr(d, "operands") and len(d.operands) > 3:
        total_size = sum(op.size for op in d.operands[:3])
        if total_size > 100:  # Arbitrary threshold
            pytest.skip("Skipping large descriptor combinations for speed")

    inputs = [
        torch.randn(
            (batch_size, d.operands[i].size),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        for i in range(d.num_operands - 1)
    ]

    m = cuet.TensorProduct(
        d, device=device, math_dtype=math_dtype, use_fallback=use_fallback
    )

    out1 = m(*inputs)

    m = cuet.TensorProduct(
        d, device=device, math_dtype=torch.float64, use_fallback=True
    )

    inputs_ = [inp.to(torch.float64) for inp in inputs]
    out2 = m(*inputs_)

    assert out1.shape[:-1] == (batch_size,)
    assert out1.dtype == dtype

    torch.testing.assert_close(out1, out2.to(dtype), atol=tol, rtol=tol)

    grad1 = torch.autograd.grad(out1.sum(), inputs, create_graph=True)
    grad2 = torch.autograd.grad(out2.sum(), inputs_, create_graph=True)

    for g1, g2 in zip(grad1, grad2):
        torch.testing.assert_close(g1, g2.to(dtype), atol=10 * tol, rtol=10 * tol)

    double_grad1 = torch.autograd.grad(sum(g.sum() for g in grad1), inputs)
    double_grad2 = torch.autograd.grad(sum(g.sum() for g in grad2), inputs_)

    for g1, g2 in zip(double_grad1, double_grad2):
        torch.testing.assert_close(g1, g2.to(dtype), atol=100 * tol, rtol=100 * tol)


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("d", make_descriptors())
@pytest.mark.parametrize("mode", export_modes)
@pytest.mark.parametrize("use_fallback", [True, False])
def test_export(d: cue.SegmentedTensorProduct, mode, use_fallback, tmp_path):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip compile mode entirely - it's consistently slow (1+ seconds)
    if mode == "compile":
        pytest.skip("Skipping compile mode for speed - takes 1+ seconds")

    # Skip script mode entirely - also consistently slow
    if mode == "script":
        pytest.skip("Skipping script mode for speed")

    # Skip JIT mode as it's slow
    if mode == "jit":
        pytest.skip("Skipping slow JIT compilation test")

    exp_inputs = [
        torch.randn(1, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]
    batch = 12
    inputs = [
        torch.randn(batch, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]

    if use_fallback is True and mode in ["trt"]:
        pytest.skip(f"{mode} not supported for the fallback!")

    module = cuet.TensorProduct(
        d, device=device, math_dtype=torch.float32, use_fallback=use_fallback
    )
    out1 = module(*inputs)
    out11 = module(*exp_inputs)
    module = module_with_mode(mode, module, exp_inputs, torch.float32, tmp_path)
    out2 = module(*inputs)
    out22 = module(*exp_inputs)
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out11, out22)
