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
import itertools

import pytest
import torch

import cuequivariance as cue
import cuequivariance.segmented_tensor_product as stp
import cuequivariance_torch as cuet
from cuequivariance import descriptors


def make_descriptors():
    yield descriptors.fully_connected_tensor_product(
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "6x0e + 6x1o"),
        cue.Irreps("O3", "5x0e + 5x1o + 5x2e + 5x1e"),
    ).d

    yield descriptors.spherical_harmonics(cue.SO3(1), [2]).d
    yield descriptors.spherical_harmonics(cue.SO3(1), [3]).d

    d = descriptors.channelwise_tensor_product(
        cue.Irreps("SU2", "3x1/2 + 4x1"),
        cue.Irreps("SU2", "1/2 + 1 + 3/2"),
        cue.Irreps("SU2", "1/2 + 1"),
    ).d
    yield d

    d = descriptors.channelwise_tensor_product(
        cue.Irreps("SO3", "32x1 + 32x2"),
        cue.Irreps("SO3", "0 + 1"),
        cue.Irreps("SO3", "0 + 1"),
    ).d
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


settings = [
    (torch.float32, torch.float64, 1e-5),
    (torch.float64, torch.float32, 1e-5),
    (torch.float32, torch.float32, 1e-5),
    (torch.float64, torch.float64, 1e-12),
]

if torch.cuda.get_device_capability()[0] >= 8:
    settings += [
        (torch.float16, torch.float32, 1.0),
        (torch.bfloat16, torch.float32, 1.0),
    ]


@pytest.mark.parametrize("d", make_descriptors())
@pytest.mark.parametrize("dtype, math_dtype, tol", settings)
def test_primitive_tensor_product_cuda_vs_fx(
    d: stp.SegmentedTensorProduct,
    dtype: torch.dtype,
    math_dtype: torch.dtype,
    tol: float,
):
    device = torch.device("cuda:0")

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

        m = cuet.TensorProduct(
            d, device=device, math_dtype=math_dtype, optimize_fallback=False
        )
        out1 = m(inputs, use_fallback=False)
        m = cuet.TensorProduct(
            d, device=device, math_dtype=torch.float64, optimize_fallback=False
        )
        inputs_ = [inp.clone().to(torch.float64) for inp in inputs]
        out2 = m(inputs_, use_fallback=True)

        assert out1.shape[:-1] == torch.broadcast_shapes(*batches)
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
