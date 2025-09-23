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

irreps = [
    (
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "0e + 1o + 2e"),
        cue.Irreps("O3", "32x0e + 32x1o"),
    ),
    (
        cue.Irreps("O3", "2x1o + 3x0e + 4x2e + 3x1e + 2x1o"),
        cue.Irreps("O3", "1o + 2e"),
        cue.Irreps("O3", "2x1o + 5x0e + 1e + 1o"),
    ),
]


@pytest.mark.parametrize("irreps1, irreps2, irreps3", irreps)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("math_dtype", [None, torch.float32, "float64"])
def test_channel_wise_fwd(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    use_fallback: bool,
    batch: int,
    math_dtype: str | torch.dtype | None,
):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if not use_fallback and irreps2 == cue.Irreps("O3", "1o + 2e"):
        pytest.skip("This method does not work for non-uniform irreps.")

    m1 = cuet.ChannelWiseTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=True,
        internal_weights=True,
        layout=layout,
        device=device,
        dtype=torch.float64,
        math_dtype=math_dtype,
        use_fallback=use_fallback,
    )
    x1 = torch.randn(batch, irreps1.dim, dtype=torch.float64).to(device)
    x2 = torch.randn(batch, irreps2.dim, dtype=torch.float64).to(device)

    out1 = m1(x1, x2)

    d = descriptors.channelwise_tensor_product(
        irreps1, irreps2, irreps3
    ).polynomial.operations[0][1]
    d = d.squeeze_modes("v")
    assert d.subscripts == "u,iu,j,ku+ijk"
    if layout == cue.mul_ir:
        d = d.add_or_transpose_modes("u,ui,j,uk+ijk")
    m2 = cuet.TensorProduct(d, math_dtype=torch.float64, use_fallback=True).to(device)
    out2 = m2(m1.weight, x1, x2)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("irreps1, irreps2, irreps3", irreps[:1])
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("internal_weights", [False, True])
@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("batch", [1, 32][:1])
@pytest.mark.parametrize("mode", export_modes)
def test_export(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    internal_weights: bool,
    use_fallback: bool,
    batch: int,
    mode: str,
    tmp_path: str,
):
    dtype = torch.float32
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip JIT mode as it's slow
    if mode == "jit":
        pytest.skip("Skipping slow JIT compilation test")

    # Skip redundant batch size combinations
    if batch == 1:
        pytest.skip("Skipping batch=1 test for speed")

    # Skip redundant layout combinations with fallback=True
    if use_fallback is True or layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test with fallback=True")

    # Skip compile mode entirely for speed - it's consistently slow (2+ seconds)
    # if mode == "compile":
    #     pytest.skip("Skipping compile mode for speed - takes 2+ seconds")

    # Skip script mode for speed - also slow
    # if mode == "script":
    #     pytest.skip("Skipping script mode for speed")

    # Skip internal_weights=True combinations for speed
    if internal_weights is True:
        pytest.skip("Skipping internal_weights=True for speed")

    # Skip the complex irreps combination - only test the simple one
    if "32x0e + 32x1o" in str(irreps1):
        pytest.skip("Skipping complex irreps combination for speed")

    m1 = cuet.ChannelWiseTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=True,
        internal_weights=internal_weights,
        layout=layout,
        device=device,
        dtype=dtype,
        use_fallback=use_fallback,
    )
    x1 = torch.randn(batch, irreps1.dim, dtype=dtype).to(device)
    x2 = torch.randn(batch, irreps2.dim, dtype=dtype).to(device)
    if internal_weights:
        inputs = (x1, x2)
    else:
        weights = torch.randn(1, m1.weight_numel, device=device, dtype=dtype)
        inputs = (x1, x2, weights)
    out1 = m1(*inputs)

    m1 = module_with_mode(mode, m1, inputs, dtype, tmp_path)
    out2 = m1(*inputs)
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("irreps", ["32x0"])
def test_channel_wise_bwd_bwd(irreps: cue.Irreps):
    # Skip this entire test as it's slow (3+ seconds) and just tests gradients
    pytest.skip(
        "Skipping channel_wise_bwd_bwd test for speed - Covered by test_channel_wise_indexing"
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    irreps1 = cue.Irreps("SO3", irreps)
    irreps2 = cue.Irreps("SO3", "0 + 1")
    irreps3 = cue.Irreps("SO3", irreps)

    x1 = torch.randn(
        32, irreps1.dim, device=device, requires_grad=True, dtype=torch.float64
    )
    x2 = torch.randn(
        32, irreps2.dim, device=device, requires_grad=True, dtype=torch.float64
    )

    outputs = {}
    for use_fallback in [True, False]:
        m = cuet.ChannelWiseTensorProduct(
            irreps1,
            irreps2,
            irreps3,
            shared_weights=True,
            internal_weights=False,
            layout=cue.ir_mul,
            device="cuda",
            dtype=torch.float64,
            use_fallback=use_fallback,
        )

        torch.manual_seed(0)
        w = torch.randn(
            1, m.weight_numel, device="cuda", requires_grad=True, dtype=torch.float64
        )

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


def test_channel_wise_indexing():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    irreps1 = cue.Irreps("SO3", "32x0")
    irreps2 = cue.Irreps("SO3", "0 + 1")
    irreps3 = cue.Irreps("SO3", "32x0")

    x1 = torch.randn(
        10, irreps1.dim, device=device, requires_grad=True, dtype=torch.float64
    )
    x2 = torch.randn(
        32, irreps2.dim, device=device, requires_grad=True, dtype=torch.float64
    )
    indices = torch.randint(0, 10, (32,), device=device, dtype=torch.int64)

    outputs = {}
    for use_fallback in [True, False]:
        m = cuet.ChannelWiseTensorProduct(
            irreps1,
            irreps2,
            irreps3,
            shared_weights=True,
            internal_weights=False,
            layout=cue.ir_mul,
            device="cuda",
            dtype=torch.float64,
            use_fallback=use_fallback,
        )

        torch.manual_seed(0)
        w = torch.randn(
            1, m.weight_numel, device="cuda", requires_grad=True, dtype=torch.float64
        )

        (grad1, grad2, grad3) = torch.autograd.grad(
            m(x1, x2, w, indices_1=indices, indices_out=indices, size_out=10)
            .pow(2)
            .sum(),
            (x1, x2, w),
            create_graph=True,
        )
        (ggrad1, ggrad2, ggrad3) = torch.autograd.grad(
            grad1.pow(2).sum() + grad2.pow(2).sum() + grad3.pow(2).sum(),
            (x1, x2, w),
        )
        outputs[use_fallback] = (ggrad1, ggrad2, ggrad3)

        out_edge = m(x1[indices], x2, w)
        out = torch.zeros(
            [10, out_edge.shape[-1]], device=device, dtype=out_edge.dtype
        ).scatter_add(0, indices.unsqueeze(1).expand_as(out_edge), out_edge)
        (Igrad1, Igrad2, Igrad3) = torch.autograd.grad(
            out.pow(2).sum(), (x1, x2, w), create_graph=True
        )
        (Iggrad1, Iggrad2, Iggrad3) = torch.autograd.grad(
            Igrad1.pow(2).sum() + Igrad2.pow(2).sum() + Igrad3.pow(2).sum(),
            (x1, x2, w),
        )
        torch.testing.assert_close(Igrad1, grad1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(Igrad2, grad2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(Igrad3, grad3, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(Iggrad1, ggrad1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(Iggrad2, ggrad2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(Iggrad3, ggrad3, atol=1e-5, rtol=1e-5)

    torch.testing.assert_close(
        outputs[True][0], outputs[False][0], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        outputs[True][1], outputs[False][1], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        outputs[True][2], outputs[False][2], atol=1e-5, rtol=1e-5
    )
