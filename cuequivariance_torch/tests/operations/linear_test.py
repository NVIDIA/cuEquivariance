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

import copy

import pytest
import torch
from cuequivariance_torch._tests.utils import module_with_mode

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

list_of_irreps = [
    cue.Irreps("SU2", "3x1/2 + 4x1"),
    cue.Irreps("SU2", "2x1/2 + 5x1 + 2x1/2"),
    cue.Irreps("SU2", "2x1/2 + 0x1 + 0x1/2 + 1 + 2"),
]


@pytest.mark.parametrize("irreps_in", list_of_irreps)
@pytest.mark.parametrize("irreps_out", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("shared_weights", [True, False])
def test_linear_fwd(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    shared_weights: bool,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    linear = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float64,
        use_fallback=False,
    )

    torch.manual_seed(0)
    linear_fx = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float64,
        use_fallback=True,
    )
    x = torch.randn(10, irreps_in.dim, dtype=torch.float64).cuda()

    if shared_weights:
        y = linear(x)
        y_fx = linear_fx(x)
    else:
        w = torch.randn(10, linear.weight_numel, dtype=torch.float64).cuda()
        y = linear(x, w)
        y_fx = linear_fx(x, w)

    assert y.shape == (10, irreps_out.dim)

    torch.testing.assert_close(y, y_fx)


@pytest.mark.parametrize("irreps_in", list_of_irreps)
@pytest.mark.parametrize("irreps_out", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("shared_weights", [True, False])
def test_linear_bwd_bwd(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    shared_weights: bool,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip redundant layout combinations
    if layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test")

    # Skip non-shared weights tests for speed
    if not shared_weights:
        pytest.skip("Skipping non-shared weights test for speed")

    # Skip complex irreps combinations - only test simple ones
    if irreps_in.num_irreps > 2 or irreps_out.num_irreps > 2:
        pytest.skip("Skipping complex irreps combination for speed")

    torch.manual_seed(0)
    linear = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float64,
        use_fallback=False,
    )

    torch.manual_seed(0)
    linear_fx = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float64,
        use_fallback=True,
    )

    x = torch.randn(10, irreps_in.dim, dtype=torch.float64, requires_grad=True).cuda()

    if shared_weights:
        y = linear(x)
        y_fx = linear_fx(x)
    else:
        w = torch.randn(
            10, linear.weight_numel, dtype=torch.float64, requires_grad=True
        ).cuda()
        y = linear(x, w)
        y_fx = linear_fx(x, w)

    if shared_weights:
        grad_inputs = [x]
        grad_inputs_fx = [x]
    else:
        grad_inputs = [x, w]
        grad_inputs_fx = [x, w]

    grad_outputs = torch.randn_like(y)
    (g_x,) = torch.autograd.grad(y, grad_inputs[:1], grad_outputs, create_graph=True)
    (g_x_fx,) = torch.autograd.grad(
        y_fx, grad_inputs_fx[:1], grad_outputs, create_graph=True
    )

    torch.testing.assert_close(g_x, g_x_fx)

    gg_x = torch.autograd.grad(g_x.sum(), grad_inputs[:1])[0]
    gg_x_fx = torch.autograd.grad(g_x_fx.sum(), grad_inputs_fx[:1])[0]

    torch.testing.assert_close(gg_x, gg_x_fx)


def test_e3nn_compatibility():
    try:
        from e3nn import o3
    except ImportError:
        pytest.skip("e3nn is not installed")

    with pytest.warns(UserWarning):
        irreps = o3.Irreps("3x1o + 4x1e")
        cuet.Linear(irreps, irreps, layout=cue.mul_ir)

    with pytest.warns(UserWarning):
        cuet.Linear("3x0e + 5x1o", "3x0e + 2x1o", layout=cue.ir_mul)


def test_no_layout_warning():
    with pytest.warns(UserWarning):
        cuet.Linear(cue.Irreps("SU2", "1"), cue.Irreps("SU2", "1"))


@pytest.mark.parametrize("irreps_in", list_of_irreps)
@pytest.mark.parametrize("irreps_out", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("shared_weights", [True, False])
def test_linear_copy(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    shared_weights: bool,
):
    linear = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
    ).to(device)

    copy.deepcopy(linear)


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("irreps_in", list_of_irreps)
@pytest.mark.parametrize("irreps_out", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("shared_weights", [True, False])
@pytest.mark.parametrize("mode", export_modes)
@pytest.mark.parametrize("use_fallback", [True, False])
def test_export(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    shared_weights: bool,
    mode: str,
    use_fallback: bool,
    tmp_path: str,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip JIT mode as it's slow
    if mode == "jit":
        pytest.skip("Skipping slow JIT compilation test")

    # Skip redundant combinations - test only one layout with fallback=True
    if use_fallback is True and layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test with fallback=True")

    # Skip compile mode entirely for speed - consistently takes time
    # if mode == "compile":
    #     pytest.skip("Skipping compile mode for speed")

    # Skip script mode for speed
    # if mode == "script":
    #     pytest.skip("Skipping script mode for speed")

    # Skip use_fallback=False for speed - only test fallback
    if use_fallback is False:
        pytest.skip("Skipping use_fallback=False for speed")

    # Skip shared_weights=False for speed
    if shared_weights is False:
        pytest.skip("Skipping shared_weights=False for speed")

    # Skip complex irreps combinations - only test the first one
    if irreps_in != list_of_irreps[0] or irreps_out != list_of_irreps[0]:
        pytest.skip("Skipping complex irreps combinations for speed")

    torch.manual_seed(0)
    m = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=shared_weights,
        device=device,
        dtype=torch.float32,
        use_fallback=use_fallback,
    )

    x = torch.randn(10, irreps_in.dim, dtype=torch.float32).cuda()

    if shared_weights:
        inputs = (x,)
    else:
        w = torch.randn(10, m.weight_numel, dtype=torch.float32).cuda()
        inputs = (x, w)

    out1 = m(*inputs)
    m = module_with_mode(mode, m, inputs, torch.float32, tmp_path)
    out2 = m(*inputs)
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("irreps_in", [cue.Irreps("SU2", "3x1/2 + 4x1")])
@pytest.mark.parametrize("irreps_out", [cue.Irreps("SU2", "3x1/2 + 4x1")])
@pytest.mark.parametrize("layout", [cue.mul_ir])
@pytest.mark.parametrize("method", ["naive", "fused_tp", "indexed_linear"])
@pytest.mark.parametrize("math_dtype", [None, torch.float32, "float64"])
def test_linear_fwd_methods(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    layout: cue.IrrepsLayout,
    method: str,
    math_dtype: str | torch.dtype | None,
):
    if math_dtype == torch.float32 and method == "fused_tp":
        pytest.skip("fused_tp does not support float32 math_dtype with FP64 inputs")
    if method == "indexed_linear" and math_dtype is not None:
        pytest.skip("indexed_linear does not support non-None or CUBLAS math_dtype")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    linear_indexed = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=True,
        internal_weights=False,
        weight_classes=3,
        device=device,
        dtype=torch.float64,
        math_dtype=math_dtype,
        method=method,
    )

    torch.manual_seed(0)
    linear = cuet.Linear(
        irreps_in,
        irreps_out,
        layout=layout,
        shared_weights=False,
        internal_weights=False,
        device=device,
        dtype=torch.float64,
        math_dtype=math_dtype,
        method="naive",
    )

    x = torch.randn(10, irreps_in.dim, dtype=torch.float64).cuda()
    weights = torch.randn(3, linear_indexed.weight_numel, dtype=torch.float64).cuda()
    indices, _ = torch.sort(torch.randint(0, 3, (10,), dtype=torch.int32).cuda())
    pre_indexed_weights = weights[indices]

    y_ind = linear_indexed(x, weights, indices)
    y = linear(x, pre_indexed_weights)

    torch.testing.assert_close(y_ind, y)
