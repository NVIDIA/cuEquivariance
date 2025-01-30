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

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance_torch._tests.utils import (
    module_with_mode,
)

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

    outputs = dict()
    for use_fallback in [True, False]:
        torch.manual_seed(0)
        linear = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=layout,
            shared_weights=shared_weights,
            device=device,
            dtype=torch.float64,
            use_fallback=use_fallback,
        )

        # reset the seed to ensure the same initialization
        torch.manual_seed(0)

        x = torch.randn(
            10, irreps_in.dim, requires_grad=True, device=device, dtype=torch.float64
        )

        if shared_weights:
            y = linear(x)
        else:
            w = torch.randn(
                10, linear.weight_numel, requires_grad=True, dtype=torch.float64
            ).cuda()
            y = linear(x, w)

        (grad,) = torch.autograd.grad(
            y.pow(2).sum(),
            x,
            create_graph=True,
        )

        grad.pow(2).sum().backward()

        outputs[use_fallback] = x.grad.clone()

    torch.testing.assert_close(outputs[True], outputs[False])


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
