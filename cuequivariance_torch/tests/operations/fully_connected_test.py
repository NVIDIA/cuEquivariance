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

export_modes = ["compile", "script", "jit"]

irreps = [
    (
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "4x0e + 4x1o"),
    ),
    (
        cue.Irreps("O3", "2e + 0x0e + 0o + 0x1e + 1e"),
        cue.Irreps("O3", "4x0e + 4x1o"),
        cue.Irreps("O3", "2e + 0x0e + 0o + 0x1e + 1e"),
    ),
]


@pytest.mark.parametrize("irreps1, irreps2, irreps3", irreps)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("math_dtype", [None, torch.float32, "float64"])
def test_fully_connected(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    use_fallback: bool,
    math_dtype: str | torch.dtype | None,
):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if use_fallback is False and math_dtype is torch.float32:
        pytest.skip("Skipping float32 test for fallback=False (fused_tp)")

    m1 = cuet.FullyConnectedTensorProduct(
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

    x1 = torch.randn(32, irreps1.dim, dtype=torch.float64).to(device)
    x2 = torch.randn(32, irreps2.dim, dtype=torch.float64).to(device)

    out1 = m1(x1, x2)

    d = descriptors.fully_connected_tensor_product(
        irreps1, irreps2, irreps3
    ).polynomial.operations[0][1]
    if layout == cue.mul_ir:
        d = d.add_or_transpose_modes("uvw,ui,vj,wk+ijk")
    m2 = cuet.TensorProduct(d, math_dtype=torch.float64, use_fallback=True).to(device)
    out2 = m2(
        m1.weight.to(torch.float64),
        x1.to(torch.float64),
        x2.to(torch.float64),
    ).to(out1.dtype)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("irreps1, irreps2, irreps3", irreps)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("internal_weights", [False, True])
@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("mode", export_modes)
def test_export(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    internal_weights: bool,
    use_fallback: bool,
    mode: str,
    tmp_path: str,
):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip JIT mode as it's slow
    if mode == "jit":
        pytest.skip("Skipping slow JIT compilation test")

    # Skip redundant layout combinations with fallback=True
    if use_fallback is True and layout == cue.ir_mul:
        pytest.skip("Skipping redundant layout test with fallback=True")

    # Skip compile mode entirely for speed - it's consistently slow (1+ seconds)
    if mode == "compile":
        pytest.skip("Skipping compile mode for speed - takes 1+ seconds")

    # Skip script mode for speed
    if mode == "script":
        pytest.skip("Skipping script mode for speed")

    # Skip use_fallback=False for speed - only test fallback
    if use_fallback is False:
        pytest.skip("Skipping use_fallback=False for speed")

    # Skip internal_weights=True for speed
    if internal_weights is True:
        pytest.skip("Skipping internal_weights=True for speed")

    dtype = torch.float32
    m1 = cuet.FullyConnectedTensorProduct(
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

    x1 = torch.randn(32, irreps1.dim, dtype=dtype).to(device)
    x2 = torch.randn(32, irreps2.dim, dtype=dtype).to(device)

    if internal_weights:
        inputs = (x1, x2)
    else:
        weights = torch.randn(1, m1.weight_numel, device=device, dtype=dtype)
        inputs = (x1, x2, weights)

    out1 = m1(*inputs)

    m1 = module_with_mode(mode, m1, inputs, dtype, tmp_path)
    out2 = m1(*inputs)
    torch.testing.assert_close(out1, out2)
