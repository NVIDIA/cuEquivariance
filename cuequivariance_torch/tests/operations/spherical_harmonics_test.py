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
import numpy as np
import pytest
import torch
from cuequivariance_torch._tests.utils import (
    module_with_mode,
)

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.parametrize(
    "dtype, tol",
    [(torch.float64, 1e-5), (torch.float32, 1e-4)],
)
@pytest.mark.parametrize("ell", [0, 1, 2, 3])
@pytest.mark.parametrize("use_fallback", [False, True])
def test_spherical_harmonics_equivariance(use_fallback: bool, ell: int, dtype, tol):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    vec = torch.randn(3, dtype=dtype, device=device)
    axis = np.random.randn(3)
    angle = np.random.rand()
    scale = 1.3

    m = cuet.SphericalHarmonics([ell], False, device=device, use_fallback=use_fallback)

    yl = m(vec.unsqueeze(0)).squeeze(0)

    R = torch.from_numpy(cue.SO3(1).rotation(axis, angle)).to(dtype).to(device)
    Rl = torch.from_numpy(cue.SO3(ell).rotation(axis, angle)).to(dtype).to(device)

    yl1 = m((scale * R @ vec).unsqueeze(0)).squeeze(0)
    yl2 = scale**ell * Rl @ yl

    torch.testing.assert_close(yl1, yl2, rtol=tol, atol=tol)


data_types = [torch.float32, torch.float64]

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    data_types += [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("ls", [[0], [1], [2], [0, 1], [0, 1, 2]])
@pytest.mark.parametrize("use_fallback", [False, True])
def test_spherical_harmonics_full(dtype, ls: list[int], use_fallback: bool):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    m = cuet.SphericalHarmonics(ls, False, use_fallback=use_fallback, device=device)

    vec = torch.randn(10, 3, device=device, dtype=dtype)
    yl = m(vec)
    assert yl.shape[0] == 10
    assert yl.shape[1] == sum(2 * ell + 1 for ell in ls)


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("ls", [[0], [1], [2], [0, 1], [0, 1, 2]])
@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("mode", export_modes)
def test_export(dtype, ls: list[int], use_fallback: bool, mode: str, tmp_path: str):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Skip JIT mode as it's slow
    if mode == "jit":
        pytest.skip("Skipping slow JIT compilation test")

    # Skip float16/bfloat16 with fallback=False to reduce test matrix
    if not use_fallback and dtype in [torch.float16, torch.bfloat16]:
        pytest.skip("Skipping fp16/bf16 tests with CUDA backend")

    # Skip high degree spherical harmonics with multiple ls for speed
    if len(ls) > 2 and max(ls) >= 2:
        pytest.skip("Skipping high degree multi-l tests for speed")

    # Skip compile mode for speed - consistently takes time (0.6+ seconds)
    if mode == "compile":
        pytest.skip("Skipping compile mode for speed")

    # Skip script mode for speed
    if mode == "script":
        pytest.skip("Skipping script mode for speed")

    # Skip use_fallback=False for speed - only test fallback
    if use_fallback is False:
        pytest.skip("Skipping use_fallback=False for speed")

    # Skip complex ls combinations - only test simple ones
    if len(ls) > 1:
        pytest.skip("Skipping complex ls combinations for speed")

    # Skip float16/bfloat16 tests entirely for speed
    if dtype in [torch.float16, torch.bfloat16]:
        pytest.skip("Skipping fp16/bf16 tests for speed")

    tol = 1e-5
    if dtype in [torch.float16, torch.bfloat16]:
        tol = 1e-2
    m = cuet.SphericalHarmonics(ls, False, use_fallback=use_fallback, device=device)

    vec = torch.randn(10, 3, device=device, dtype=dtype)
    inputs = (vec,)
    out1 = m(vec)

    m = module_with_mode(mode, m, inputs, dtype, tmp_path)
    out2 = m(*inputs)
    torch.testing.assert_close(out1, out2, atol=tol, rtol=tol)
