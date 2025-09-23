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

import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

dtypes = [torch.float32, torch.float64]
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    dtypes += [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("dtype", dtypes)
def test_transpose(use_fallback: bool, dtype: torch.dtype):
    """
    1 2 3       1 4
    4 5 6       2 5
                3 6
        ------>
    10 11       10 12
    12 13       11 13
    """
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10, 11, 12, 13]], dtype=dtype, device=device
    )
    segments = [(2, 3), (2, 2)]
    xt = torch.tensor(
        [[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 10, 12, 11, 13]], dtype=dtype, device=device
    )

    m = cuet.TransposeSegments(segments, device, use_fallback=use_fallback)
    torch.testing.assert_close(m(x), xt)


@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("dtype", dtypes)
def test_transpose_empty_tensor(use_fallback: bool, dtype: torch.dtype):
    if use_fallback is False and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    x = torch.zeros((0, 10), dtype=dtype, device=device)
    segments = [(2, 3), (2, 2)]
    xt = torch.zeros((0, 10), dtype=dtype, device=device)

    m = cuet.TransposeSegments(segments, device, use_fallback=use_fallback)
    torch.testing.assert_close(m(x), xt)


export_modes = ["compile", "script", "jit"]


@pytest.mark.parametrize("mode", export_modes)
@pytest.mark.parametrize("use_fallback", [True, False])
def test_export(mode, use_fallback, tmp_path):
    if not use_fallback and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dtype = torch.float32
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10, 11, 12, 13]], dtype=dtype, device=device
    )
    segments = [(2, 3), (2, 2)]
    m = cuet.TransposeSegments(segments, device, use_fallback=use_fallback)
    out1 = m(x)
    m = module_with_mode(mode, m, (x,), dtype, tmp_path)
    out2 = m(x)
    torch.testing.assert_close(out1, out2)
