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
from cuequivariance_torch.primitives.symmetric_tensor_product import (
    CUDAKernel as SymmetricTensorProduct,
)
from cuequivariance_torch.primitives.tensor_product import (
    FusedTensorProductOp3,
    FusedTensorProductOp4,
    TensorProductUniform3x1d,
    TensorProductUniform4x1d,
)

import cuequivariance as cue

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

export_modes = ["script", "compile", "jit"]


@pytest.mark.parametrize("mode", export_modes)
def test_script_symmetric_contraction(mode, tmp_path):
    pytest.skip("This is deprecated and will be removed soon.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    e = cue.descriptors.symmetric_contraction(
        32 * cue.Irreps("SO3", "0 + 1"), 32 * cue.Irreps("SO3", "0 + 1"), [1, 2, 3]
    )
    ds = [stp for _, stp in e.polynomial.operations]

    batch = 12
    x0 = torch.randn(3, ds[0].operands[0].size, device=device, dtype=torch.float32)
    i0 = torch.zeros(batch, device=device, dtype=torch.int32)
    x1 = torch.randn(batch, ds[0].operands[1].size, device=device, dtype=torch.float32)

    m = SymmetricTensorProduct(ds, device, torch.float32)
    inputs = (x0, i0, x1)
    module = module_with_mode(mode, m, inputs, torch.float32, tmp_path)
    out1 = m(*inputs)
    out2 = module(*inputs)
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("mode", export_modes + ["trt"])
def test_script_fused_tp_3(mode, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    d = (
        cue.descriptors.full_tensor_product(
            cue.Irreps("SO3", "32x1"), cue.Irreps("SO3", "1")
        )
        .polynomial.operations[0][1]
        .flatten_coefficient_modes()
        .squeeze_modes("v")
    )

    exp_inputs = [
        torch.randn(1, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]
    batch = 12
    inputs = [
        torch.randn(batch, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]
    module = FusedTensorProductOp3(d, (0, 1), device, torch.float32)
    out1 = module(*inputs)
    out11 = module(*exp_inputs)
    module = module_with_mode(mode, module, exp_inputs, torch.float32, tmp_path)
    out2 = module(*inputs)
    out22 = module(*exp_inputs)
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out11, out22)


@pytest.mark.parametrize("mode", export_modes + ["trt"])
def test_script_fused_tp_4(mode, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    d = (
        cue.descriptors.fully_connected_tensor_product(
            cue.Irreps("SO3", "32x1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "32x1")
        )
        .polynomial.operations[0][1]
        .flatten_coefficient_modes()
        .squeeze_modes("v")
        .permute_operands([1, 2, 0, 3])
    )

    exp_inputs = [
        torch.randn(1, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]
    batch = 12
    inputs = [
        torch.randn(batch, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]

    module = FusedTensorProductOp4(d, [0, 1, 2], device, torch.float32)
    out1 = module(*inputs)
    out11 = module(*exp_inputs)
    module = module_with_mode(mode, module, exp_inputs, torch.float32, tmp_path)

    out2 = module(*inputs)
    out22 = module(*exp_inputs)

    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out11, out22)


@pytest.mark.parametrize("mode", export_modes)
def test_script_uniform_tp_3(mode, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    d = (
        cue.descriptors.full_tensor_product(
            cue.Irreps("SO3", "32x1"), cue.Irreps("SO3", "1")
        )
        .polynomial.operations[0][1]
        .flatten_coefficient_modes()
        .squeeze_modes("v")
    )

    exp_inputs = [
        torch.randn(1, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]
    batch = 12
    inputs = [
        torch.randn(batch, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]

    module = TensorProductUniform3x1d(d, device, torch.float32)
    out1 = module(*inputs)
    out11 = module(*exp_inputs)
    module = module_with_mode(mode, module, exp_inputs, torch.float32, tmp_path)
    out2 = module(*inputs)
    out22 = module(*exp_inputs)
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out11, out22)


@pytest.mark.parametrize("mode", export_modes)
def test_script_uniform_tp_4(mode, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    d = (
        cue.descriptors.channelwise_tensor_product(
            cue.Irreps("SO3", "32x1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "32x1")
        )
        .polynomial.operations[0][1]
        .flatten_coefficient_modes()
        .squeeze_modes("v")
    )

    exp_inputs = [
        torch.randn(1, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]
    batch = 12
    inputs = [
        torch.randn(batch, ope.size, device=device, dtype=torch.float32)
        for ope in d.operands[:-1]
    ]
    module = TensorProductUniform4x1d(d, device, torch.float32)
    out1 = module(*inputs)
    out11 = module(*exp_inputs)
    module = module_with_mode(mode, module, exp_inputs, torch.float32, tmp_path)
    out2 = module(*inputs)
    out22 = module(*exp_inputs)
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out11, out22)
