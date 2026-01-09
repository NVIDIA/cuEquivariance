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
from cuequivariance_torch._tests.utils import module_with_mode

import cuequivariance as cue
import cuequivariance_torch as cuet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

export_modes = ["compile"]  # script/jit mode has TorchScript issues with dict indexing


@pytest.mark.parametrize("mode", export_modes)
@pytest.mark.parametrize("internal_weights", [True, False])
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_export_channelwise_tensor_product(mode, internal_weights, layout, tmp_path):
    """Test export of ChannelWiseTensorProduct in various modes."""
    irreps_in1 = cue.Irreps("SO3", "8x0 + 8x1")
    irreps_in2 = cue.Irreps("SO3", "1")

    # Create ChannelWiseTensorProduct module
    module = cuet.ChannelWiseTensorProduct(
        irreps_in1,
        irreps_in2,
        layout=layout,
        internal_weights=internal_weights,
        device=device,
        dtype=torch.float32,
    )

    # Create test inputs
    batch = 12
    x1 = torch.randn(batch, irreps_in1.dim, device=device, dtype=torch.float32)
    x2 = torch.randn(batch, irreps_in2.dim, device=device, dtype=torch.float32)

    if internal_weights:
        inputs = (x1, x2)
    else:
        weight = torch.randn(1, module.weight_numel, device=device, dtype=torch.float32)
        inputs = (x1, x2, weight)

    # Get reference output
    out1 = module(*inputs)

    # Export module
    exported_module = module_with_mode(mode, module, inputs, torch.float32, tmp_path)

    # Test exported module
    out2 = exported_module(*inputs)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)
