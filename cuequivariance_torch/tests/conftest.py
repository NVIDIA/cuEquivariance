# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch._dynamo

# torch 2.11+ no longer honors allow_in_graph for torch.autograd.grad; it gates
# tracing through it under torch.compile/export behind this config flag. Several
# tests (e.g. segmented_polynomial in compile/export modes) call torch.autograd.grad
# inside a compiled module, so enable it session-wide. Guarded for older torch.
if hasattr(torch._dynamo.config, "trace_autograd_ops"):
    torch._dynamo.config.trace_autograd_ops = True


@pytest.fixture(autouse=True)
def check_torch_memory():
    yield
    try:
        if torch.cuda.is_available():
            usage_gib = torch.cuda.max_memory_allocated() / (1024**3)
            limit = 2.0
            assert usage_gib <= limit, (
                f"PyTorch peak memory usage {usage_gib:.2f}GiB exceeds {limit}GiB limit!"
            )
    except Exception:
        pass  # No CUDA available
