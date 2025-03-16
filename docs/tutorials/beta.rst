.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Beta features with unstable API
===============================

The API for the following features are likely to change in the future.

Enabling the JIT kernel
-----------------------

The segmented tensor product with 3 or 4 operands with one mode can be executed using an experimental JIT kernel. Here is how to enable it:

.. jupyter-execute::

    import os
    import torch

    import cuequivariance as cue
    import cuequivariance_torch as cuet

    os.environ["CUEQUIVARIANCE_OPS_USE_JIT"] = "1"

    e = (
        cue.descriptors.channelwise_tensor_product(
            128 * cue.Irreps("SO3", "0 + 1 + 2"),
            cue.Irreps("SO3", "0 + 1 + 2 + 3"),
            cue.Irreps("SO3", "0 + 1 + 2"),
        )
        .squeeze_modes()
        .flatten_coefficient_modes()
    )
    print(e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul, device=device)
    x0 = torch.randn(128, e.inputs[0].dim, device=device)
    x1 = torch.randn(128, e.inputs[1].dim, device=device)
    x2 = torch.randn(128, e.inputs[2].dim, device=device)
    print(m(x0, x1, x2).shape)

Fused scatter/gather kernel
---------------------------

Again for segmented tensor product with 3 or 4 operands with one mode, we can use a fused scatter/gather kernel. This kernel is not JIT compiled.

.. jupyter-execute::

    from cuequivariance_torch.primitives.tensor_product import (
        TensorProductUniform4x1dIndexed,
    )

    if device.type == "cuda":
        ((_, d),) = e.polynomial.operations
        m = TensorProductUniform4x1dIndexed(d, device, torch.float32)

        x0 = torch.randn(16, e.inputs[0].dim, device=device)
        i0 = torch.randint(0, 16, (128,), device=device)
        x1 = torch.randn(128, e.inputs[1].dim, device=device)
        x2 = torch.randn(128, e.inputs[2].dim, device=device)
        i_out = torch.randint(0, 16, (128,), device=device)
        print(m(x0, x1, x2, i0, None, None, i_out, 16).shape)