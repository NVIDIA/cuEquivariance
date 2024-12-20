.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _tuto_kernels:

Available CUDA Kernels
======================

Here we list the available CUDA kernels in cuEquivariance and their use cases.

.. jupyter-execute::
    :hide-code:

    import cuequivariance as cue
    import cuequivariance_torch as cuet


Fused TP
--------

This kernel is useful for tensor products which have small operand sizes.

.. jupyter-execute::
    :hide-output:

    ns, nv = 48, 10
    irreps_feat = cue.Irreps("O3", f"{ns}x0e+{nv}x1o+{nv}x1e+{ns}x0o")
    irreps_sh = cue.Irreps("O3", "0e + 1o + 2e")
    cuet.FullyConnectedTensorProduct(irreps_feat, irreps_sh, irreps_feat, layout=cue.ir_mul)

Low level interface (non public API): ``cuequivariance_ops_torch.FusedTensorProductOp3`` and ``cuequivariance_ops_torch.FusedTensorProductOp4``

Uniform 1d
----------

This kernel works for STP with subscripts
 - ``^(|u),(|u),(|u),(|u)$``
 - ``^(|u),(|u),(|u)$``

A typical use case is the channel wise tensor product used in NequIP.

.. jupyter-execute::
    :hide-output:

    irreps_feat = 128 * cue.Irreps("O3", "0e + 1o + 2e")
    irreps_sh = cue.Irreps("O3", "0e + 1o + 2e + 3o")
    cuet.ChannelWiseTensorProduct(irreps_feat, irreps_sh, irreps_feat, layout=cue.ir_mul)

Low level interface (non public API): ``cuequivariance_ops_torch.TensorProductUniform1d``

Symmetric Contractions
----------------------

This kernel is designed for the symmetric contraction of MACE.
It supports subscripts ``u,u,u``, ``u,u,u,u``, etc up to 8 operands.
The first operand is the weights which are optionally indexed by integers.
The last operand is the output.
The other operands are the repeated input.

.. jupyter-execute::
    :hide-output:

    irreps_feat = 128 * cue.Irreps("O3", "0e + 1o + 2e")
    cuet.SymmetricContraction(irreps_feat, irreps_feat, 3, 1, layout=cue.ir_mul)

Low level interface (non public API): ``cuequivariance_ops_torch.SymmetricTensorContraction``
