.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Attention pair bias
===================

``cuequivariance_jax.attention_pair_bias`` is the JAX frontend for APB. The
frontend owns the public API, argument validation, and portable pure-JAX
implementation. When ``cuequivariance_ops_jax`` is installed, the same call is
delegated to its backend, which owns CUDA FFI eligibility and differentiation.
Keeping those responsibilities separate means installing the CUDA package is
an optimization, rather than a requirement for API availability.

Raw and cached pair representations
-----------------------------------

The API matches the APB interface used by the other cuEquivariance frontends.
It accepts raw pair features with shape ``(B, N, N, D_z)``. With
``return_z_proj=True``, the second return value has shape ``(B, H, N, N)`` and
can be passed back as ``pair_repr`` with ``is_cached_z_proj=True``. The raw and
cached forms produce the same attention result up to the selected numeric
precision. ``single_repr`` may have batch ``B * M``; pair features and a
``(B, N)`` mask are repeated over that multiplicity.

Portability and numeric behavior
--------------------------------

The pure-JAX implementation handles CPU and GPU execution, JIT, VJP, and
nested ``vmap``. It is also the fallback when the CUDA backend is unavailable
or an attention cell is not eligible for FFI. Eligible CUDA cells use FP16 or
BF16, a head dimension divisible by eight and no larger than 128, and a
sequence length divisible by eight. FP32 and unsupported shapes use pure JAX.

The public APB contract requires at least one valid key in every mask row. The
optimized backend relies on that precondition and does not add a
data-dependent reduction or conditional for fully masked rows. Fully masked
rows are unsupported in both the frontend and backend.

For FP16 and BF16 inputs, layer-normalization statistics are accumulated in
FP32 and outputs retain the input dtype. ``precision`` controls JAX contraction
precision. Masking is a finite additive penalty (``inf=1e6`` by default), not a
Boolean-only transformation, so floating masks retain gradients.

Validated hardware
------------------

The current backend validation matrix is:

==================  ==================  ==========================  ================================
GPU                 Compute capability  Eligible BF16 FFI backend  Status
==================  ==================  ==========================  ================================
RTX A6000           8.6                 generic                     Forward/backward parity verified
A100 80 GB PCIe     8.0                 generic                     Forward/backward parity verified
H100 80 GB HBM3     9.0                 generic                     Forward/backward parity verified
B200                10.0                SM100f                      Forward/backward parity verified
B300                10.3                SM100f                      Forward/backward parity verified
==================  ==================  ==========================  ================================

The matrix is target validation, not a claim that launch tuning transfers
unchanged between architectures.
