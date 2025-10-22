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

import enum

import jax


# Precision modes matching cuequivariance_ops
class Precision(enum.IntEnum):
    """Precision modes for matrix multiplication operations."""

    DEFAULT = 0
    TF32 = 1
    TF32x3 = 2
    IEEE = 3

    def _to_jax(self) -> jax.lax.Precision:
        """Convert Precision enum to JAX precision."""
        if self == Precision.DEFAULT:
            return jax.lax.Precision.DEFAULT
        elif self == Precision.TF32:
            return jax.lax.Precision.HIGH
        elif self == Precision.TF32x3:
            return jax.lax.Precision.HIGHEST
        elif self == Precision.IEEE:
            return jax.lax.Precision.HIGHEST
