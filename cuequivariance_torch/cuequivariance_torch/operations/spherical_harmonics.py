# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import torch

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


def spherical_harmonics(
    ls: list[int],
    vectors: torch.Tensor,
    normalize: bool = True,
    optimize_fallback: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the spherical harmonics of the input vectors.

    Args:
        ls (list of int): List of spherical harmonic degrees.
        vectors (torch.Tensor): Input vectors of shape (..., 3).
        normalize (bool, optional): Whether to normalize the input vectors. Defaults to True.
        optimize_fallback (bool, optional): Whether to optimize fallback. Defaults to None.

    Returns:
        torch.Tensor: The spherical harmonics of the input vectors of shape (..., dim)
        where dim is the sum of 2*l+1 for l in ls.
    """
    if isinstance(ls, int):
        ls = [ls]
    assert ls == sorted(set(ls))
    assert vectors.shape[-1] == 3

    if normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)

    x = vectors.reshape(-1, 3)
    m = cuet.EquivariantTensorProduct(
        descriptors.spherical_harmonics(cue.SO3(1), ls),
        layout=cue.ir_mul,
        device=x.device,
        math_dtype=x.dtype,
        optimize_fallback=optimize_fallback,
    )
    y = m([x])
    y = y.reshape(vectors.shape[:-1] + (y.shape[-1],))
    return y
