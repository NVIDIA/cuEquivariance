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
import warnings
from typing import Optional

import torch
import torch.nn as nn

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


class SphericalHarmonics(nn.Module):
    r"""Compute the spherical harmonics of the input vectors as a torch module."""

    def __init__(
        self,
        ls: list[int],
        normalize: bool = True,
        device: Optional[torch.device] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
    ):
        """
        Args:
            ls (list of int): List of spherical harmonic degrees.
            normalize (bool, optional): Whether to normalize the input vectors. Defaults to True.
            device (torch.device, optional): The device to use for the operation.
            math_dtype (torch.dtype or string, optional): The dtype to use for the operation, by default it follows the dtype of the input tensors,
                if possible, or the torch default dtype (see SegmentedPolynomial for more details).
            method (str, optional): The method to use for the operation, by default "uniform_1d" (using a CUDA kernel).
            use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
                If `True` the "naive" method is used.
                If `False` or None (default) the "uniform_1d" method is used.
        """
        super().__init__()
        self.ls = ls if isinstance(ls, list) else [ls]
        assert self.ls == sorted(set(self.ls))
        self.normalize = normalize

        e = descriptors.spherical_harmonics(cue.SO3(1), self.ls)

        if method is None:
            if use_fallback is None:
                # No warning here as it's the default behavior
                self.method = "uniform_1d"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                self.method = "naive" if use_fallback else "uniform_1d"
        else:
            self.method = method

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vectors (torch.Tensor): Input vectors of shape (batch, 3).

        Returns:
            torch.Tensor: The spherical harmonics of the input vectors of shape (batch, dim),
            where dim is the sum of 2*l+1 for l in ls.
        """
        torch._assert(
            vectors.ndim == 2, f"Input must have shape (batch, 3) - got {vectors.shape}"
        )

        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=1)

        [output] = self.f([vectors])
        return output
