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
from typing import Optional

import torch

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
)


class Linear(torch.nn.Module):
    """
    A class that represents an equivariant linear layer.

    Args:
        irreps_in (Irreps): The input irreducible representations.
        irreps_out (Irreps): The output irreducible representations.
        layout (IrrepsLayout, optional): The layout of the irreducible representations, by default ``cue.mul_ir``. This is the layout used in the e3nn library.
        shared_weights (bool, optional): Whether to use shared weights, by default True.
        internal_weights (bool, optional): Whether to use internal weights, by default True if shared_weights is True, otherwise False.
        use_fallback (bool, optional): If `None` (default), a CUDA kernel will be used if available.
                If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
                If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        shared_weights: bool = True,
        internal_weights: bool = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()
        irreps_in, irreps_out = default_irreps(irreps_in, irreps_out)
        assert_same_group(irreps_in, irreps_out)

        math_dtype = math_dtype or dtype

        e = descriptors.linear(irreps_in, irreps_out)
        assert e.polynomial.operations[0][1].subscripts == "uv,iu,iv"

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.weight_numel = e.inputs[0].dim

        self.shared_weights = shared_weights
        self.internal_weights = (
            internal_weights if internal_weights is not None else shared_weights
        )

        if self.internal_weights:
            if not self.shared_weights:
                raise ValueError("Internal weights should be shared")
            self.weight = torch.nn.Parameter(
                torch.randn(1, self.weight_numel, device=device, dtype=dtype)
            )
        else:
            self.weight = None

        self.f = cuet.EquivariantTensorProduct(
            e,
            layout=layout,
            layout_in=layout_in,
            layout_out=layout_out,
            device=device,
            math_dtype=math_dtype,
            use_fallback=use_fallback,
        )

    def extra_repr(self) -> str:
        return f"shared_weights={self.shared_weights}, internal_weights={self.internal_weights}, weight_numel={self.weight_numel}"

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the linear layer.

        Args:
            x (torch.Tensor): The input tensor.
            weight (torch.Tensor, optional): The weight tensor. If None, the internal weight tensor is used.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.

        Raises:
            ValueError: If internal weights are used and weight is not None,
                or if shared weights are used and weight is not a 1D tensor,
                or if shared weights are not used and weight is not a 2D tensor.
        """
        if self.internal_weights:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")

            weight = self.weight

        if weight is None:
            raise ValueError("Weights should not be None")

        return self.f(weight, x)
