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
from typing import Optional, Sequence

import torch

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
)


class ChannelWiseTensorProduct(torch.nn.Module):
    """
    Channel-wise tensor product layer.

    Args:
        irreps_in1 (Irreps): Input irreps for the first operand.
        irreps_in2 (Irreps): Input irreps for the second operand.
        filter_irreps_out (Sequence of Irrep, optional): Filter for the output irreps. Default is None.
        layout (IrrepsLayout, optional): The layout of the input and output irreps. Default is ``cue.mul_ir`` which is the layout corresponding to e3nn.
        shared_weights (bool, optional): Whether to share weights across the batch dimension. Default is True.
        internal_weights (bool, optional): Whether to create module parameters for weights. Default is None.
        use_fallback (bool, optional): If `None` (default), a CUDA kernel will be used if available.
                If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
                If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

    Note:
        In e3nn there was a irrep_normalization and path_normalization parameters.
        This module currently only supports "component" irrep normalization and "element" path normalization.
    """

    def __init__(
        self,
        irreps_in1: cue.Irreps,
        irreps_in2: cue.Irreps,
        filter_irreps_out: Sequence[cue.Irrep] = None,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in1: Optional[cue.IrrepsLayout] = None,
        layout_in2: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        shared_weights: bool = True,
        internal_weights: bool = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()
        irreps_in1, irreps_in2 = default_irreps(irreps_in1, irreps_in2)
        assert_same_group(irreps_in1, irreps_in2)

        math_dtype = math_dtype or dtype

        e = descriptors.channelwise_tensor_product(
            irreps_in1, irreps_in2, filter_irreps_out
        )
        descriptor, irreps_out = (
            e.polynomial.operations[0][1],
            e.operands[-1].irreps,
        )
        assert descriptor.subscripts == "uv,iu,jv,kuv+ijk"

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        self.weight_numel = descriptor.operands[0].size

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
            layout_in=(cue.ir_mul, layout_in1, layout_in2),
            layout_out=layout_out,
            device=device,
            math_dtype=math_dtype,
            use_fallback=use_fallback,
        )

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return (
            f"shared_weights={self.shared_weights}"
            f", internal_weights={self.internal_weights}"
            f", weight_numel={self.weight_numel}"
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the fully connected tensor product operation.

        Args:
            x1 (torch.Tensor): Input tensor for the first operand. It should have the shape (batch_size, irreps_in1.dim).
            x2 (torch.Tensor):  Input tensor for the second operand. It should have the shape (batch_size, irreps_in2.dim).
            weight (torch.Tensor, optional): Weights for the tensor product. It should have the shape (batch_size, weight_numel)
                if shared_weights is False, or (weight_numel,) if shared_weights is True.
                If None, the internal weights are used.

        Returns:
            torch.Tensor:
                Output tensor resulting from the fully connected tensor product operation.
                It will have the shape (batch_size, irreps_out.dim).

        Raises:
            ValueError: If internal weights are used and weight is not None,
                or if shared weights are used and weight is not a 1D tensor,
                or if shared weights are not used and weight is not a 2D tensor.
        """
        if self.weight is not None:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            else:
                return self.f(self.weight, x1, x2)
        else:
            if weight is None:
                raise ValueError(
                    "Internal weights are not used, weight should not be None"
                )
            else:
                return self.f(weight, x1, x2)
