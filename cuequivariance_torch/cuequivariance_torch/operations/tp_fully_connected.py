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
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
    default_layout,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


class FullyConnectedTensorProduct(torch.nn.Module):
    """
    Fully connected tensor product layer.

    Args:
        irreps_in1 (Irreps): Input irreps for the first operand.
        irreps_in2 (Irreps): Input irreps for the second operand.
        irreps_out (Irreps): Output irreps.
        layout (IrrepsLayout, optional): The layout of the input and output irreps. Default is ``cue.mul_ir`` which is the layout corresponding to e3nn.
        layout_in1 (IrrepsLayout, optional): The layout of the first input irreducible representations, by default ``layout``.
        layout_in2 (IrrepsLayout, optional): The layout of the second input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        shared_weights (bool, optional): Whether to share weights across the batch dimension. Default is True.
        internal_weights (bool, optional): Whether to create module parameters for weights. Default is None.
        device (torch.device, optional): The device to use for the operation.
        dtype (torch.dtype, optional): The dtype to use for the operation weights, by default the torch default dtype.
        math_dtype (torch.dtype or string, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors,
            if possible, or the torch default dtype (see SegmentedPolynomial for more details).
        method (str, optional): The method to use for the linear layer, by default "fused_tp" (using a CUDA kernel).
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True`, the "naive" method is used.
            If `False` or `None` (default), the "fused_tp" method is used.

    Note:
        In e3nn there was a irrep_normalization and path_normalization parameters.
        This module currently only supports "component" irrep normalization and "element" path normalization.
    """

    def __init__(
        self,
        irreps_in1: cue.Irreps,
        irreps_in2: cue.Irreps,
        irreps_out: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in1: Optional[cue.IrrepsLayout] = None,
        layout_in2: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        shared_weights: bool = True,
        internal_weights: bool = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
    ):
        super().__init__()
        irreps_in1, irreps_in2, irreps_out = default_irreps(
            irreps_in1, irreps_in2, irreps_out
        )
        assert_same_group(irreps_in1, irreps_in2, irreps_out)

        if dtype is None:
            dtype = torch.get_default_dtype()

        e = descriptors.fully_connected_tensor_product(
            irreps_in1, irreps_in2, irreps_out
        )
        assert e.polynomial.operations[0][1].subscripts == "uvw,iu,jv,kw+ijk"

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        self.weight_numel = e.polynomial.operations[0][1].operands[0].size

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

        layout_in1 = default_layout(layout_in1 or layout)
        self.transpose_in1 = cuet.TransposeIrrepsLayout(
            e.inputs[1].irreps,
            source=layout_in1,
            target=e.inputs[1].layout,
            device=device,
            use_fallback=use_fallback,
        )

        layout_in2 = default_layout(layout_in2 or layout)
        self.transpose_in2 = cuet.TransposeIrrepsLayout(
            e.inputs[2].irreps,
            source=layout_in2,
            target=e.inputs[2].layout,
            device=device,
            use_fallback=use_fallback,
        )

        layout_out = default_layout(layout_out or layout)
        self.transpose_out = cuet.TransposeIrrepsLayout(
            e.outputs[0].irreps,
            source=e.outputs[0].layout,
            target=layout_out,
            device=device,
            use_fallback=use_fallback,
        )

        if method is None:
            if use_fallback is None:
                # No warning here as it's the default behavior
                self.method = "fused_tp"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                self.method = "naive" if use_fallback else "fused_tp"
        else:
            self.method = method

        if self.method == "fused_tp" and math_dtype is None:
            math_dtype = dtype

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

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
            x2 (torch.Tensor): Input tensor for the second operand. It should have the shape (batch_size, irreps_in2.dim).
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
        x1 = self.transpose_in1(x1)
        x2 = self.transpose_in2(x2)

        if self.weight is not None:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            output = self.f([self.weight, x1, x2])
        else:
            if weight is None:
                raise ValueError(
                    "Internal weights are not used, weight should not be None"
                )
            else:
                output = self.f([weight, x1, x2])

        return self.transpose_out(output[0])
