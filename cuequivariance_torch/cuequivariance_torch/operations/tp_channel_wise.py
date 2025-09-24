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
from typing import Optional, Sequence

import torch
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
    default_layout,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


class ChannelWiseTensorProduct(torch.nn.Module):
    """
    Channel-wise tensor product layer.

    Args:
        irreps_in1 (Irreps): Input irreps for the first operand.
        irreps_in2 (Irreps): Input irreps for the second operand.
        filter_irreps_out (Sequence of Irrep, optional): Filter for the output irreps. Default is None.
        layout (IrrepsLayout, optional): The layout of the input and output irreps. Default is ``cue.mul_ir`` which is the layout corresponding to e3nn.
        layout_in1 (IrrepsLayout, optional): The layout of the first input irreducible representations, by default ``layout``.
        layout_in2 (IrrepsLayout, optional): The layout of the second input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        shared_weights (bool, optional): Whether to share weights across the batch dimension. Default is True.
        internal_weights (bool, optional): Whether to create module parameters for weights. Default is None.
        device (torch.device, optional): The device to use for the operation.
        dtype (torch.dtype, optional): The dtype to use for the operation weights, by default ``torch.float32``.
        math_dtype (torch.dtype or string, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors,
            if possible, or the torch default dtype (see SegmentedPolynomial for more details).
        method (str, optional): The method to use for the operation, by default "uniform_1d" (using a CUDA kernel)
            if all segments have the same shape, otherwise "naive" (using a PyTorch implementation).
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True` the "naive" method is used.
            If `False` the "uniform_1d" method is used (make sure all segments have the same shape).

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
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
    ):
        super().__init__()
        irreps_in1, irreps_in2 = default_irreps(irreps_in1, irreps_in2)
        assert_same_group(irreps_in1, irreps_in2)

        e = descriptors.channelwise_tensor_product(
            irreps_in1, irreps_in2, filter_irreps_out
        )
        descriptor, irreps_out = (
            e.polynomial.operations[0][1],
            e.operands[-1].irreps,
        )
        assert descriptor.subscripts == "uv,iu,jv,kuv+ijk"
        e2 = e.flatten_coefficient_modes().squeeze_modes()
        u1d_compatible = (
            e2.all_same_segment_shape()
            and len(e2.polynomial.operations[0][1].subscripts.modes()) == 1
        )

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
                if u1d_compatible:
                    # No warning here as it's the default behavior
                    self.method = "uniform_1d"
                else:
                    warnings.warn(
                        "Segments are not the same shape, falling back to `naive` method\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                    self.method = "naive"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                if not u1d_compatible and not use_fallback:
                    raise ValueError(
                        "`uniform_1d` method requires segments to be the same shape\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                self.method = "naive" if use_fallback else "uniform_1d"
        else:
            if method == "uniform_1d" and not u1d_compatible:
                raise ValueError(
                    "`uniform_1d` method requires segments to be the same shape\n"
                    "You can consider making the segments uniform in the descriptor."
                )
            self.method = method

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

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
        indices_1: Optional[torch.Tensor] = None,
        indices_2: Optional[torch.Tensor] = None,
        indices_out: Optional[torch.Tensor] = None,
        size_out: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the channel-wise tensor product operation.

        Args:
            x1 (torch.Tensor): Input tensor for the first operand. It should have the shape (:, irreps_in1.dim).
            x2 (torch.Tensor):  Input tensor for the second operand. It should have the shape (:, irreps_in2.dim).
            weight (torch.Tensor, optional): Weights for the tensor product. It should have the shape (batch_size, weight_numel)
                if shared_weights is False, or (1, weight_numel) if shared_weights is True.
                If None, the internal weights are used.
            indices_1 (torch.Tensor, optional): Indices to gather elements for the first operand.
            indices_2 (torch.Tensor, optional): Indices to gather elements for the second operand.
            indices_out (torch.Tensor, optional): Indices to scatter elements for the output.
            size_out (int, optional): Batch dimension of the output. Needed if indices_out are provided.

        Returns:
            torch.Tensor:
                Output tensor resulting from the channel-wise tensor product operation.
                It will have the shape (batch_size, irreps_out.dim).

        Raises:
            ValueError: If internal weights are used and weight is not None,
                or if shared weights are used and weight is not a 1D tensor,
                or if shared weights are not used and weight is not a 2D tensor.
                or if size_out is not provided and indices_out is provided.
        """
        x1 = self.transpose_in1(x1)
        x2 = self.transpose_in2(x2)

        indices_in = {}
        if indices_1 is not None:
            indices_in[1] = indices_1
        if indices_2 is not None:
            indices_in[2] = indices_2
        if indices_out is not None:
            indices_out = {0: indices_out}
            if size_out is None:
                raise ValueError(
                    "size_out should be provided if indices_out is provided"
                )
            else:
                sizes_out = {0: torch.empty(size_out, 1).to(x1.device)}
        else:
            sizes_out = {}

        if self.weight is not None:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            else:
                weight = self.weight
        else:
            if weight is None:
                raise ValueError(
                    "Internal weights are not used, weight should not be None"
                )

        output = self.f(
            [weight, x1, x2],
            input_indices=indices_in,
            output_shapes=sizes_out,
            output_indices=indices_out,
        )
        return self.transpose_out(output[0])
