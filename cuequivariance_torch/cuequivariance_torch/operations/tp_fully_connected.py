# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import *

import torch

import cuequivariance as cue
import cuequivariance.segmented_tensor_product as stp
import cuequivariance.equivariant_tensor_product as etp
import cuequivariance_torch as cuet

from cuequivariance.irreps_array.misc_ui import (
    default_irreps,
    default_layout,
    assert_same_group,
)


class FullyConnectedTensorProduct(torch.nn.Module):
    """
    Fully connected tensor product layer.

    Parameters
    ----------
    irreps_in1 : cue.Irreps
        Input irreps for the first operand.
    irreps_in2 : cue.Irreps
        Input irreps for the second operand.
    irreps_out : cue.Irreps
        Output irreps.
    layout : cue.IrrepsLayout, optional
        The layout of the input and output irreps. Default is `cue.mul_ir` which is the layout corresponding to e3nn.
    shared_weights : bool, optional
        Whether to share weights across the batch dimension. Default is True.
    internal_weights : bool, optional
        Whether to create module parameters for weights. Default is None.

    Notes
    -----
    In e3nn there was a irrep_normalization and path_normalization parameters.
    This module currently only supports "component" irrep normalization and "element" path normalization.
    """

    def __init__(
        self,
        irreps_in1: cue.Irreps,
        irreps_in2: cue.Irreps,
        irreps_out: cue.Irreps,
        *,
        layout: cue.IrrepsLayout = None,
        shared_weights: bool = True,
        internal_weights: bool = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[torch.dtype] = None,
        optimize_fallback: Optional[bool] = None,
    ):
        super().__init__()
        self.layout = layout = default_layout(layout)
        irreps_in1, irreps_in2, irreps_out = default_irreps(
            irreps_in1, irreps_in2, irreps_out
        )
        assert_same_group(irreps_in1, irreps_in2, irreps_out)

        descriptor = etp.fully_connected_tensor_product(
            irreps_in1, irreps_in2, irreps_out
        ).d
        assert descriptor.subscripts == "uvw,iu,jv,kw+ijk"

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
                torch.randn(self.weight_numel, device=device, dtype=dtype)
            )
        else:
            self.weight = None

        self.transpose_in1 = None
        self.transpose_in2 = None
        self.transpose_out = None

        if self.layout == cue.mul_ir:
            # descriptor = descriptor.add_or_transpose_modes("uvw,ui,vj,wk+ijk")
            self.transpose_in1 = cuet.TransposeIrrepsLayout(
                self.irreps_in1, source=cue.mul_ir, target=cue.ir_mul, device=device
            )
            self.transpose_in2 = cuet.TransposeIrrepsLayout(
                self.irreps_in2, source=cue.mul_ir, target=cue.ir_mul, device=device
            )
            self.transpose_out = cuet.TransposeIrrepsLayout(
                self.irreps_out, source=cue.ir_mul, target=cue.mul_ir, device=device
            )

        self.f = cuet.TensorProduct(
            descriptor,
            device=device,
            math_dtype=math_dtype,
            optimize_fallback=optimize_fallback,
        )
        self.descriptor = descriptor

    def extra_repr(self) -> str:
        return (
            f"{self.irreps_in1} x {self.irreps_in2} --> {self.irreps_out}"
            f", shared_weights={self.shared_weights}, internal_weights={self.internal_weights}, layout={self.layout}"
            f", weight_numel={self.weight_numel}"
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        *,
        use_fallback: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the fully connected tensor product operation.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor for the first operand. It should have the shape (batch_size, irreps_in1.dim).
        x2 : torch.Tensor
            Input tensor for the second operand. It should have the shape (batch_size, irreps_in2.dim).
        weight : torch.Tensor, optional
            Weights for the tensor product. It should have the shape (batch_size, weight_numel)
            if shared_weights is False, or (weight_numel,) if shared_weights is True.
            If None, the internal weights are used.
        use_fallback : Optional[bool], optional
            If `None` (default), a CUDA kernel will be used if available.
            If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
            If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Returns
        -------
        torch.Tensor
            Output tensor resulting from the fully connected tensor product operation. It will have the shape
            (batch_size, irreps_out.dim).

        Raises
        ------
        ValueError
            If internal weights are used and weight is not None.
            If shared weights are used and weight is not a 1D tensor.
            If shared weights are not used and weight is not a 2D tensor.
        """
        if self.transpose_in1 is not None:
            x1 = self.transpose_in1(x1, use_fallback=use_fallback)
        if self.transpose_in2 is not None:
            x2 = self.transpose_in2(x2, use_fallback=use_fallback)

        if self.internal_weights:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")

            weight = self.weight

        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights should be 1D tensor")
        if not self.shared_weights and weight.ndim != 2:
            raise ValueError("Weights should be 2D tensor")

        out = self.f(weight, x1, x2, use_fallback=use_fallback)

        if self.transpose_out is not None:
            out = self.transpose_out(out, use_fallback=use_fallback)

        return out
