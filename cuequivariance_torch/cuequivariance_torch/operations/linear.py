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
import cuequivariance.equivariant_tensor_product as etp
import cuequivariance_torch as cuet
from cuequivariance.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
    default_layout,
)


class Linear(torch.nn.Module):
    """
    A class that represents an equivariant linear layer.

    Parameters
    ----------
    irreps_in : cue.Irreps
        The input irreducible representations.
    irreps_out : cue.Irreps
        The output irreducible representations.
    layout : cue.IrrepsLayout, optional
        The layout of the irreducible representations, by default cue.mul_ir. This is the layout used in the e3nn library.
    shared_weights : bool, optional
        Whether to use shared weights, by default True.
    internal_weights : bool, optional
        Whether to use internal weights, by default True if shared_weights is True, otherwise False.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
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
        irreps_in, irreps_out = default_irreps(irreps_in, irreps_out)
        assert_same_group(irreps_in, irreps_out)

        e = etp.linear(irreps_in, irreps_out)
        assert e.d.subscripts == "uv,iu,iv"

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.transpose_in = None
        self.transpose_out = None

        if layout == cue.ir_mul:
            pass
        elif layout == cue.mul_ir:
            self.transpose_in = cuet.TransposeIrrepsLayout(
                self.irreps_in, source=cue.mul_ir, target=cue.ir_mul, device=device
            )
            self.transpose_out = cuet.TransposeIrrepsLayout(
                self.irreps_out, source=cue.ir_mul, target=cue.mul_ir, device=device
            )
        else:
            raise ValueError(f"Unsupported layout {layout}")

        self.weight_numel = e.inputs[0].irreps.dim

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

        self.f = cuet.EquivariantTensorProduct(
            e,
            device=device,
            math_dtype=math_dtype,
            optimize_fallback=optimize_fallback,
        )

    def extra_repr(self) -> str:
        return f"{self.irreps_in} --> {self.irreps_out}, shared_weights={self.shared_weights}, internal_weights={self.internal_weights}, layout={self.layout}, weight_numel={self.weight_numel}"

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        *,
        use_fallback: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the linear layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        weight : torch.Tensor, optional
            The weight tensor. If None, the internal weight tensor is used.
        use_fallback : Optional[bool], optional
            If `None` (default), a CUDA kernel will be used if available.
            If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
            If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the linear transformation.

        Raises
        ------
        ValueError
            If internal weights are used and weight is not None.
            If shared weights are used and weight is not a 1D tensor.
            If shared weights are not used and weight is not a 2D tensor.
        """
        if self.internal_weights:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")

            weight = self.weight

        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights should be 1D tensor")
        if not self.shared_weights and weight.ndim != 2:
            raise ValueError("Weights should be 2D tensor")

        if self.transpose_in is not None:
            x = self.transpose_in(x, use_fallback=use_fallback)

        out = self.f(weight, x, use_fallback=use_fallback)

        if self.transpose_out is not None:
            out = self.transpose_out(out, use_fallback=use_fallback)

        return out
