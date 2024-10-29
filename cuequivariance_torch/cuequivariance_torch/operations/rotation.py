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
from cuequivariance.irreps_array.misc_ui import default_irreps, default_layout


class Rotation(torch.nn.Module):
    """
    A class that represents a rotation layer for SO3 or O3 representations.

    Parameters
    ----------
    irreps : cue.Irreps
        The irreducible representations of the tensor to rotate.
    layout : cue.IrrepsLayout, optional
        The memory layout of the tensor, cue.ir_mul is preferred.
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        layout: cue.IrrepsLayout = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        optimize_fallback: Optional[bool] = None,
    ):
        super().__init__()
        self.layout = layout = default_layout(layout)
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.SO3, cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be SO3 or O3."
            )

        self.transpose_in = None
        self.transpose_out = None

        if layout == cue.ir_mul:
            pass
        elif layout == cue.mul_ir:
            self.transpose_in = cuet.TransposeIrrepsLayout(
                irreps, source=cue.mul_ir, target=cue.ir_mul, device=device
            )
            self.transpose_out = cuet.TransposeIrrepsLayout(
                irreps, source=cue.ir_mul, target=cue.mul_ir, device=device
            )
        else:
            raise ValueError(f"Unsupported layout {layout}")

        self.irreps = irreps

        self.f = cuet.EquivariantTensorProduct(
            etp.yxy_rotation(irreps),
            device=device,
            math_dtype=math_dtype,
            optimize_fallback=optimize_fallback,
        )
        self.lmax = max(ir.l for _, ir in irreps)

    def extra_repr(self) -> str:
        return f"{self.irreps}, layout={self.layout}"

    def forward(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        x: torch.Tensor,
        *,
        use_fallback: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the rotation layer.

        Parameters
        ----------
        gamma : torch.Tensor
            The gamma angles. First rotation around the y-axis.
        beta : torch.Tensor
            The beta angles. Second rotation around the x-axis.
        alpha : torch.Tensor
            The alpha angles. Third rotation around the y-axis.
        x : torch.Tensor
            The input tensor.
        use_fallback : Optional[bool], optional
            If `None` (default), a CUDA kernel will be used if available.
            If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
            If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Returns
        -------
        torch.Tensor
            The rotated tensor.
        """
        if self.transpose_in is not None:
            x = self.transpose_in(x)

        gamma = torch.as_tensor(gamma, dtype=x.dtype, device=x.device)
        beta = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
        alpha = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)

        encodings_gamma = encode_rotation_angle(gamma, self.lmax)
        encodings_beta = encode_rotation_angle(beta, self.lmax)
        encodings_alpha = encode_rotation_angle(alpha, self.lmax)

        out = self.f(
            encodings_gamma,
            encodings_beta,
            encodings_alpha,
            x,
            use_fallback=use_fallback,
        )

        if self.transpose_out is not None:
            out = self.transpose_out(out)

        return out


def encode_rotation_angle(angle: torch.Tensor, l: int) -> torch.Tensor:
    """Encode a angle into a tensor of cosines and sines.

    The encoding is [cos(l * angle), cos((l - 1) * angle), ..., cos(angle), 1, sin(angle), sin(2 * angle), ..., sin(l * angle)].
    This encoding is used to feed the segmented tensor products that perform rotations.
    """
    angle = torch.as_tensor(angle)
    angle = angle.unsqueeze(-1)

    m = torch.arange(1, l + 1, device=angle.device, dtype=angle.dtype)
    c = torch.cos(m * angle)
    s = torch.sin(m * angle)
    one = torch.ones_like(angle)
    return torch.cat([c.flip(-1), one, s], dim=-1)


def vector_to_euler_angles(vector: torch.Tensor) -> torch.Tensor:
    assert vector.shape[-1] == 3
    shape = vector.shape[:-1]
    vector = vector.reshape(-1, 3)

    x, y, z = torch.nn.functional.normalize(vector, dim=-1).T

    x_ = torch.where((x == 0.0) & (z == 0.0), 0.0, x)
    y_ = torch.where((x == 0.0) & (z == 0.0), 0.0, y)
    z_ = torch.where((x == 0.0) & (z == 0.0), 1.0, z)

    beta = torch.where(y == 1.0, 0.0, torch.where(y == -1, torch.pi, torch.acos(y_)))
    alpha = torch.atan2(x_, z_)

    beta = beta.reshape(shape)
    alpha = alpha.reshape(shape)

    return beta, alpha


class Inversion(torch.nn.Module):
    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be O3."
            )

        self.irreps = irreps
        self.f = cuet.EquivariantTensorProduct(
            etp.inversion(irreps), device=device, math_dtype=math_dtype
        )

    def extra_repr(self) -> str:
        return f"{self.irreps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)
