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
from cuequivariance.group_theory.irreps_array.misc_ui import default_irreps


class Rotation(torch.nn.Module):
    """
    A class that represents a rotation layer for SO3 or O3 representations.

    Args:
        irreps (Irreps): The irreducible representations of the tensor to rotate.
        layout (IrrepsLayout, optional): The memory layout of the tensor, ``cue.ir_mul`` is preferred.
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.SO3, cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be SO3 or O3."
            )

        self.irreps = irreps
        self.lmax = max(ir.l for _, ir in irreps)

        self.f = cuet.EquivariantTensorProduct(
            descriptors.yxy_rotation(irreps),
            layout=layout,
            layout_in=layout_in,
            layout_out=layout_out,
            device=device,
            math_dtype=math_dtype,
            use_fallback=use_fallback,
        )

    def forward(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the rotation layer.

        Args:
            gamma (torch.Tensor): The gamma angles. First rotation around the y-axis.
            beta (torch.Tensor): The beta angles. Second rotation around the x-axis.
            alpha (torch.Tensor): The alpha angles. Third rotation around the y-axis.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The rotated tensor.
        """
        gamma = torch.as_tensor(gamma, dtype=x.dtype, device=x.device)
        beta = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
        alpha = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)

        encodings_gamma = encode_rotation_angle(gamma, self.lmax)
        encodings_beta = encode_rotation_angle(beta, self.lmax)
        encodings_alpha = encode_rotation_angle(alpha, self.lmax)

        return self.f(encodings_gamma, encodings_beta, encodings_alpha, x)


def encode_rotation_angle(angle: torch.Tensor, ell: int) -> torch.Tensor:
    """Encode a angle into a tensor of cosines and sines.

    The encoding is::

        [cos(l * angle), cos((l - 1) * angle), ..., cos(angle), 1, sin(angle), sin(2 * angle), ..., sin(l * angle)].

    This encoding is used to feed the segmented tensor products that perform rotations.
    """
    angle = torch.as_tensor(angle)
    angle = angle.unsqueeze(-1)

    m = torch.arange(1, ell + 1, device=angle.device, dtype=angle.dtype)
    c = torch.cos(m * angle)
    s = torch.sin(m * angle)
    one = torch.ones_like(angle)
    return torch.cat([c.flip(-1), one, s], dim=-1)


def vector_to_euler_angles(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert a 3D vector to Euler angles.

    .. math::

        R_y(\alpha) R_x(\beta) \vec{e}_y = \vec{v}

    Args:
        vector (torch.Tensor): The 3D vector.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The beta and alpha angles.
    """
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
    """
    Inversion layer for :math:`O(3)` representations.

    Args:
        irreps (Irreps): The irreducible representations of the tensor to invert.
        layout (IrrepsLayout, optional): The memory layout of the tensor, ``cue.ir_mul`` is preferred.
        use_fallback (bool, optional): If `None` (default), a CUDA kernel will be used if available.
                If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
                If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be O3."
            )

        self.irreps = irreps
        self.f = cuet.EquivariantTensorProduct(
            descriptors.inversion(irreps),
            layout=layout,
            layout_in=layout_in,
            layout_out=layout_out,
            device=device,
            math_dtype=math_dtype,
            use_fallback=use_fallback,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inversion layer."""
        return self.f(x)
