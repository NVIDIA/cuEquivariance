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
    default_irreps,
    default_layout,
)

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors


class Rotation(torch.nn.Module):
    """
    A class that represents a rotation layer for SO3 or O3 representations.

    Args:
        irreps (Irreps): The irreducible representations of the tensor to rotate.
        layout (IrrepsLayout, optional): The memory layout of the tensor, ``cue.ir_mul`` is preferred.
        layout_in (IrrepsLayout, optional): The layout of the input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        device (torch.device, optional): The device to use for the operation.
        math_dtype (torch.dtype or string, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors,
            if possible, or the torch default dtype (see SegmentedPolynomial for more details).
        method (str, optional): The method to use for the operation, by default "uniform_1d" (using a CUDA kernel)
            if all segments have the same shape, otherwise "naive" (using a PyTorch implementation).
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True` the "naive" method is used.
            If `False` the "uniform_1d" method is used (make sure all segments have the same shape).
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[str | torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        method: Optional[str] = None,
    ):
        super().__init__()
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.SO3, cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be SO3 or O3."
            )

        e = descriptors.yxy_rotation(irreps)
        same_shape = e.all_same_segment_shape()

        self.irreps = irreps
        self.lmax = max(ir.l for _, ir in irreps)

        layout_in = default_layout(layout_in or layout)
        self.transpose_in = cuet.TransposeIrrepsLayout(
            e.inputs[3].irreps,
            source=layout_in,
            target=e.inputs[3].layout,
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
                if same_shape:
                    # No warning here as it's the default behavior
                    self.method = "uniform_1d"
                else:
                    warnings.warn(
                        "Segments are not the same shape, falling back to naive method\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                    self.method = "naive"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                if not same_shape and not use_fallback:
                    raise ValueError(
                        "Uniform 1D method requires segments to be the same shape\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                self.method = "naive" if use_fallback else "uniform_1d"
        else:
            if method == "uniform_1d" and not same_shape:
                raise ValueError(
                    "Uniform 1D method requires segments to be the same shape\n"
                    "You can consider making the segments uniform in the descriptor."
                )
            self.method = method

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

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

        output = self.f(
            [encodings_gamma, encodings_beta, encodings_alpha, self.transpose_in(x)]
        )
        return self.transpose_out(output[0])


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
        layout_in (IrrepsLayout, optional): The layout of the input irreducible representations, by default ``layout``.
        layout_out (IrrepsLayout, optional): The layout of the output irreducible representations, by default ``layout``.
        device (torch.device, optional): The device to use for the linear layer.
        math_dtype (torch.dtype, optional): The dtype to use for the math operations, by default it follows the dtype of the input tensors.
        method (str, optional): The method to use for the linear layer, by default "uniform_1d" (using a CUDA kernel)
            if all segments have the same shape, otherwise "naive" (using a PyTorch implementation).
        use_fallback (bool, optional, deprecated): Whether to use a "fallback" implementation, now maps to method:
            If `True` the "naive" method is used.
            If `False` the "uniform_1d" method is used (make sure all segments have the same shape).
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
        method: Optional[str] = None,
    ):
        super().__init__()
        (irreps,) = default_irreps(irreps)

        if irreps.irrep_class not in [cue.O3]:
            raise ValueError(
                f"Unsupported irrep class {irreps.irrep_class}. Must be O3."
            )

        e = descriptors.inversion(irreps)
        same_shape = e.all_same_segment_shape()

        layout_in = default_layout(layout_in or layout)
        self.transpose_in = cuet.TransposeIrrepsLayout(
            e.inputs[0].irreps,
            source=layout_in,
            target=e.inputs[0].layout,
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
                if same_shape:
                    # No warning here as it's the default behavior
                    self.method = "uniform_1d"
                else:
                    warnings.warn(
                        "Segments are not the same shape, falling back to naive method\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                    self.method = "naive"
            else:
                warnings.warn(
                    "`use_fallback` is deprecated, please use `method` instead",
                    DeprecationWarning,
                )
                if not same_shape and not use_fallback:
                    raise ValueError(
                        "Uniform 1D method requires segments to be the same shape\n"
                        "You can consider making the segments uniform in the descriptor."
                    )
                self.method = "naive" if use_fallback else "uniform_1d"
        else:
            if method == "uniform_1d" and not same_shape:
                raise ValueError(
                    "Uniform 1D method requires segments to be the same shape\n"
                    "You can consider making the segments uniform in the descriptor."
                )
            self.method = method

        self.f = cuet.SegmentedPolynomial(
            e.polynomial,
            method=self.method,
            math_dtype=math_dtype,
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inversion layer."""
        output = self.f([self.transpose_in(x)])
        return self.transpose_out(output[0])
