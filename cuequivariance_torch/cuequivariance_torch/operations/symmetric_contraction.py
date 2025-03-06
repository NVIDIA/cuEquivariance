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
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction,
)
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
)


class SymmetricContraction(torch.nn.Module):
    """
    Accelerated implementation of the symmetric contraction operation introduced in https://arxiv.org/abs/2206.07697.

    Args:
        irreps_in (Irreps): The input irreps. All multiplicities (mul) within the irreps must be identical,
            indicating that each irrep appears the same number of times.
        irreps_out (Irreps): The output irreps. Similar to `irreps_in`, all multiplicities must be the same.
        contraction_degree (int): The degree of the symmetric contraction, specifying the maximum degree of the
            polynomial in the symmetric contraction.
        num_elements (int): The number of elements for the weight tensor.
        layout (IrrepsLayout, optional): The layout of the input and output irreps. If not provided, a default layout is used.
        math_dtype (torch.dtype, optional): The data type for mathematical operations. If not specified, the default data type
            from the torch environment is used.
        use_fallback (bool, optional): If `None` (default), a CUDA kernel will be used if available.
                If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
                If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

    Examples:
        >>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        >>> irreps_in = cue.Irreps("O3", "32x0e + 32x1o")
        >>> irreps_out = cue.Irreps("O3", "32x0e")
        >>> layer = SymmetricContraction(irreps_in, irreps_out, contraction_degree=3, num_elements=5, layout=cue.ir_mul, dtype=torch.float32, device=device)

        Now `layer` can be used as part of a PyTorch model.

        The argument `original_mace` can be set to `True` to emulate the original MACE implementation.

        >>> feats_irreps = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")
        >>> target_irreps = cue.Irreps("O3", "32x0e + 32x1o")
        >>> # OLD FUNCTION DEFINITION:
        >>> # symmetric_contractions_old = SymmetricContraction(
        >>> #     irreps_in=feats_irreps,
        >>> #     irreps_out=target_irreps,
        >>> #     correlation=3,
        >>> #     num_elements=10,
        >>> # )
        >>> # NEW FUNCTION DEFINITION:
        >>> symmetric_contractions_new = cuet.SymmetricContraction(
        ...     irreps_in=feats_irreps,
        ...     irreps_out=target_irreps,
        ...     contraction_degree=3,
        ...     num_elements=10,
        ...     layout_in=cue.ir_mul,
        ...     layout_out=cue.mul_ir,
        ...     original_mace=True,
        ...     dtype=torch.float64,
        ...     device=device,
        ... )

        Then the execution is as follows:

        >>> node_feats = torch.randn(128, 32, feats_irreps.dim // 32, dtype=torch.float64, device=device)
        >>> # with node_attrs_index being the index version of node_attrs, sth like:
        >>> # node_attrs_index = torch.nonzero(node_attrs)[:, 1].int()
        >>> node_attrs_index = torch.randint(0, 10, (128,), dtype=torch.int32, device=device)
        >>> # OLD CALL:
        >>> # symmetric_contractions_old(node_feats, node_attrs)
        >>> # NEW CALL:
        >>> node_feats = torch.transpose(node_feats, 1, 2).flatten(1)
        >>> symmetric_contractions_new(node_feats, node_attrs_index)
        tensor([[...)

    Note:
        The term 'mul' refers to the multiplicity of an irrep, indicating how many times it appears
        in the representation. This layer requires that all input and output irreps have the same
        multiplicity for the symmetric contraction operation to be well-defined.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        contraction_degree: int,
        num_elements: int,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[torch.dtype] = None,
        original_mace: bool = False,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()

        if dtype is None:
            dtype = torch.get_default_dtype()

        irreps_in, irreps_out = default_irreps(irreps_in, irreps_out)
        assert_same_group(irreps_in, irreps_out)
        self.contraction_degree = contraction_degree

        if len(set(irreps_in.muls) | set(irreps_out.muls)) != 1:
            raise ValueError("Input/Output irreps must have the same mul")

        mul = irreps_in.muls[0]

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.etp, p = symmetric_contraction(
            irreps_in, irreps_out, range(1, contraction_degree + 1)
        )
        if original_mace:
            self.register_buffer(
                "projection", torch.tensor(p, dtype=dtype, device=device)
            )
            self.weight_shape = (p.shape[0], mul)
        else:
            self.projection = None
            self.weight_shape = (self.etp.inputs[0].dim // mul, mul)

        self.num_elements = num_elements
        self.weight = torch.nn.Parameter(
            torch.randn(
                self.num_elements, *self.weight_shape, device=device, dtype=dtype
            )
        )

        self.f = cuet.EquivariantTensorProduct(
            self.etp,
            layout=layout,
            layout_in=layout_in,
            layout_out=layout_out,
            device=device,
            math_dtype=math_dtype or dtype,
            use_fallback=use_fallback,
        )

    def extra_repr(self) -> str:
        return (
            f"contraction_degree={self.contraction_degree}"
            f", weight_shape={self.weight_shape}"
        )

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the symmetric contraction operation.

        Args:
            x (torch.Tensor): The input tensor. It should have shape (batch, irreps_in.dim).
            indices (torch.Tensor): The index of the weight to use for each batch element.
                It should have shape (batch,).

        Returns:
            torch.Tensor: The output tensor. It has shape (batch, irreps_out.dim).
        """

        if self.projection is not None:
            weight = torch.einsum("zau,ab->zbu", self.weight, self.projection)
        else:
            weight = self.weight
        weight = weight.flatten(1)

        return self.f(weight, x, indices=indices)
