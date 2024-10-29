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

import cuequivariance.equivariant_tensor_product as etp
import cuequivariance_torch as cuet


class EquivariantTensorProduct(torch.nn.Module):
    def __init__(
        self,
        e: etp.EquivariantTensorProduct,
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        optimize_fallback: Optional[bool] = None,
    ):
        super().__init__()

        self.etp = e

        if any(d.num_operands != e.num_inputs + 1 for d in e.ds):
            self.tp = None

            if e.num_inputs == 1:
                self.symm_tp = cuet.SymmetricTensorProduct(
                    e.ds,
                    device=device,
                    math_dtype=math_dtype,
                    optimize_fallback=optimize_fallback,
                )
            elif e.num_inputs == 2:
                self.symm_tp = cuet.IWeightedSymmetricTensorProduct(
                    e.ds,
                    device=device,
                    math_dtype=math_dtype,
                    optimize_fallback=optimize_fallback,
                )
            else:
                raise NotImplementedError("This should not happen")
        else:
            [d] = e.ds

            self.tp = cuet.TensorProduct(
                d,
                device=device,
                math_dtype=math_dtype,
                optimize_fallback=optimize_fallback,
            )
            self.symm_tp = None

    def extra_repr(self) -> str:
        return str(self.etp)

    def forward(
        self,
        *inputs: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        use_fallback: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        If ``indices`` is not None, the first input is indexed by ``indices``.
        """
        inputs: list[torch.Tensor] = list(inputs)

        assert len(inputs) == len(self.etp.inputs)
        for a, b in zip(inputs, self.etp.inputs):
            assert a.shape[-1] == b.irreps.dim

        if self.tp is not None:
            if indices is not None:
                # TODO: at some point we will have kernel for this
                assert len(inputs) >= 1
                inputs[0] = inputs[0][indices]
            return self.tp(*inputs, use_fallback=use_fallback)

        if self.symm_tp is not None:
            if len(inputs) == 1:
                assert indices is None
                return self.symm_tp(inputs[0], use_fallback=use_fallback)

            if len(inputs) == 2:
                [x0, x1] = inputs
                if indices is None:
                    if x0.shape[0] == 1:
                        indices = torch.zeros(
                            (x1.shape[0],), dtype=torch.int32, device=x1.device
                        )
                    elif x0.shape[0] == x1.shape[0]:
                        indices = torch.arange(
                            x1.shape[0], dtype=torch.int32, device=x1.device
                        )

                if indices is not None:
                    return self.symm_tp(x0, indices, x1, use_fallback=use_fallback)

        raise NotImplementedError("This should not happen")
