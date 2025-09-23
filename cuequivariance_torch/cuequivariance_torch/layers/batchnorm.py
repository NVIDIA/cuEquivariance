# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from cuequivariance.group_theory.irreps_array.misc_ui import (
    default_irreps,
    default_layout,
)

import cuequivariance as cue


# This implementation is an adaptation of https://github.com/e3nn/e3nn/blob/ef93f876c9985b3816aefb2982b3cf4325df6ba4/e3nn/nn/_batchnorm.py
class BatchNorm(torch.nn.Module):
    """Batch normalization for orthonormal representations.

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.

    Args:
        irreps (Irreps): Input irreps.
        layout (IrrepsLayout, optional): Layout of the input tensor, by default `IrrepsLayout.mul_ir`.
        eps (float, optional): Epsilon value for numerical stability, by default 1e-5.
        momentum (float, optional): Momentum for the running mean and variance, by default 0.1.
        affine (bool, optional): Whether to apply an affine transformation, by default True.
        reduce (str, optional): How to reduce the norm of the representations, by default "mean".
        instance (bool, optional): Whether to use instance normalization, by default False.
        include_bias (bool, optional): Whether to include a bias term, by default True.
    """

    __constants__ = ["instance", "normalization", "irs", "affine"]

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        layout: cue.IrrepsLayout = None,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        reduce: str = "mean",
        instance: bool = False,
        include_bias: bool = True,
    ):
        super().__init__()
        self.layout = default_layout(layout)
        (self.irreps,) = default_irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance
        self.include_bias = include_bias

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps
        self.features = []

        if self.instance:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
        else:
            self.register_buffer("running_mean", torch.zeros(num_scalar))
            self.register_buffer("running_var", torch.ones(num_features))

        if affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            if self.include_bias:
                self.bias = torch.nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("weight", None)
            if self.include_bias:
                self.register_parameter("bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce
        irs = []
        for mul, ir in self.irreps:
            irs.append((mul, ir.dim, ir.is_scalar()))
        self.irs = irs

    def extra_repr(self) -> str:
        return f"{self.irreps}, layout={self.layout}, eps={self.eps}, momentum={self.momentum}"

    def _roll_avg(self, curr: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return (1 - self.momentum) * curr + self.momentum * update.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        Args:
            input (torch.Tensor): Input tensor. The last dimension should match with the input irreps.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        orig_shape = input.shape
        batch = input.shape[0]
        dim = input.shape[-1]
        input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]

        if self.training and not self.instance:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, d, is_scalar in self.irs:
            field = input[:, :, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            if self.layout == cue.mul_ir:
                # [batch, sample, mul, repr]
                field = field.reshape(batch, -1, mul, d)
            elif self.layout == cue.ir_mul:
                # [batch, sample, repr, mul]
                field = field.reshape(batch, -1, d, mul).transpose(2, 3)
            else:
                raise ValueError(f"Invalid layout option {self.layout}")

            if is_scalar:
                if self.training or self.instance:
                    if self.instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(
                            self._roll_avg(
                                self.running_mean[irm : irm + mul], field_mean
                            )
                        )
                else:
                    field_mean = self.running_mean[irm : irm + mul]
                irm += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if self.training or self.instance:
                field_norm = field.pow(2).mean(3)  # [batch, sample, mul]

                if self.reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif self.reduce == "max":
                    field_norm = field_norm.max(1).values  # [batch, mul]
                else:
                    raise ValueError(f"Invalid reduce option {self.reduce}")

                if not self.instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(
                        self._roll_avg(self.running_var[irv : irv + mul], field_norm)
                    )
            else:
                field_norm = self.running_var[irv : irv + mul]
            irv += mul

            field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[iw : iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [(batch,) mul]

            field = field * field_norm.reshape(
                -1, 1, mul, 1
            )  # [batch, sample, mul, repr]

            if self.affine and self.include_bias and is_scalar:
                bias = self.bias[ib : ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch, sample, mul, repr]

            if self.layout == cue.mul_ir:
                pass
            elif self.layout == cue.ir_mul:
                field = field.transpose(2, 3)

            fields.append(
                field.reshape(batch, -1, mul * d)
            )  # [batch, sample, mul * repr]

        torch._assert(
            ix == dim,
            f"`ix` should have reached input.size(-1) ({dim}), but it ended at {ix}",
        )

        if self.training and not self.instance:
            torch._assert(
                irm == self.running_mean.numel(), "irm == self.running_mean.numel()"
            )
            torch._assert(
                irv == self.running_var.size(0), "irv == self.running_var.size(0)"
            )
        if self.affine:
            torch._assert(iw == self.weight.size(0), "iw == self.weight.size(0)")
            if self.include_bias:
                torch._assert(ib == self.bias.numel(), "ib == self.bias.numel()")

        if self.training and not self.instance:
            if len(new_means) > 0:
                torch.cat(new_means, out=self.running_mean)
            if len(new_vars) > 0:
                torch.cat(new_vars, out=self.running_var)

        output = torch.cat(fields, dim=2)  # [batch, sample, stacked features]
        return output.reshape(orig_shape)
