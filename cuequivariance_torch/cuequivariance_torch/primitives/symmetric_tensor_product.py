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
import logging
import math
from typing import Optional

import torch
import torch.fx

import cuequivariance.segmented_tensor_product as stp
import cuequivariance_torch as cuet
from cuequivariance_torch.primitives.tensor_product import broadcast_shapes, prod

logger = logging.getLogger(__name__)


class SymmetricTensorProduct(torch.nn.Module):
    """
    PyTorch module

    Args:
        descriptors (list of SegmentedTensorProduct): The list of SegmentedTensorProduct descriptors.
        math_dtype (torch.dtype, optional): The data type of the coefficients and calculations.
        optimize_fallback (bool, optional): If `True`, the torch.fx graph will be optimized before execution. Because the optimization takes time, it is turned off by default.
    """

    def __init__(
        self,
        descriptors: list[stp.SegmentedTensorProduct],
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        optimize_fallback: Optional[bool] = None,
    ):
        super().__init__()

        self.descriptors = descriptors

        if any(d.num_operands < 2 for d in descriptors):
            d0 = next(d for d in descriptors if d.num_operands == 1)
            descriptors = [d for d in descriptors if d.num_operands >= 2]
            assert len(descriptors) + 1 == len(self.descriptors)
            self.f0 = cuet.TensorProduct(
                d0,
                device=device,
                math_dtype=math_dtype,
                use_fallback=use_fallback,
                optimize_fallback=optimize_fallback,
            )
        else:
            self.f0 = None

        descriptors = [
            stp.SegmentedTensorProduct(
                operands=[stp.Operand.empty_segments(1)] + d.operands,
                paths=[
                    stp.Path((0,) + path.indices, path.coefficients) for path in d.paths
                ],
                coefficient_subscripts=d.coefficient_subscripts,
            )
            for d in descriptors
        ]
        try:
            d = next(d for d in descriptors if d.num_operands >= 1)
        except StopIteration:
            raise ValueError("At least one STP must have at least 2 operands.")

        self.x0_size = d.operands[0].size
        self.x1_size = d.operands[1].size

        self.f = cuet.IWeightedSymmetricTensorProduct(
            descriptors,
            device=device,
            math_dtype=math_dtype,
            use_fallback=use_fallback,
            optimize_fallback=optimize_fallback,
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        r"""
        Perform the forward pass of the indexed symmetric tensor product operation.

        Args:
            x0 (torch.Tensor): The input tensor for the first operand. It should have the shape (batch, x0_size).
            use_fallback (bool, optional):  If `None` (default), a CUDA kernel will be used if available.
                If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
                If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Returns:
            torch.Tensor:
                The output tensor resulting from the indexed symmetric tensor product operation.
                It will have the shape (batch, x1_size).
        """
        out = self.f(
            torch.ones((1, 1), dtype=x0.dtype, device=x0.device),
            torch.zeros((x0.shape[0],), dtype=torch.int32, device=x0.device),
            x0,
        )
        if self.f0 is not None:
            out += self.f0([])
        return out


class IWeightedSymmetricTensorProduct(torch.nn.Module):
    """
    PyTorch module

    Parameters
    ----------
    descriptors : list[stp.SegmentedTensorProduct]
        The list of SegmentedTensorProduct descriptors
    math_dtype : torch.dtype, optional
        The data type of the coefficients and calculations
    optimize_fallback : bool, optional
        If `True`, the torch.fx graph will be optimized before execution
        Because the optimization takes time, it is turned off by default.
    """

    def __init__(
        self,
        descriptors: list[stp.SegmentedTensorProduct],
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        optimize_fallback: Optional[bool] = None,
    ):
        super().__init__()

        if math_dtype is None:
            math_dtype = torch.get_default_dtype()

        _check_descriptors(descriptors)
        self.descriptors = descriptors

        d = next(d for d in descriptors if d.num_operands >= 3)
        self.x0_size = d.operands[0].size
        self.x1_size = d.operands[1].size
        self.x2_size = d.operands[-1].size

        self.has_cuda = False

        self.f = None

        if use_fallback is None or use_fallback is False:
            try:
                self.f = CUDAKernel(descriptors, device, math_dtype)
                self.has_cuda = True
                return
            except NotImplementedError as e:
                logger.info(f"Failed to initialize CUDA implementation: {e}")
            except ImportError as e:
                logger.warning(f"Failed to initialize CUDA implementation: {e}")

        if use_fallback is False and not self.has_cuda:
            raise RuntimeError(
                "`use_fallback` is `False` and no CUDA kernel is available!"
            )

        if self.f is None:
            self.f = FallbackImpl(
                descriptors,
                device,
                math_dtype=math_dtype,
                optimize_fallback=optimize_fallback,
            )

    def __repr__(self):
        has_cuda_kernel = (
            "(with CUDA kernel)"
            if self.has_cuda is not None
            else "(without CUDA kernel)"
        )
        return f"IWeightedSymmetricTensorProduct({has_cuda_kernel})"

    def forward(
        self,
        x0: torch.Tensor,
        i0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Perform the forward pass of the indexed symmetric tensor product operation.

        Parameters
        ----------

        x0 : torch.Tensor
            The input tensor for the first operand. It should have the shape (i0.max() + 1, x0_size).
        i0 : torch.Tensor
            The index tensor for the first operand. It should have the shape (...).
        x1 : torch.Tensor
            The repeated input tensor. It should have the shape (..., x1_size).

        Returns
        -------
        torch.Tensor
            The output tensor resulting from the indexed symmetric tensor product operation.
            It will have the shape (batch, x2_size).
        """

        torch._assert(
            x0.ndim == 2,
            f"Expected 2 dims (i0.max() + 1, x0_size), got shape {x0.shape}",
        )
        shape = broadcast_shapes([i0.shape, x1.shape[:-1]])
        i0 = i0.expand(shape).reshape((prod(shape),))
        x1 = x1.expand(shape + (x1.shape[-1],)).reshape((prod(shape), x1.shape[-1]))

        out = self.f(x0, i0, x1)
        out = out.reshape(shape + (self.x2_size,))
        return out


def _check_descriptors(descriptors: list[stp.SegmentedTensorProduct]):
    if len(descriptors) == 0:
        raise ValueError("stps must contain at least one STP.")

    try:
        d = next(d for d in descriptors if d.num_operands >= 3)
    except StopIteration:
        raise ValueError("At least one STP must have at least 3 operands.")

    x0 = d.operands[0]
    x1 = d.operands[1]
    x2 = d.operands[-1]

    for d in descriptors:
        if d.operands[0].size != x0.size:
            raise ValueError("All STPs must have the same first operand (x0).")

        if any(ope.size != x1.size for ope in d.operands[1:-1]):
            raise ValueError("All STPs must have the operands[1:-1] identical (x1).")

        if d.operands[-1].size != x2.size:
            raise ValueError("All STPs must have the same last operand (x2, output).")


class CUDAKernel(torch.nn.Module):
    def __init__(
        self,
        stps: list[stp.SegmentedTensorProduct],
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()

        max_degree = max(d.num_operands - 2 for d in stps)
        if max_degree > 6:
            raise NotImplementedError("Correlation > 6 is not implemented.")
        if min(d.num_operands for d in stps) == 2:
            raise NotImplementedError(
                "Only STPs with at least 3 operands are supported."
            )

        def f(d: stp.SegmentedTensorProduct) -> stp.SegmentedTensorProduct:
            d = d.move_operand(0, -2)
            d = d.flatten_coefficient_modes(force=True)
            d = d.flatten_modes(
                [
                    m
                    for m in d.subscripts.modes()
                    if not all(m in ope.subscripts for ope in d.operands)
                ]
            )
            d = d.consolidate_modes()

            # ops.SymmetricTensorContraction will "symmetrize" for the derivatives so we can sort for the forward pass
            d = d.sort_indices_for_identical_operands(range(0, d.num_operands - 2))

            if len(set(ope.subscripts for ope in d.operands)) != 1:
                raise NotImplementedError("Different subscripts are not supported.")
            return d

        ds = [f(d) for d in stps]

        if (
            len(
                set(
                    (
                        d.operands[0].num_segments,
                        d.operands[-2].num_segments,
                        d.operands[-1].num_segments,
                    )
                    for d in ds
                )
            )
            != 1
        ):
            raise ValueError("All STPs must have the same number of segments.")

        import cuequivariance_ops_torch as ops

        self.f = ops.SymmetricTensorContraction(
            sum((d.indices.tolist() for d in ds), []),
            sum((d.stacked_coefficients.tolist() for d in ds), []),
            ds[0].operands[0].num_segments,
            ds[0].operands[-2].num_segments,
            ds[0].operands[-1].num_segments,
            max_degree,
            math_dtype,
        ).to(device=device)
        self.u = ds[0].operands[0].size // ds[0].operands[0].num_segments
        self.descriptors = ds

    def forward(
        self, x0: torch.Tensor, i0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        r"""
        .. math::

            x_2[j_{n+1}] = val x_0[i_0][j_0] \prod_{k=1}^{n} x_1[j_k]

        """
        i0 = i0.to(torch.int32)
        x0 = x0.reshape(x0.shape[0], x0.shape[1] // self.u, self.u)
        x1 = x1.reshape(x1.shape[0], x1.shape[1] // self.u, self.u)
        if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
            logger.debug(
                f"Calling SymmetricTensorContraction: {self.descriptors}, input shapes: {x0.shape}, {i0.shape}, {x1.shape}"
            )
        out = self.f(x1, x0, i0)
        out = out.reshape(out.shape[0], out.shape[1] * self.u)
        return out


class FallbackImpl(torch.nn.Module):
    def __init__(
        self,
        stps: list[stp.SegmentedTensorProduct],
        device: Optional[torch.device],
        math_dtype: Optional[torch.dtype],
        optimize_fallback: Optional[bool],
    ):
        super().__init__()
        self.fs = torch.nn.ModuleList(
            [
                cuet.TensorProduct(
                    d,
                    device=device,
                    math_dtype=math_dtype,
                    use_fallback=True,
                    optimize_fallback=optimize_fallback,
                )
                for d in stps
            ]
        )

    def forward(
        self, x0: torch.Tensor, i0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        return sum(
            f([x0[i0]] + [x1] * (f.descriptor.num_operands - 2)) for f in self.fs
        )
