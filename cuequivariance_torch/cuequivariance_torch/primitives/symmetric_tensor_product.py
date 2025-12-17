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
import logging
import math
from typing import List, Optional

import torch
import torch.fx

import cuequivariance as cue
import cuequivariance_torch as cuet

logger = logging.getLogger(__name__)


class SymmetricTensorProduct(torch.nn.Module):
    """
    PyTorch module

    Args:
        descriptors (list of SegmentedTensorProduct): The list of SegmentedTensorProduct descriptors.
        math_dtype (torch.dtype, optional): The data type of the coefficients and calculations.
    """

    def __init__(
        self,
        descriptors: list[cue.SegmentedTensorProduct],
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()

        self.descriptors = descriptors

        descriptors = [
            cue.SegmentedTensorProduct(
                operands_and_subscripts=[(cue.SegmentedOperand.empty_segments(1), "")]
                + list(d.operands_and_subscripts),
                paths=[
                    cue.segmented_polynomials.Path(
                        (0,) + path.indices, path.coefficients
                    )
                    for path in d.paths
                ],
                coefficient_subscripts=d.coefficient_subscripts,
            )
            for d in descriptors
        ]
        d_max = max(descriptors, key=lambda d: d.num_operands)

        self.x0_size = d_max.operands[0].size
        self.x1_size = d_max.operands[1].size if d_max.num_operands >= 3 else 1

        self.f = cuet.IWeightedSymmetricTensorProduct(
            descriptors,
            device=device,
            math_dtype=math_dtype,
            use_fallback=use_fallback,
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
        torch._assert(
            x0.ndim == 2, f"Expected 2 dims (batch, x0_size), got shape {x0.shape}"
        )
        return self.f(
            torch.ones((1, 1), dtype=x0.dtype, device=x0.device),
            torch.zeros((x0.shape[0],), dtype=torch.int32, device=x0.device),
            x0,
        )


class IWeightedSymmetricTensorProduct(torch.nn.Module):
    """
    PyTorch module

    Parameters
    ----------
    descriptors : list[cue.SegmentedTensorProduct]
        The list of SegmentedTensorProduct descriptors
    math_dtype : torch.dtype, optional
        The data type of the coefficients and calculations
    """

    def __init__(
        self,
        descriptors: list[cue.SegmentedTensorProduct],
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()
        logger.warning("SymmetricTensorProduct is deprecated and will be removed soon.")
        logger.warning("Please use SegmentedPolynomial instead.")

        if math_dtype is None:
            math_dtype = torch.get_default_dtype()

        _check_descriptors(descriptors)
        self.descriptors = descriptors

        d = max(descriptors, key=lambda d: d.num_operands)
        self.x0_size = d.operands[0].size
        self.x1_size = d.operands[1].size if d.num_operands >= 3 else 1
        self.x2_size = d.operands[-1].size

        self.has_cuda = False

        if use_fallback is False:
            self.f = CUDAKernel(descriptors, device, math_dtype)
            self.has_cuda = True
        elif use_fallback is None:
            try:
                self.f = CUDAKernel(descriptors, device, math_dtype)
                self.has_cuda = True
            except NotImplementedError as e:
                logger.info(f"Failed to initialize CUDA implementation: {e}")
            except ImportError as e:
                logger.warning(f"Failed to initialize CUDA implementation: {e}")

        if not self.has_cuda:
            self.f = FallbackImpl(
                descriptors,
                device,
                math_dtype=math_dtype,
            )

    @torch.jit.ignore
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
            The index tensor for the first operand. It should have the shape (batch).
        x1 : torch.Tensor
            The repeated input tensor. It should have the shape (batch, x1_size).

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
        torch._assert(
            i0.ndim == 1,
            f"Expected 1 dim (batch), got shape {i0.shape}",
        )
        torch._assert(
            x1.ndim == 2,
            f"Expected 2 dims (batch, x1_size), got shape {x1.shape}",
        )
        return self.f(x0, i0, x1)


def _check_descriptors(descriptors: list[cue.SegmentedTensorProduct]):
    if len(descriptors) == 0:
        raise ValueError("stps must contain at least one STP.")

    d_max = max(descriptors, key=lambda d: d.num_operands)
    assert d_max.num_operands >= 2  # at least x0 and x2

    for d in descriptors:
        if d.operands[0].size != d_max.operands[0].size:
            raise ValueError("All STPs must have the same first operand (x0).")

        if any(ope.size != d_max.operands[1].size for ope in d.operands[1:-1]):
            raise ValueError("All STPs must have the operands[1:-1] identical (x1).")

        if d.operands[-1].size != d_max.operands[-1].size:
            raise ValueError("All STPs must have the same last operand (x2, output).")


class CUDAKernel(torch.nn.Module):
    def __init__(
        self,
        ds: list[cue.SegmentedTensorProduct],
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()

        if not torch.cuda.is_available():
            raise NotImplementedError("CUDA is not available.")

        max_degree = max(d.num_operands - 2 for d in ds)

        if max_degree > 6:
            raise NotImplementedError("Correlation > 6 is not implemented.")

        if len({d.operands[0].num_segments for d in ds}) != 1:
            raise ValueError("All STPs must have the same number of segments in x0.")
        if len({ope.num_segments for d in ds for ope in d.operands[1:-1]}) > 1:
            raise ValueError("All STPs must have the same number of segments in x1.")
        if len({d.operands[-1].num_segments for d in ds}) != 1:
            raise ValueError("All STPs must have the same number of segments in x2.")

        def f(d: cue.SegmentedTensorProduct) -> cue.SegmentedTensorProduct:
            d = d.move_operand(0, -2)
            d = d.flatten_coefficient_modes(force=True)
            d = d.flatten_modes(
                [
                    m
                    for m in d.subscripts.modes()
                    if not all(m in ss for ss in d.subscripts.operands)
                ]
            )
            d = d.consolidate_modes()
            if d.subscripts.modes() == []:
                d = d.append_modes_to_all_operands("u", dict(u=1))

            # ops.SymmetricTensorContraction will "symmetrize" for the derivatives so we can sort for the forward pass
            d = d.sort_indices_for_identical_operands(range(0, d.num_operands - 2))

            if len(d.subscripts.modes()) != 1:
                raise NotImplementedError("Different modes are not supported.")

            m = d.subscripts.modes()[0]

            if not all(ss == m for ss in d.subscripts.operands):
                raise NotImplementedError("Different subscripts are not supported.")

            d = d.split_mode(m, math.gcd(*d.get_dims(m)))

            return d

        ds_ = [f(d) for d in ds]
        import cuequivariance_ops_torch as ops

        d_max = max(ds_, key=lambda d: d.num_operands)

        path_segment_indices = sum((d.indices.tolist() for d in ds_), [])
        path_coefficients = sum((d.stacked_coefficients.tolist() for d in ds_), [])
        num_in_segments = (
            d_max.operands[0].num_segments if d_max.num_operands >= 3 else 1
        )
        num_couplings = d_max.operands[-2].num_segments
        num_out_segments = d_max.operands[-1].num_segments
        correlation = max(1, max_degree)
        math_dtype = math_dtype
        logger.debug(f"""cuequivariance_ops_torch.SymmetricTensorContraction(
    path_segment_indices={path_segment_indices},
    path_coefficients={path_coefficients},
    num_in_segments={num_in_segments},
    num_couplings={num_couplings},
    num_out_segments={num_out_segments},
    correlation={correlation},
    math_dtype={math_dtype},
        )""")

        self.f = ops.SymmetricTensorContraction(
            path_segment_indices=path_segment_indices,
            path_coefficients=path_coefficients,
            num_in_segments=num_in_segments,
            num_couplings=num_couplings,
            num_out_segments=num_out_segments,
            correlation=correlation,
            math_dtype=math_dtype,
        ).to(device=device)
        self.u = d_max.operands[0].size // d_max.operands[0].num_segments
        self.descriptors = ds_

    def forward(
        self, x0: torch.Tensor, i0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        r"""
        .. math::

            x_2[j_{n+1}] = val x_0[i_0][j_0] \prod_{k=1}^{n} x_1[j_k]

        """
        torch._assert(x0.ndim == 2, f"Expected shape (num_x0, x0_size), got {x0.shape}")
        torch._assert(x1.ndim == 2, f"Expected shape (batch, x1_size), got {x1.shape}")
        torch._assert(i0.ndim == 1, f"Expected shape (batch,), got {i0.shape}")

        i0 = i0.to(torch.int32)
        x0 = x0.reshape(x0.shape[0], x0.shape[1] // self.u, self.u)
        x1 = x1.reshape(x1.shape[0], x1.shape[1] // self.u, self.u)
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            logger.debug(
                f"Calling SymmetricTensorContraction: {self.descriptors}, input shapes: {x0.shape}, {i0.shape}, {x1.shape}"
            )
        out: torch.Tensor = self.f(x1, x0, i0)
        out = out.reshape(out.shape[0], out.shape[1] * self.u)
        return out


class FallbackImpl(torch.nn.Module):
    def __init__(
        self,
        stps: list[cue.SegmentedTensorProduct],
        device: Optional[torch.device],
        math_dtype: Optional[torch.dtype],
    ):
        super().__init__()
        self.fs = torch.nn.ModuleList(
            [
                cuet.TensorProduct(
                    d, device=device, math_dtype=math_dtype, use_fallback=True
                )
                for d in stps
            ]
        )

    def forward(
        self, x0: torch.Tensor, i0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        outs: List[torch.Tensor] = []

        for f in self.fs:
            if f.num_operands == 8:
                outs.append(f(x0[i0], x1, x1, x1, x1, x1, x1))
            elif f.num_operands == 7:
                outs.append(f(x0[i0], x1, x1, x1, x1, x1))
            elif f.num_operands == 6:
                outs.append(f(x0[i0], x1, x1, x1, x1))
            elif f.num_operands == 5:
                outs.append(f(x0[i0], x1, x1, x1))
            elif f.num_operands == 4:
                outs.append(f(x0[i0], x1, x1))
            elif f.num_operands == 3:
                outs.append(f(x0[i0], x1))
            else:
                outs.append(f(x0[i0]))

        return torch.sum(torch.stack(outs), dim=0)
