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
import torch.fx

import cuequivariance as cue


class TransposeIrrepsLayout(torch.nn.Module):
    """Transpose the irreps layout of a tensor.

    Parameters
    ----------
    irreps : Irreps
        The irreps of the tensor.
    source : IrrepsLayout
        The source layout.
    target : IrrepsLayout
        The target layout.
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        source: cue.IrrepsLayout,
        target: cue.IrrepsLayout,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if (source, target) == (cue.mul_ir, cue.ir_mul):
            self.f = TransposeSegments(
                [(mul, ir.dim) for mul, ir in irreps], device=device
            )
        elif (source, target) == (cue.ir_mul, cue.mul_ir):
            self.f = TransposeSegments(
                [(ir.dim, mul) for mul, ir in irreps], device=device
            )
        else:
            self.f = _Identity()

        self.source, self.target = source, target

    # def extra_repr(self) -> str:
    #     return f"{self.source} -> {self.target}"

    def __repr__(self):
        return f"TransposeIrrepsLayout({self.source} -> {self.target})"

    def forward(
        self, x: torch.Tensor, *, use_fallback: Optional[bool] = None
    ) -> torch.Tensor:
        r"""
        Perform the transposition.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        use_fallback : Optional[bool], optional
            If `None` (default), a CUDA kernel will be used if available.
            If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
            If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Returns
        -------
        torch.Tensor
            The transposed tensor.
        """

        return self.f(x, use_fallback=use_fallback)


class _Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs):
        return x


class TransposeSegments(torch.nn.Module):

    def __init__(
        self, segments: list[tuple[int, int]], device: Optional[torch.device] = None
    ):
        super().__init__()

        info = _transpose_info(segments, device=device)

        if info is not None:
            try:
                import cuequivariance_ops_torch
            except ImportError:
                self.f_cuda = None
            else:
                self.f_cuda = _transpose(info).to(device=device)

            self.f = _transpose_segments_fx(segments).to(device=device)
        else:
            self.f_cuda = torch.nn.Identity()
            self.f = torch.nn.Identity()

    def __repr__(self):
        return f"TransposeSegments()"

    def forward(
        self, x: torch.Tensor, *, use_fallback: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Perform the transposition of the input tensor using either a CUDA kernel or a PyTorch fallback.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be transposed.
        use_fallback : Optional[bool], optional
            If `None` (default), a CUDA kernel will be used if available.
            If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
            If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Returns
        -------
        torch.Tensor
            The transposed tensor.

        Raises
        ------
        RuntimeError
            If `use_fallback` is `False` and a CUDA kernel is not available or the input is not on CUDA.
        """
        if (
            x.device.type == "cuda"
            and self.f_cuda is not None
            and (use_fallback is not True)
        ):
            return self.f_cuda(x)

        if use_fallback is False:
            if self.f_cuda is not None:
                raise RuntimeError("CUDA kernel available but input is not on CUDA")
            else:
                raise RuntimeError("No CUDA kernel available")

        return self.f(x)


def _transpose_segments_fx(segments: list[tuple[int, int]]) -> torch.nn.Module:
    graph = torch.fx.Graph()
    tracer = torch.fx.proxy.GraphAppendingTracer(graph)
    x = torch.fx.Proxy(graph.placeholder("input"), tracer)
    outputs = []

    source = cue.segmented_tensor_product.Operand(subscripts="ij", segments=segments)
    for sl, (u, v) in zip(source.segment_slices(), source.segments):
        outputs += [
            x[..., sl]
            .reshape(x.shape[:-1] + (u, v))
            .transpose(-2, -1)
            .reshape(x.shape[:-1] + (v * u,))
        ]
    output = torch.cat(outputs, dim=-1)
    graph.output(output.node)
    graph.lint()
    graphmod = torch.fx.GraphModule(torch.nn.Module(), graph)
    return graphmod


def _transpose_info(
    segments: list[tuple[int, int]], device
) -> Optional[torch.IntTensor]:
    info = []
    offset = 0
    is_trivial = True
    for u, v in segments:
        info.append([offset, u, v, -1])
        offset += u * v
        is_trivial = is_trivial and (u == 1 or v == 1)

    if is_trivial:
        return None
    return torch.IntTensor(info).to(device=device)


class _transpose(torch.nn.Module):
    def __init__(self, info: torch.IntTensor):
        super().__init__()
        self.register_buffer("_info", info, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from cuequivariance_ops_torch import segmented_transpose

        return segmented_transpose(x, self._info, True)
