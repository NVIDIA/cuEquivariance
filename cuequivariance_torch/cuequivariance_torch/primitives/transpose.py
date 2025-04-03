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
import torch.fx

import cuequivariance as cue


class TransposeIrrepsLayout(torch.nn.Module):
    """Transpose the irreps layout of a tensor.

    Args:
        irreps (Irreps): The irreps of the tensor.
        source (IrrepsLayout): The source layout.
        target (IrrepsLayout): The target layout.
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        *,
        source: cue.IrrepsLayout,
        target: cue.IrrepsLayout,
        device: Optional[torch.device] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()

        if (source, target) == (cue.mul_ir, cue.ir_mul):
            self.f = TransposeSegments(
                [(mul, ir.dim) for mul, ir in irreps],
                device=device,
                use_fallback=use_fallback,
            )
        elif (source, target) == (cue.ir_mul, cue.mul_ir):
            self.f = TransposeSegments(
                [(ir.dim, mul) for mul, ir in irreps],
                device=device,
                use_fallback=use_fallback,
            )
        else:
            self.f = torch.nn.Identity()

        self.source, self.target = source, target

    # def extra_repr(self) -> str:
    #     return f"{self.source} -> {self.target}"

    def __repr__(self):
        return f"TransposeIrrepsLayout({self.source} -> {self.target})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Perform the transposition.

        Args:
            x (torch.Tensor): The input tensor.
            use_fallback (bool, optional): If `None` (default), a CUDA kernel will be used if available.
                If `False`, a CUDA kernel will be used, and an exception is raised if it's not available.
                If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Returns:
            torch.Tensor: The transposed tensor.
        """

        return self.f(x)


class TransposeSegments(torch.nn.Module):
    def __init__(
        self,
        segments: list[tuple[int, int]],
        device: Optional[torch.device] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()

        info = _transpose_info(segments, device=device)
        self.f = None

        if info is not None:
            import_error = None
            if use_fallback is False or use_fallback is None:
                try:
                    import cuequivariance_ops_torch  # noqa: F401
                except ImportError as e:
                    import_error = e
                else:
                    if torch.cuda.is_available():
                        self.f = _transpose(info).to(device=device)

            if use_fallback is False and self.f is None:
                raise RuntimeError(
                    f"CUDA kernel not available for TransposeSegments: {import_error}"
                )

            if self.f is None:
                self.f = _transpose_segments_fx(segments).to(device=device)
        else:
            self.f = torch.nn.Identity()

    def __repr__(self):
        return "TransposeSegments()"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.f(x)


def _transpose_segments_fx(segments: list[tuple[int, int]]) -> torch.nn.Module:
    graph = torch.fx.Graph()
    tracer = torch.fx.proxy.GraphAppendingTracer(graph)
    x = torch.fx.Proxy(graph.placeholder("input"), tracer)
    outputs = []

    source = cue.SegmentedOperand(ndim=2, segments=segments)
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


try:
    from cuequivariance_ops_torch import segmented_transpose
except ImportError:
    pass


class _transpose(torch.nn.Module):
    def __init__(self, info: torch.IntTensor):
        super().__init__()
        self.register_buffer("_info", info, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return segmented_transpose(x, self._info, True)
