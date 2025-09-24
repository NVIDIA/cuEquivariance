# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Dict, List, Optional

import torch
import torch.fx
import torch.nn as nn

import cuequivariance as cue


# Single classes for each number of inputs for scripting purposes
class FusedTP3(nn.Module):
    def __init__(self, d: cue.SegmentedTensorProduct, math_dtype: torch.dtype):
        super().__init__()
        import cuequivariance_ops_torch as ops

        self.tp = ops.FusedTensorProductOp3(
            operand_segment_modes=d.subscripts.operands,
            operand_segment_offsets=[
                [s.start for s in ope.segment_slices()] for ope in d.operands
            ],
            operand_segment_shapes=[ope.segments for ope in d.operands],
            path_indices=d.indices,
            path_coefficients=d.stacked_coefficients,
            math_dtype=math_dtype,
        )
        self.repr = d.__repr__()

    def __repr__(self):
        return self.repr

    def forward(self, inputs: List[torch.Tensor]):
        return self.tp(inputs[0], inputs[1])


class FusedTP4(nn.Module):
    def __init__(self, d: cue.SegmentedTensorProduct, math_dtype: torch.dtype):
        super().__init__()
        import cuequivariance_ops_torch as ops

        self.tp = ops.FusedTensorProductOp4(
            operand_segment_modes=d.subscripts.operands,
            operand_segment_offsets=[
                [s.start for s in ope.segment_slices()] for ope in d.operands
            ],
            operand_segment_shapes=[ope.segments for ope in d.operands],
            path_indices=d.indices,
            path_coefficients=d.stacked_coefficients,
            math_dtype=math_dtype,
        )
        self.repr = d.__repr__()

    def __repr__(self):
        return self.repr

    def forward(self, inputs: List[torch.Tensor]):
        return self.tp(inputs[0], inputs[1], inputs[2])


class SegmentedPolynomialFusedTP(nn.Module):
    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        math_dtype: Optional[str | torch.dtype] = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
    ):
        super().__init__()
        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.input_sizes = [o.size for o in polynomial.inputs]
        self.name = name
        if math_dtype is None:
            warnings.warn(
                "`math_dtype` is not provided for method `fused_tp`: using float32."
            )
            math_dtype = torch.float32
        if type(math_dtype) is str:
            try:
                math_dtype = getattr(torch, math_dtype)
            except AttributeError:
                raise ValueError(
                    f"Math_dtype {math_dtype} is not accepted for method `fused_tp`."
                )
        if math_dtype not in [torch.float32, torch.float64]:
            raise ValueError(
                "Fused TP only supports math_dtype==float32 or math_dtype==float64"
            )
        self.math_dtype = math_dtype
        self.out_size = [o.size for o in polynomial.outputs]
        default_dtype_map = [
            0 if polynomial.num_inputs >= 1 else -1
        ] * polynomial.num_outputs
        self.dtypes = list(range(self.num_inputs)) + (
            default_dtype_map if output_dtype_map is None else output_dtype_map
        )
        supported_targets = [
            cue.segmented_polynomials.Subscripts(subscripts)
            for subscripts in [
                "u__uw_w",
                "_v_vw_w",
                "u_v_uv_u",
                "u_v_uv_v",
                "u_u_uw_w",
                "u_v_uvw_w",
                "u_u_u",
                "u_v_uv",
                "u_uv_v",
                "u__u",
                "_v_v",
            ]
        ]

        # Build the TPs
        self.tps = torch.nn.ModuleList()
        self.b_outs = []
        self.input_inds = []
        for operation, d in polynomial.operations:
            ope_out, b_out = operation.output_operand_buffer(self.num_inputs)
            if d.num_paths == 0:
                raise NotImplementedError("Fused TP does not support empty paths.")
                # Should we fall back to naive just for this operation?

            if d.num_operands not in (3, 4):
                raise NotImplementedError(
                    f"Fused TP only supports 3 or 4 operands. Got {d.subscripts}."
                )

            try:
                d, perm = next(
                    cue.segmented_polynomials.dispatch(
                        d, supported_targets, "permute_all_but_last"
                    )
                )
            except StopIteration:
                raise NotImplementedError(
                    f"Fused TP does not support {d}."
                    " Supported targets are: "
                    + ", ".join(str(t) for t in supported_targets)
                )

            if d.num_operands == 3:
                self.tps.append(FusedTP3(d, self.math_dtype))
            elif d.num_operands == 4:
                self.tps.append(FusedTP4(d, self.math_dtype))

            self.input_inds.append(
                [perm[i] for i in operation.input_buffers(self.num_inputs)]
            )
            self.b_outs.append(b_out - self.num_inputs)

    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Dict[int, torch.Tensor],
        output_shapes: Dict[int, torch.Tensor],
        output_indices: Dict[int, torch.Tensor],
    ):
        for i, x, size in zip(range(self.num_inputs), inputs, self.input_sizes):
            if x.ndim == 0:
                raise ValueError(f"Input {i} has no dimensions")
            if x.shape[-1] != size:
                raise ValueError(
                    f"Input {i} has shape {x.shape} but expected shape {size}."
                )

        # This is not supported:
        if self.math_dtype == torch.float32 and any(
            t.dtype == torch.float64 for t in inputs
        ):
            raise ValueError(
                "Fused TP does not support float32 math_dtype with float64 inputs."
            )

        # Input indexing
        for k, v in input_indices.items():
            inputs[k] = inputs[k][v]

        # Output indices:
        out_indices = [torch.empty(0) for _ in range(self.num_outputs)]
        for k, v in output_indices.items():
            out_indices[k] = v
            assert k in output_shapes.keys(), (
                "output shapes must be provided for output indices"
            )

        input_batch_sizes = [t.shape[0] for t in inputs]
        batch_size = 1
        for size in input_batch_sizes:
            if size != 1:
                assert batch_size in [1, size]
            batch_size = size
        outputs_dims = [
            (batch_size, shape)
            if i not in output_shapes.keys()
            else (output_shapes[i].shape[0], shape)
            for i, shape in enumerate(self.out_size)
        ]
        out_buffers = [
            torch.zeros(
                out_shape, dtype=inputs[out_dtype_ind].dtype, device=inputs[0].device
            )
            for (out_shape, out_dtype_ind) in zip(outputs_dims, self.dtypes)
        ]

        # Apply TPs
        for i, tp in enumerate(self.tps):
            b_out = self.b_outs[i]
            input_list = [inputs[j] for j in self.input_inds[i]]
            out = tp(input_list)
            if out_indices[b_out].size() != torch.Size([0]):
                # In case we need to replicate before scattering:
                if out.shape[0] == 1 and out_indices[b_out].shape[0] > 1:
                    out = out.expand(out_indices[b_out].shape[0], out.shape[1])
                inds = out_indices[b_out].unsqueeze(-1).expand_as(out)
                out_buffers[b_out].scatter_add_(0, inds, out)
            else:
                if out_buffers[b_out].shape[0] == out.shape[0]:
                    out_buffers[b_out] += out
                elif out_buffers[b_out].shape[0] == 1:
                    out_buffers[b_out] += out.sum(dim=0)
                elif out.shape[0] == 1:
                    out_buffers[b_out] += out.expand_as(out_buffers[b_out])
                else:
                    raise ValueError(
                        f"Input/output batch size mismatch {out_buffers[b_out].shape} vs {out.shape}."
                    )

        return out_buffers
