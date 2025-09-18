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
from typing import Dict, List, Optional, Tuple

import torch
import torch.fx
import torch.nn as nn

import cuequivariance as cue


class IndexedLinear(nn.Module):
    def __init__(
        self,
        d: cue.SegmentedTensorProduct,
        subscripts: Tuple[str],
        dim_indices: List[List[int]],
        math_dtype: Optional[str] = None,
    ):
        super().__init__()
        self.repr = d.__repr__()
        self.subscripts = subscripts
        self.Zind = dim_indices[0]
        self.Cind = dim_indices[1]
        self.uind = dim_indices[2]
        self.vind = dim_indices[3]
        self.wind = dim_indices[4]
        self.reshape_inputs = dim_indices[5]
        self.math_dtype = math_dtype

        d = d.sort_paths(-1)
        pids = d.compressed_path_segment(-1)
        self.slices = [operand.segment_slices() for operand in d.operands]
        # Assuming all paths have the same 2 inputs
        self.indices = [
            [[path.indices[0], path.indices[1]] for path in d.paths[pid_start:pid_end]]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]
        self.segment_shapes = [
            [
                [d.get_segment_shape(0, path), d.get_segment_shape(1, path)]
                for path in d.paths[pid_start:pid_end]
            ]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]
        coefficients = []
        self.coefficients_indices = []
        n = 0
        for pid_start, pid_end in zip(pids[:-1], pids[1:]):
            coeffs = []
            for path in d.paths[pid_start:pid_end]:
                coefficients.append(path.coefficients.item())
                coeffs.append(n)
                n += 1
            self.coefficients_indices.append(coeffs)
        self.register_buffer("coefficients", torch.tensor(coefficients))

    def __repr__(self):
        return self.repr

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        out_shape: torch.Size,
        counts: torch.Tensor,
    ):
        shape_list = [input1.shape, input2.shape, out_shape, [1]]
        Z = shape_list[self.Zind[0]][self.Zind[1]]
        C = shape_list[self.Cind[0]][self.Cind[1]]
        outputs = [
            [
                apply_linear(
                    input1[
                        :, self.slices[0][i[0]].start : self.slices[0][i[0]].stop
                    ].reshape((input1.shape[0],) + shape[0]),
                    input2[
                        :, self.slices[1][i[1]].start : self.slices[1][i[1]].stop
                    ].reshape((input2.shape[0],) + shape[1]),
                    Z,
                    C,
                    self.uind,
                    self.vind,
                    self.wind,
                    self.reshape_inputs,
                    counts,
                    self.subscripts,
                    self.coefficients[coef],
                    self.math_dtype,
                )
                for i, shape, coef in zip(ii, shapes, coefs)
            ]
            for ii, shapes, coefs in zip(
                self.indices, self.segment_shapes, self.coefficients_indices
            )
        ]
        return torch.cat(
            [o[0].reshape(out_shape[0], -1) for o in outputs], dim=-1
        )  # TODO: More general case?


def apply_linear(
    input1: torch.Tensor,
    input2: torch.Tensor,
    Z: int,
    C: int,
    uind: List[int],
    vind: List[int],
    wind: List[int],
    reshape_inputs: List[int],
    counts: torch.Tensor,
    subscripts: Tuple[str],
    coefficients: torch.Tensor,
    math_dtype: Optional[str] = None,
):
    from cuequivariance_ops_torch import indexed_linear

    shape_list = [input1.shape, input2.shape, [], [1]]
    u = shape_list[uind[0]][uind[1]]
    v = shape_list[vind[0]][vind[1]]
    w = shape_list[wind[0]][wind[1]]
    outsh = None
    if reshape_inputs[0]:
        outsh = input1.shape[:-1]
        input1 = input1.reshape(-1, input1.shape[-1])
    if reshape_inputs[1]:
        outsh = input2.shape[:-1]
        input2 = input2.reshape(-1, input2.shape[-1])
    output = indexed_linear(
        input1,
        input2,
        counts * w,
        u,
        v,
        C,
        Z * w,
        subscripts,
        coefficients,  # math_dtype
    )
    if outsh is not None:
        output = output.reshape(outsh + output.shape[1:])
    return output


class SegmentedPolynomialIndexedLinear(nn.Module):
    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        math_dtype: Optional[str] = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
    ):
        super().__init__()
        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.input_sizes = [o.size for o in polynomial.inputs]
        self.name = name
        if math_dtype is not None:
            warnings.warn(
                "`indexed_linear` does not support explicit `math_dtype`."
                "This will be ignored."
            )
        self.out_size = [o.size for o in polynomial.outputs]
        default_dtype_map = [
            0 if polynomial.num_inputs >= 1 else -1
        ] * polynomial.num_outputs
        self.dtypes = list(range(self.num_inputs)) + (
            default_dtype_map if output_dtype_map is None else output_dtype_map
        )

        # Build the TPs
        self.tps = torch.nn.ModuleList()
        self.b_outs = []
        self.input_inds = []
        self.indexed_input = [None for _ in range(self.num_inputs + self.num_outputs)]
        for operation, d in polynomial.operations:
            ope_out, b_out = operation.output_operand_buffer(self.num_inputs)
            self.b_outs.append(b_out - self.num_inputs)
            self.input_inds.append(operation.input_buffers(self.num_inputs))
            d = d.move_operand_last(ope_out)
            subscripts = d.canonicalize_subscripts().subscripts

            if subscripts not in SUBDICT.keys():
                raise NotImplementedError(
                    f"Indexed_linear does not support the operation {subscripts}."
                )

            signature = SUBDICT[subscripts]
            # Each input has to be either indexed or not
            C_index = signature[2][0]
            if C_index < 2:
                C_index = self.input_inds[-1][C_index]
            else:
                C_index = b_out
            if self.indexed_input[C_index] is None:
                self.indexed_input[C_index] = True
            else:
                assert self.indexed_input[C_index], (
                    f"Buffer {C_index} has multiple indexed values."
                )
            Z_index = signature[1][0]
            if Z_index < 2:
                Z_index = self.input_inds[-1][Z_index]
            else:
                Z_index = b_out
            if self.indexed_input[Z_index] is None:
                self.indexed_input[Z_index] = False
            else:
                assert not self.indexed_input[Z_index], (
                    f"Buffer {Z_index} has multiple indexed values."
                )
            self.tps.append(IndexedLinear(d, signature[0], signature[1:], math_dtype))

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

        # Input indexing
        count_indices = {}
        for k, v in input_indices.items():
            if not self.indexed_input[k]:
                inputs[k] = inputs[k][v]
            else:
                assert torch.all(v[1:] - v[:-1] >= 0), (
                    "Indexed_linear does not support non-sorted indices."
                )
                count_indices[k] = torch.bincount(v, minlength=inputs[k].shape[0]).int()

        # Output indices:
        out_indices = [torch.empty(0) for _ in range(self.num_outputs)]
        for k, v in output_indices.items():
            assert k in output_shapes.keys(), (
                "output shapes must be provided for output indices"
            )
            if not self.indexed_input[k + self.num_inputs]:
                out_indices[k] = v
            else:
                assert torch.all(v[1:] - v[:-1] >= 0), (
                    "Indexed_linear does not support non-sorted indices."
                )
                out_indices[k] = torch.bincount(
                    v, minlength=output_shapes[k].shape[0]
                ).int()

        input_batch_sizes = [t.shape[0] for t in inputs]
        batch_size = 1
        for k, size in enumerate(input_batch_sizes):
            if not self.indexed_input[k]:
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
            if tp.Cind[0] < 2:
                counts = count_indices[self.input_inds[i][tp.Cind[0]]]
            else:
                counts = out_indices[b_out]
            out = tp(input_list[0], input_list[1], outputs_dims[b_out], counts)
            if self.indexed_input[b_out + self.num_inputs]:
                out = out.reshape(out_indices[b_out].shape[0], -1)
            elif out_indices[b_out].size() != torch.Size([0]):
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


# Dict of subscripts to
# [canonicalized subscripts, indices of Z, C, u, v, w, whether to reshape each input]
# indices in a list [input1.shape, input2.shape, output.shape, [1]]
SUBDICT = {
    "uv,u,v": [("uv", "u", "v"), [1, 0], [0, 0], [0, 1], [0, 2], [3, 0], [0, 0]],
    "uv,v,u": [("uv", "v", "u"), [1, 0], [0, 0], [0, 1], [0, 2], [3, 0], [0, 0]],
    "uv,wu,wv": [("uv", "u", "v"), [1, 0], [0, 0], [0, 1], [0, 2], [1, 1], [0, 1]],
    "uv,wv,wu": [("uv", "v", "u"), [1, 0], [0, 0], [0, 1], [0, 2], [1, 1], [0, 1]],
    "u,uv,v": [("u", "uv", "v"), [0, 0], [1, 0], [0, 1], [1, 2], [3, 0], [0, 0]],
    "u,vu,v": [
        ("u", "vu", "v"),
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [3, 0],
        [0, 0],
    ],
    "uv,wv,uw": [
        ("u", "vu", "v"),
        [0, 0],
        [1, 0],
        [0, 2],
        [1, 1],
        [0, 1],
        [1, 0],
    ],
    "uv,vw,uw": [
        ("u", "uv", "v"),
        [0, 0],
        [1, 0],
        [0, 2],
        [1, 2],
        [0, 1],
        [1, 0],
    ],
    "u,v,vu": [
        ("u", "v", "vu"),
        [0, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [3, 0],
        [0, 0],
    ],
    "u,v,uv": [
        ("u", "v", "uv"),
        [0, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [3, 0],
        [0, 0],
    ],
    "uv,uw,wv": [
        ("u", "v", "vu"),
        [0, 0],
        [2, 0],
        [0, 2],
        [1, 2],
        [0, 1],
        [1, 1],
    ],
    "uv,uw,vw": [
        ("u", "v", "uv"),
        [0, 0],
        [2, 0],
        [0, 2],
        [1, 2],
        [0, 1],
        [1, 1],
    ],
}
