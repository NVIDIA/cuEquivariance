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

import logging
import math
from functools import partial
from typing import Dict, List, Optional, OrderedDict

import torch
import torch.fx
import torch.nn as nn

import cuequivariance as cue

logger = logging.getLogger(__name__)


def prod(numbers: List[int]):
    """
    This method is a workaround for script() not recognizing math.prod()
    """
    if torch.jit.is_scripting():
        product = 1
        for num in numbers:
            product *= num
        return product
    else:
        return math.prod(numbers)


def to_notypeconv(t, *args, **kwargs):
    new_kwargs = kwargs.copy()
    new_kwargs.pop("dtype", None)
    new_args = [None if isinstance(a, torch.dtype) else a for a in args]
    result = t.__original_to(*new_args, **new_kwargs)
    return result


def disable_type_conv(t):
    """
    This modifier can be used on Tensors or whole Modules
    to prevent them from being modified during to(dtype=x) calls
    """
    t.__original_to = t.to
    t.to = partial(to_notypeconv, t)
    return t


def _tensor_product_fx(
    descriptor: cue.SegmentedTensorProduct,
    device: Optional[torch.device],
    math_dtype: Optional[torch.dtype],
    optimize_einsums: bool,
) -> torch.nn.Module:
    """
    batch support of this function:
    - at least one input operand should have a batch dimension (ndim=2)
    - the output operand will have a batch dimension (ndim=2)
    """
    descriptor = descriptor.remove_zero_paths()
    descriptor = descriptor.remove_empty_segments()

    num_inputs = descriptor.num_operands - 1

    if num_inputs > 0 and descriptor.num_paths > 0:
        graph = torch.fx.Graph()
        tracer = torch.fx.proxy.GraphAppendingTracer(graph)
        constants = OrderedDict()

        inputs = [
            torch.fx.Proxy(graph.placeholder(f"input_{i}"), tracer)
            for i in range(num_inputs)
        ]

        operand_subscripts = [f"Z{ss}" for ss in descriptor.subscripts.operands]

        formula = (
            ",".join([descriptor.coefficient_subscripts] + operand_subscripts[:-1])
            + "->"
            + operand_subscripts[-1]
        )
        slices = [ope.segment_slices() for ope in descriptor.operands]

        outputs = []
        for path_idx, path in enumerate(descriptor.paths):
            segments = []
            for oid in range(num_inputs):
                seg_shape = descriptor.get_segment_shape(oid, path)
                inp = inputs[oid][..., slices[oid][path.indices[oid]]]
                if len(seg_shape) > 0:
                    inp = inp.reshape(inputs[oid].shape[:-1] + seg_shape)
                else:
                    inp = inp.reshape(inputs[oid].shape[:-1])

                if math_dtype is not None:
                    segments.append(inp.to(dtype=math_dtype))
                else:
                    segments.append(inp)

            if math_dtype is not None:
                c_tensor = disable_type_conv(
                    torch.tensor(path.coefficients, dtype=math_dtype, device=device)
                )
            else:
                c_tensor = torch.tensor(
                    path.coefficients, dtype=torch.float64, device=device
                )
            constants[f"c{path_idx}"] = c_tensor

            c = torch.fx.Proxy(graph.get_attr(f"c{path_idx}"), tracer=tracer).clone()
            if math_dtype is None:
                c = c.to(inputs[0].dtype)
            out = torch.einsum(formula, c, *segments)
            if math_dtype is not None:
                out = out.to(inputs[0].dtype)

            seg_shape = descriptor.get_segment_shape(-1, path)
            outputs += [
                out.reshape(out.shape[: out.ndim - len(seg_shape)] + (prod(seg_shape),))
            ]

        if len(outputs) == 0:
            raise NotImplementedError("No FX implementation for empty paths")

        def _sum(tensors, *, shape=None, like=None):
            if len(tensors) == 0:
                return like.new_zeros(shape)
            out = tensors[0]
            for t in tensors[1:]:
                out = torch.add(out, t)
            return out

        batch_shape = outputs[0].shape[:-1]
        output = torch.cat(
            [
                _sum(
                    [
                        out
                        for out, path in zip(outputs, descriptor.paths)
                        if path.indices[-1] == i
                    ],
                    shape=batch_shape + (prod(descriptor.operands[-1][i]),),
                    like=outputs[0],
                )
                for i in range(descriptor.operands[-1].num_segments)
            ],
            dim=-1,
        )

        graph.output(output.node)

        graph.lint()
        constants_root = torch.nn.Module()
        for key, value in constants.items():
            constants_root.register_buffer(key, value)
        graphmod = torch.fx.GraphModule(constants_root, graph)

        if optimize_einsums:
            try:
                import opt_einsum_fx
            except ImportError:
                logger.warning(
                    "opt_einsum_fx not available.\n"
                    "To use the optimization, please install opt_einsum_fx.\n"
                    "pip install opt_einsum_fx"
                )
            else:
                example_inputs = [
                    torch.zeros((10, operand.size))
                    for operand in descriptor.operands[:num_inputs]
                ]
                graphmod = opt_einsum_fx.optimize_einsums_full(graphmod, example_inputs)
    elif num_inputs == 0:

        class _no_input(torch.nn.Module):
            def __init__(self, descriptor: cue.SegmentedTensorProduct):
                super().__init__()

                for pid, path in enumerate(descriptor.paths):
                    if math_dtype is not None:
                        self.register_buffer(
                            f"c{pid}",
                            torch.tensor(
                                path.coefficients, dtype=math_dtype, device=device
                            ),
                        )
                    else:
                        self.register_buffer(
                            f"c{pid}",
                            torch.tensor(
                                path.coefficients, dtype=torch.float64, device=device
                            ),
                        )

            def forward(self):
                if math_dtype is not None:
                    output = torch.zeros(
                        (descriptor.operands[-1].size,),
                        device=self.c0.device,
                        dtype=math_dtype,
                    )
                else:
                    output = torch.zeros(
                        (descriptor.operands[-1].size,),
                        device=self.c0.device,
                        dtype=self.c0.dtype,
                    )
                for pid in range(descriptor.num_paths):
                    output[descriptor.paths[pid].indices[-1]] += torch.einsum(
                        descriptor.coefficient_subscripts
                        + "->"
                        + descriptor.subscripts.operands[-1],
                        getattr(self, f"c{pid}"),
                    )
                return output

        graphmod = _no_input(descriptor)

    else:
        raise NotImplementedError(
            "No FX implementation for empty paths and non-empty inputs"
        )

    return graphmod


class graph_inputs_base(nn.Module):
    def __init__(self, graph: torch.fx.GraphModule, repr: str):
        super().__init__()
        self.graph = graph
        self.repr = repr

    def __repr__(self):
        return self.repr


# Single classes for each number of inputs for scripting purposes
class graph_inputs0(graph_inputs_base):
    def forward(self, inputs: List[torch.Tensor]):
        return self.graph()


class graph_inputs1(graph_inputs_base):
    def forward(self, inputs: List[torch.Tensor]):
        return self.graph(inputs[0])


class graph_inputs2(graph_inputs_base):
    def forward(self, inputs: List[torch.Tensor]):
        return self.graph(inputs[0], inputs[1])


class graph_inputs3(graph_inputs_base):
    def forward(self, inputs: List[torch.Tensor]):
        return self.graph(inputs[0], inputs[1], inputs[2])


class graph_inputs4(graph_inputs_base):
    def forward(self, inputs: List[torch.Tensor]):
        return self.graph(inputs[0], inputs[1], inputs[2], inputs[3])


class graph_inputs5(graph_inputs_base):
    def forward(self, inputs: List[torch.Tensor]):
        return self.graph(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])


class graph_inputs6(graph_inputs_base):
    def forward(self, inputs: List[torch.Tensor]):
        return self.graph(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
        )


GRAPH_INPUTS = [
    graph_inputs0,
    graph_inputs1,
    graph_inputs2,
    graph_inputs3,
    graph_inputs4,
    graph_inputs5,
    graph_inputs6,
]


class SegmentedPolynomialNaive(nn.Module):
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
        if type(math_dtype) is str:
            try:
                math_dtype = getattr(torch, math_dtype)
            except AttributeError:
                raise ValueError(
                    f"Math_dtype {math_dtype} is not accepted for method `naive`."
                )
        self.math_dtype = math_dtype
        self.out_size = [o.size for o in polynomial.outputs]
        default_dtype_map = [
            0 if polynomial.num_inputs >= 1 else -1
        ] * polynomial.num_outputs
        self.dtypes = list(range(self.num_inputs)) + (
            default_dtype_map if output_dtype_map is None else output_dtype_map
        )

        # Build the graph
        self.graphs = torch.nn.ModuleList()
        self.b_outs = []
        self.input_inds = []
        for operation, d in polynomial.operations:
            ope_out, b_out = operation.output_operand_buffer(self.num_inputs)
            self.b_outs.append(b_out - self.num_inputs)
            self.input_inds.append(operation.input_buffers(self.num_inputs))
            gr = _tensor_product_fx(
                d.move_operand_last(ope_out),
                device=None,
                math_dtype=self.math_dtype,
                optimize_einsums=True,
            )
            if len(self.input_inds[-1]) < 6:
                self.graphs.append(
                    GRAPH_INPUTS[len(self.input_inds[-1])](gr, d.__repr__())
                )
            else:
                raise ValueError(
                    f"Unsupported number of inputs: {len(self.input_inds[-1])}"
                )

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
        out_buffers = [torch.empty(0) for _ in range(self.num_outputs)]

        # Apply TPs
        for i, graph in enumerate(self.graphs):
            b_out = self.b_outs[i]
            input_list = [inputs[j] for j in self.input_inds[i]]
            out = graph(input_list)
            # For 0 inputs case:
            if len(input_list) == 0:
                out = out.unsqueeze(0).to(inputs[0].dtype)
            if out_indices[b_out].size() != torch.Size([0]):
                # In case we need to replicate before scattering:
                if out.shape[0] == 1 and out_indices[b_out].shape[0] > 1:
                    out = out.expand(out_indices[b_out].shape[0], out.shape[1])
                inds = out_indices[b_out].unsqueeze(-1).expand_as(out)
                tmp_out = torch.zeros(
                    outputs_dims[b_out],
                    dtype=inputs[self.dtypes[b_out]].dtype,
                    device=inputs[0].device,
                ).scatter_add_(0, inds, out)
            else:
                if outputs_dims[b_out][0] == out.shape[0]:
                    tmp_out = out
                elif outputs_dims[b_out][0] == 1:
                    tmp_out = out.sum(dim=0, keepdim=True)
                elif out.shape[0] == 1:
                    tmp_out = out.expand(outputs_dims[b_out]).clone()
                else:
                    raise ValueError(
                        f"Input/output batch size mismatch {outputs_dims[b_out]} vs {out.shape}."
                    )
            if out_buffers[b_out].size() == torch.Size([0]):
                out_buffers[b_out] = tmp_out
            else:
                out_buffers[b_out] += tmp_out

        return out_buffers
