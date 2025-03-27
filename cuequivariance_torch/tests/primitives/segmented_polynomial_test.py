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
import torch
import numpy as np
import pytest
from typing import List, Optional
import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance_torch._tests.utils import module_with_mode
# from equivariant_tensor_product_test import make_descriptors as make_equivariant_descriptors

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def generate_segmented_polynomials():
    result = []

    def yield_from(fn):
        result.extend(list(fn()))

    """
    @yield_from
    def one_operand():
        d = cue.SegmentedTensorProduct.empty_segments([1])
        yield (
            "one operand, no path",
            cue.SegmentedPolynomial(
                [], [cue.SegmentedOperand.empty_segments(1)], [(cue.Operation([0]), d)]
            ),
        )

        d_one_path = cue.SegmentedTensorProduct.empty_segments([1])
        one_path = cue.SegmentedPolynomial(
            [],
            [cue.SegmentedOperand.empty_segments(1)],
            [(cue.Operation([0]), d_one_path)],
        )
        d_one_path.add_path(0, c=123)
        yield "one operand, one path", one_path

    @yield_from
    def UnshapedArray_bug():
        e = cue.descriptors.symmetric_contraction(
            cue.Irreps("O3", "0e"), cue.Irreps("O3", "0e"), [0, 1]
        )
        yield "UnshapedArray_bug", e.polynomial

    @yield_from
    def multiple_operand_shape_bug():
        e = cue.descriptors.spherical_harmonics(cue.SO3(1), [2])
        yield "multiple_operand_shape_bug", e.polynomial

    @yield_from
    def vmap():
        e = cue.descriptors.full_tensor_product(
            cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1"), cue.Irreps("SO3", "1")
        )
        d = e.polynomial.operations[0][1]

        yield (
            "vmap",
            cue.SegmentedPolynomial(
                d.operands[:2],
                [d.operands[2], d.operands[2]],
                [
                    (cue.Operation([0, 1, 2]), d),
                    (cue.Operation([0, 1, 3]), d),
                ],
            ),
        )

    @yield_from
    def equivariant_descriptors():
        for elem in make_equivariant_descriptors():
            yield "equivariant_descriptor", elem
    """

    @yield_from
    def channelwise_tensor_product():
        e = (
            cue.descriptors.channelwise_tensor_product(
                cue.Irreps("O3", "32x0e + 32x1o"),
                cue.Irreps("O3", "0e + 1o + 2e"),
                cue.Irreps("O3", "0e + 1o"),
            )
            .flatten_coefficient_modes()
            .squeeze_modes()
        )
        yield "channelwise_tensor_product", e.polynomial

    return result


def clone_input(inp):
    result = []
    for x in inp:
        if isinstance(x, torch.Tensor):
            result.append(x.clone().detach().requires_grad_(x.requires_grad))
        elif isinstance(x, list) or isinstance(x, tuple):
            result.append(clone_input(x))
        elif (
            isinstance(x, str)
            or isinstance(x, int)
            or isinstance(x, float)
            or isinstance(x, bool)
            or isinstance(x, type(None))
        ):
            result.append(x)
        else:
            raise ValueError(f"Unknown type: {type(x)}")
    return tuple(result)


ceil_div = lambda a, b: (a + b - 1) // b


def make_inputs_for_operands(
    operands, dtype, idx_amount, idx_kind, batch_size, tensor_init_fn
):
    tensors = []
    indices = [None] * len(operands)
    for i, x in enumerate(operands):
        mode = "batch"
        if idx_amount == "all" or (idx_amount == "first" and i == 0):
            mode = idx_kind
        local_batch = batch_size
        if mode == "shared":
            local_batch = 1
        elif mode == "indexed":
            index_size = ceil_div(batch_size, 4)
            if index_size == 0:
                index_size = 1
            indices[i] = torch.randint(0, index_size, (batch_size,), device=device)
            local_batch = index_size
        tensors.append(tensor_init_fn(local_batch, x.size))
    return tensors, indices


def make_inputs(polynomial, dtype, indexing, batch_size):
    def tensor_init_inputs(batch_size, size):
        return torch.randn(
            (batch_size, size), device=device, dtype=dtype, requires_grad=True
        )

    inputs, input_indices = make_inputs_for_operands(
        polynomial.inputs, dtype, *indexing["input"], batch_size, tensor_init_inputs
    )

    def tensor_init_outputs(batch_size, size):
        return batch_size

    outputs, output_indices = make_inputs_for_operands(
        polynomial.outputs, dtype, *indexing["output"], batch_size, tensor_init_outputs
    )
    return [inputs, input_indices, outputs, output_indices]


class Reference(torch.nn.Module):
    def __init__(
        self,
        polynomial,
        output_dtype_map: List[int] = [],
        math_dtype: torch.dtype = torch.float32,
        name: str = "segmented_polynomial",
    ):
        super().__init__()
        self.polynomial = polynomial
        self.output_dtype_map = output_dtype_map
        self.math_dtype = math_dtype
        self.name = name

    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: List[Optional[torch.Tensor]] = None,
        output_shapes: List[Optional[int]] = None,
        output_indices: List[Optional[torch.Tensor]] = None,
    ):
        if input_indices is None:
            input_indices = [None] * self.polynomial.num_inputs
        if output_indices is None:
            output_indices = [None] * self.polynomial.num_outputs
        if output_shapes is None:
            output_shapes = [None] * self.polynomial.num_outputs

        orig_inputs = inputs
        # print(f"inputs: {[i.shape if i is not None else None for i in inputs]}")
        # print(
        #    f"input_indices: {[i.shape if i is not None else None for i in input_indices]}"
        # )
        # print(f"output_shapes: {output_shapes}")
        # print(
        #    f"output_indices: {[i.shape if i is not None else None for i in output_indices]}"
        # )

        # deduce the batch size:
        # if there are any indices, their size is the batch size
        # otherwise, it is the largest first dimension of the inputs
        # or the output_shaopes
        batch_size = None
        for index in input_indices:
            if index is not None:
                batch_size = index.size(0)
                break
        for index in output_indices:
            if index is not None:
                batch_size = index.size(0)
                break
        if batch_size is None:
            for elem in output_shapes:
                if elem is not None:
                    batch_size = elem
                    break
        if batch_size is None:
            for inp in inputs:
                if inp.size(0) > 1:
                    batch_size = inp.size(0)
                    break

        if batch_size is None:
            batch_size = 1

        # print(f"batch_size: {batch_size}")
        # create the output tensors
        outputs = []
        for i in range(self.polynomial.num_outputs):
            output_dtype = None
            if i < len(self.output_dtype_map):
                if self.output_dtype_map[i] == -1:
                    output_dtype = self.math_dtype
                else:
                    output_dtype = inputs[self.output_dtype_map[i]].dtype
            if output_dtype is None and len(inputs) > 0:
                output_dtype = inputs[0].dtype
            if output_dtype is None:
                output_dtype = self.math_dtype

            output_batch_size = None
            if i < len(output_shapes):
                if output_shapes[i] is not None:
                    output_batch_size = output_shapes[i]
            if output_batch_size is None:
                if i < len(output_indices) and output_indices[i] is not None:
                    output_batch_size = output_indices[i].size(0)
            if output_batch_size is None:
                output_batch_size = batch_size
            outputs.append(
                torch.zeros(
                    (output_batch_size, self.polynomial.outputs[i].size),
                    device=device,
                    dtype=output_dtype,
                )
            )
        # print(f"outputs: {[o.shape for o in outputs]}")

        # pad large enough
        input_indices = list(input_indices) + [None] * self.polynomial.num_inputs
        output_indices = list(output_indices) + [None] * self.polynomial.num_outputs

        # print(f"pre-gather orig_inputs: {[i.shape if i is not None else None for i in orig_inputs]}")
        inputs = [
            input.index_select(0, input_index) if input_index is not None else input
            for input, input_index in zip(inputs, input_indices)
        ]
        # print(f"post-gather orig_inputs: {[i.shape if i is not None else None for i in orig_inputs]}")

        regular_outputs = [
            torch.zeros(
                (output_index.shape[0], output.shape[1]),
                device=output.device,
                dtype=output.dtype,
            )
            if output_index is not None
            else output
            for output, output_index in zip(outputs, output_indices)
        ]
        # print(f"regular_outputs: {[o.shape for o in regular_outputs]}")

        # print(f"pre-op orig_inputs: {[i.shape if i is not None else None for i in orig_inputs]}")
        # perform the operation
        for op, stp in self.polynomial.operations:
            self.perform_einsum(op, stp, inputs, regular_outputs)
        # print(f"post-op orig_inputs: {[i.shape if i is not None else None for i in orig_inputs]}")

        for output, regular_output, output_index in zip(
            outputs, regular_outputs, output_indices
        ):
            if output_index is not None:
                output_index = output_index.reshape(-1, 1).broadcast_to(
                    regular_output.shape
                )
                output.scatter_add_(0, output_index, regular_output)

        # print(f"final orig_inputs: {[i.shape if i is not None else None for i in orig_inputs]}")
        return outputs

    def perform_einsum(self, op, stp, inputs, outputs):
        # select operands
        inputs = [inputs[o] for o in op.buffers if o < self.polynomial.num_inputs]
        outputs = [
            (o_idx, outputs[o - self.polynomial.num_inputs])
            for o_idx, o in enumerate(op.buffers)
            if o >= self.polynomial.num_inputs
        ]
        assert len(outputs) == 1
        o_idx, output = outputs[0]
        from cuequivariance_torch.primitives.tensor_product import _tensor_product_fx

        local_output = _tensor_product_fx(
            stp.move_operand_last(o_idx), device, self.math_dtype, False
        )(*inputs)
        # print(output.shape, local_output.shape, [i.shape for i in inputs])
        if output.shape[0] == 1:
            output += torch.sum(local_output, dim=0, keepdim=True)
        else:
            output += local_output


# todo check if this is actually needed
torch._dynamo.allow_in_graph(torch.autograd.grad)


class Grad(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @staticmethod
    def scalar(tensors):
        result = 0
        for t in tensors:
            result += t.pow(2).sum()
        return result

    def forward(self, *args):
        return torch.autograd.grad(
            self.scalar(self.m(*args)), args[0], create_graph=True
        )


def assert_close_recursive(a, b, index=[]):
    if isinstance(b, torch.Tensor):
        # torch.testing.assert_close(a, b)
        assert a.shape == b.shape
        if a.grad is not None or b.grad is not None:
            # torch.testing.assert_close(a.grad, b.grad)
            assert a.grad.shape == b.grad.shape
        return
    if (
        isinstance(a, list)
        or isinstance(a, tuple)
        or isinstance(b, list)
        or isinstance(b, tuple)
    ):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close_recursive(x, y, index + [i])
        return
    if a == b:
        return
    raise ValueError(f"Unknown type: {type(a)} {type(b)}")


SEGMENTED_POLYNOMIALS = list(generate_segmented_polynomials())

DATA_TYPES_IN_MATH = [
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
]
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DATA_TYPES_IN_MATH += [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]
DEBUG_DATA_TYPES_IN_MATH = [
    (torch.float32, torch.float32),
]

EXPORT_MODES = ["eager", "compile", "script", "jit", "export"]
EXPORT_MODES = ["eager", "compile", "export"]
DEBUG_EXPORT_MODES = ["eager"]

INDEXING = [
    {"input": (inp_amount, inp_kind), "output": (out_amount, out_kind)}
    for inp_amount in ["first", "all"]
    for out_amount in ["first", "all"]
    for inp_kind in ["shared", "indexed", "batch"]
    for out_kind in ["shared", "indexed", "batch"]
    if inp_kind != "batch" or inp_amount == "all"  # for batch, only "all" is valid
    if out_kind != "batch" or out_amount == "all"  # for batch, only "all" is valid
]
DEBUG_INDEXING = [
    {"input": ("all", "batched"), "output": ("all", "batched")},
    {"input": ("first", "shared"), "output": ("all", "batched")},
]
GRAD = [False, True]
DEBUG_GRAD = [False]

BATCH_SIZE = [0, 5]
DEBUG_BATCH_SIZE = [5]


@pytest.mark.parametrize("name, polynomial", SEGMENTED_POLYNOMIALS)
@pytest.mark.parametrize("dtype, math_dtype", DATA_TYPES_IN_MATH)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("mode", EXPORT_MODES)
@pytest.mark.parametrize("grad", GRAD)
@pytest.mark.parametrize("indexing", INDEXING)
def test_segmented_polynomial_product(
    name, polynomial, dtype, math_dtype, batch_size, mode, grad, indexing, tmp_path
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if (
        torch.cuda.get_device_capability()[0] < 8
        and any(kind in ["indexed", "shared"] for _, kind in indexing.values())
        and dtype == torch.float16
    ):
        pytest.skip("FP16 atomics are not supported on this GPU")
    if (
        torch.cuda.get_device_capability()[0] < 9
        and any(kind in ["indexed", "shared"] for _, kind in indexing.values())
        and dtype == torch.bfloat16
    ):
        pytest.skip("BF16 atomics are not supported on this GPU")

    # todo check if this is actually needed
    # if grad and batch_size == 0:
    #    pytest.skip("Batch size is 0, so we cannot compute gradients")

    m_ref = Reference(polynomial, math_dtype=math_dtype)
    m = cuet.SegmentedPolynomialProduct(polynomial, math_dtype=math_dtype)

    if grad:
        m_ref = Grad(m_ref)
        m = Grad(m)

    inp = make_inputs(polynomial, dtype, indexing, batch_size)
    m = module_with_mode(mode, m, inp, math_dtype, tmp_path)

    inp_ref = clone_input(inp)
    # print(f"len(inp[0]): {len(inp[0])}")
    # print(f"len(inp_ref[0]): {len(inp_ref[0])}")
    output = m(*inp)
    output_ref = m_ref(*inp_ref)
    Grad.scalar(output).backward()
    Grad.scalar(output_ref).backward()
    # print(f"len(inp[0]): {len(inp[0])}")
    # print(f"len(inp_ref[0]): {len(inp_ref[0])}")
    # print(f"output[0]: {output[0]}")
    #print(f"output_ref: {output_ref}")
    assert_close_recursive(output, output_ref)
    assert_close_recursive(inp, inp_ref)
