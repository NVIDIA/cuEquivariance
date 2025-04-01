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
from typing import Dict, List, Optional

import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance_torch._tests.utils import module_with_mode, tol_dict

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def generate_segmented_polynomials():
    result = []

    def yield_from(fn):
        result.extend(list(fn()))

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

    @yield_from
    def symmetric_contraction():
        e = cue.descriptors.symmetric_contraction(
            32 * cue.Irreps("O3", "0e + 1o + 2e"),
            32 * cue.Irreps("O3", "0e + 1o"),
            [1, 2],
        )
        yield "symmetric_contraction", e.polynomial

    return result


def clone_input(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().requires_grad_(x.requires_grad)
    elif isinstance(x, list) or isinstance(x, tuple):
        return tuple([clone_input(y) for y in x])
    elif isinstance(x, dict):
        return {k: clone_input(v) for k, v in x.items()}
    elif (
        isinstance(x, str)
        or isinstance(x, int)
        or isinstance(x, float)
        or isinstance(x, bool)
        or isinstance(x, type(None))
    ):
        return x
    else:
        raise ValueError(f"Unknown type: {type(x)}")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_inputs_for_operands(
    operands, dtype, idx_amount, idx_kind, batch_size, tensor_init_fn
):
    tensors = []
    indices = {}
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
        return torch.empty(
            1, device=device, dtype=dtype, requires_grad=False
        ).broadcast_to(batch_size, size)

    outputs, output_indices = make_inputs_for_operands(
        polynomial.outputs, dtype, *indexing["output"], batch_size, tensor_init_outputs
    )
    outputs = {i: o for i, o in enumerate(outputs)}
    result = {"inputs": inputs}
    if input_indices:
        result["input_indices"] = input_indices
    if outputs:
        result["output_shapes"] = outputs
    if output_indices:
        result["output_indices"] = output_indices
    return result


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
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
    ):
        if input_indices is None:
            input_indices = {}
        if output_indices is None:
            output_indices = {}
        if output_shapes is None:
            output_shapes = {}

        # deduce the batch size:
        # if there are any indices, their size is the batch size
        # otherwise, it is the largest first dimension of the inputs
        # or the output_shaopes
        batch_size = None
        for index in input_indices.values():
            batch_size = index.size(0)
            break
        for index in output_indices.values():
            batch_size = index.size(0)
            break
        if batch_size is None:
            for elem in output_shapes.values():
                if elem.size(0) != 1:
                    batch_size = elem.size(0)
                    break
        if batch_size is None:
            for inp in inputs:
                if inp.size(0) != 1:
                    batch_size = inp.size(0)
                    break

        if batch_size is None:
            batch_size = 1

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
            if i in output_shapes:
                output_batch_size = output_shapes[i].size(0)
            if output_batch_size is None:
                if i in output_indices:
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

        inputs = [
            input.index_select(0, input_indices[idx]) if idx in input_indices else input
            for idx, input in enumerate(inputs)
        ]

        regular_outputs = [
            torch.zeros(
                (output_indices[idx].size(0), output.size(1)),
                device=output.device,
                dtype=output.dtype,
            )
            if idx in output_indices
            else output
            for idx, output in enumerate(outputs)
        ]

        # perform the operation
        for op, stp in self.polynomial.operations:
            self.perform_einsum(op, stp, inputs, regular_outputs)

        for idx, (output, regular_output) in enumerate(zip(outputs, regular_outputs)):
            if idx in output_indices:
                output_index = (
                    output_indices[idx]
                    .reshape(-1, 1)
                    .broadcast_to(regular_output.shape)
                )
                output.scatter_add_(0, output_index, regular_output)

        return outputs

    def perform_einsum(self, op, stp, inputs, outputs):
        # select operands
        inputs = [inputs[o] for o in op.buffers if o < self.polynomial.num_inputs]
        o_idx, o_buf = op.output_operand_buffer(self.polynomial.num_inputs)
        output = outputs[o_buf - self.polynomial.num_inputs]
        from cuequivariance_torch.primitives.tensor_product import _tensor_product_fx

        local_output = _tensor_product_fx(
            stp.move_operand_last(o_idx), device, self.math_dtype, False
        )(*inputs)
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
    def scalar(tensors: List[torch.Tensor]) -> torch.Tensor:
        result = tensors[0].pow(2).sum()
        for t in tensors[1:]:
            result += t.pow(2).sum()
        return result

    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
    ):
        return torch.autograd.grad(
            [self.scalar(self.m(inputs, input_indices, output_shapes, output_indices))],
            inputs,
            create_graph=True,
        )


def tol_dict_grad(tol_dict):
    return {"atol": 10 * tol_dict["atol"], "rtol": 10 * tol_dict["rtol"]}


def assert_close_recursive(a, b, tol_dict, index=[]):
    if isinstance(b, torch.Tensor):
        torch.testing.assert_close(a, b, **tol_dict, equal_nan=True)
        assert a.shape == b.shape
        assert a.requires_grad == b.requires_grad
        if a.requires_grad and (a.grad is not None or b.grad is not None):
            assert_close_recursive(
                a.grad, b.grad, tol_dict_grad(tol_dict), index + ["grad"]
            )
        return
    if (
        isinstance(a, list)
        or isinstance(a, tuple)
        or isinstance(b, list)
        or isinstance(b, tuple)
    ):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close_recursive(x, y, tol_dict, index + [i])
        return
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_close_recursive(a[k], b[k], tol_dict, index + [k])
        return
    if a == b:
        return
    raise ValueError(f"Unknown type: {type(a)} {type(b)}")


def run_segmented_polynomial_test(
    name,
    polynomial,
    dtype,
    math_dtype,
    batch_size,
    mode,
    grad,
    backward,
    indexing,
    tmp_path,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if grad and mode == "jit":
        pytest.skip("torch.jit.trace does not work with inline autograd")
    if grad and backward and dtype.itemsize <= 2:
        pytest.skip("double backward with fp16/bf16 lacks accuracy")

    m_ref = Reference(polynomial, math_dtype=math_dtype)
    m = cuet.SegmentedPolynomial(polynomial, math_dtype=math_dtype)

    test_tol_dict = tol_dict[(dtype, math_dtype)]

    if grad:
        m_ref = Grad(m_ref)
        m = Grad(m)
        test_tol_dict = tol_dict_grad(test_tol_dict)

    inp = make_inputs(polynomial, dtype, indexing, batch_size)
    m = module_with_mode(mode, m, inp, math_dtype, tmp_path)

    inp_ref = clone_input(inp)

    output = m(**inp)
    output_ref = m_ref(**inp_ref)

    if backward:
        Grad.scalar(output).backward()
        Grad.scalar(output_ref).backward()

    assert_close_recursive(output, output_ref, test_tol_dict)
    assert_close_recursive(inp, inp_ref, test_tol_dict)


SEGMENTED_POLYNOMIALS = list(generate_segmented_polynomials())

DATA_TYPES_IN_MATH = [
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
]

EXPORT_MODES = ["eager", "compile", "script", "jit", "export"]

ALL_INDEXING = [
    {"input": (inp_amount, inp_kind), "output": (out_amount, out_kind)}
    for inp_amount in ["first", "all"]
    for out_amount in ["first", "all"]
    for inp_kind in ["shared", "indexed", "batch"]
    for out_kind in ["shared", "indexed", "batch"]
    if inp_kind != "batch" or inp_amount == "all"  # for batch, only "all" is valid
    if out_kind != "batch" or out_amount == "all"  # for batch, only "all" is valid
]

SHORT_INDEXING = [
    {"input": ("all", "batch"), "output": ("all", "batch")},
    {"input": ("all", "shared"), "output": ("all", "batch")},
    {"input": ("all", "batch"), "output": ("all", "shared")},
    {"input": ("first", "indexed"), "output": ("all", "indexed")},
]


GRAD = [False, True]

BACKWARD = [False, True]

BATCH_SIZE = [0, 5]


@pytest.mark.parametrize("name, polynomial", SEGMENTED_POLYNOMIALS[:1])
@pytest.mark.parametrize("dtype, math_dtype", DATA_TYPES_IN_MATH[:1])
@pytest.mark.parametrize("batch_size", BATCH_SIZE[1:])
@pytest.mark.parametrize("mode", EXPORT_MODES[:1])
@pytest.mark.parametrize("grad", GRAD[1:])
@pytest.mark.parametrize("backward", BACKWARD[1:])
@pytest.mark.parametrize("indexing", ALL_INDEXING)
def test_segmented_polynomial_indexing(
    name,
    polynomial,
    dtype,
    math_dtype,
    batch_size,
    mode,
    grad,
    backward,
    indexing,
    tmp_path,
):
    run_segmented_polynomial_test(
        name,
        polynomial,
        dtype,
        math_dtype,
        batch_size,
        mode,
        grad,
        backward,
        indexing,
        tmp_path,
    )


@pytest.mark.parametrize("name, polynomial", SEGMENTED_POLYNOMIALS)
@pytest.mark.parametrize("dtype, math_dtype", DATA_TYPES_IN_MATH)
@pytest.mark.parametrize("batch_size", BATCH_SIZE[1:])
@pytest.mark.parametrize("mode", EXPORT_MODES[:1])
@pytest.mark.parametrize("grad", GRAD)
@pytest.mark.parametrize("backward", BACKWARD)
@pytest.mark.parametrize("indexing", SHORT_INDEXING)
def test_segmented_polynomial_dytpes(
    name,
    polynomial,
    dtype,
    math_dtype,
    batch_size,
    mode,
    grad,
    backward,
    indexing,
    tmp_path,
):
    run_segmented_polynomial_test(
        name,
        polynomial,
        dtype,
        math_dtype,
        batch_size,
        mode,
        grad,
        backward,
        indexing,
        tmp_path,
    )


@pytest.mark.parametrize("name, polynomial", SEGMENTED_POLYNOMIALS)
@pytest.mark.parametrize("dtype, math_dtype", DATA_TYPES_IN_MATH[:1])
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("mode", EXPORT_MODES)
@pytest.mark.parametrize("grad", GRAD)
@pytest.mark.parametrize("backward", BACKWARD[1:])
@pytest.mark.parametrize("indexing", SHORT_INDEXING)
def test_segmented_polynomial_export(
    name,
    polynomial,
    dtype,
    math_dtype,
    batch_size,
    mode,
    grad,
    backward,
    indexing,
    tmp_path,
):
    run_segmented_polynomial_test(
        name,
        polynomial,
        dtype,
        math_dtype,
        batch_size,
        mode,
        grad,
        backward,
        indexing,
        tmp_path,
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
