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
from cuequivariance_torch._tests.utils import module_with_mode, tol_dict

import cuequivariance as cue
import cuequivariance_torch as cuet

global_device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)


def generate_segmented_polynomials():
    result = []

    def yield_from(fn):
        result.extend(list(fn()))

    @yield_from
    def channelwise_tensor_product():
        e = (
            cue.descriptors.channelwise_tensor_product(
                cue.Irreps("O3", "16x0e + 16x1o"),
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
            16 * cue.Irreps("O3", "0e + 1o + 2e"),
            16 * cue.Irreps("O3", "0e + 1o"),
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
    operands, dtype, idx_amount, idx_kind, batch_size, tensor_init_fn, device
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
            inds = torch.randint(0, index_size, (batch_size,), device=device)
            indices[i], _ = torch.sort(inds)
            local_batch = index_size
        tensors.append(tensor_init_fn(local_batch, x.size))
    return tensors, indices


def make_inputs(polynomial, dtype, indexing, batch_size, device):
    def tensor_init_inputs(batch_size, size):
        return torch.randn(
            (batch_size, size), device=device, dtype=dtype, requires_grad=True
        )

    inputs, input_indices = make_inputs_for_operands(
        polynomial.inputs,
        dtype,
        *indexing["input"],
        batch_size,
        tensor_init_inputs,
        device,
    )

    def tensor_init_outputs(batch_size, size):
        return torch.empty(
            1, device=device, dtype=dtype, requires_grad=False
        ).broadcast_to(batch_size, size)

    outputs, output_indices = make_inputs_for_operands(
        polynomial.outputs,
        dtype,
        *indexing["output"],
        batch_size,
        tensor_init_outputs,
        device,
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
    method,
    dtype,
    math_dtype,
    batch_size,
    mode,
    grad,
    backward,
    indexing,
    tmp_path,
    device=global_device,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if grad and mode == "jit":
        pytest.skip("torch.jit.trace does not work with inline autograd")
    if grad and backward and dtype.itemsize <= 2:
        pytest.skip("double backward with fp16/bf16 lacks accuracy")

    # Special case for indexed_linear dtype
    if math_dtype == "CUBLAS_COMPUTE_32F" and method == "indexed_linear":
        o_math_dtype = math_dtype
        math_dtype = torch.float32
    else:
        o_math_dtype = math_dtype

    m_ref = cuet.SegmentedPolynomial(
        polynomial, method="naive", math_dtype=math_dtype
    ).to(device)
    m = cuet.SegmentedPolynomial(polynomial, method=method, math_dtype=o_math_dtype).to(
        device
    )

    t_math_dtype = math_dtype if math_dtype is not None else dtype
    test_tol_dict = tol_dict[(dtype, t_math_dtype)]

    if grad:
        m_ref = Grad(m_ref)
        m = Grad(m)
        test_tol_dict = tol_dict_grad(test_tol_dict)

    inp = make_inputs(polynomial, dtype, indexing, batch_size, device)
    m = module_with_mode(mode, m, inp, math_dtype, tmp_path)
    m.to(device)

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
    (torch.float32, None),
    (torch.float64, None),
    (torch.float32, "float32"),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
]

METHODS = ["uniform_1d", "fused_tp"]

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

DEVICES = (
    [torch.device("cuda:0"), torch.device("cpu")]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)


@pytest.mark.parametrize("name, polynomial", SEGMENTED_POLYNOMIALS[:1])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "dtype, math_dtype",
    [
        (torch.float32, torch.float64),
    ],
)
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("mode", ["eager"])
@pytest.mark.parametrize("grad", [True])
@pytest.mark.parametrize("backward", [True])
@pytest.mark.parametrize("indexing", ALL_INDEXING)
def test_segmented_polynomial_indexing(
    name,
    polynomial,
    method,
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
        method,
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
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype, math_dtype", DATA_TYPES_IN_MATH)
@pytest.mark.parametrize("batch_size", BATCH_SIZE[1:])
@pytest.mark.parametrize("mode", EXPORT_MODES[:1])
@pytest.mark.parametrize("grad", GRAD)
@pytest.mark.parametrize("backward", BACKWARD)
@pytest.mark.parametrize("indexing", SHORT_INDEXING)
@pytest.mark.parametrize("device", DEVICES)
def test_segmented_polynomial_dytpes(
    name,
    polynomial,
    method,
    dtype,
    math_dtype,
    batch_size,
    mode,
    grad,
    backward,
    indexing,
    tmp_path,
    device,
):
    # Skipping all tests that have many options that are not default
    complexity = 0
    if name != "channelwise_tensor_product":
        complexity += 2
    if dtype != torch.float32:
        complexity += 1
    if grad:
        complexity += 2
    if backward:
        complexity += 2
    if indexing != SHORT_INDEXING[0]:
        complexity += 1
    if device == torch.device("cpu"):
        complexity += 2
    if complexity > 2:
        pytest.skip("Skipping tests with many options that are not default")

    # Unsupported combinations
    if method == "fused_tp" and name == "symmetric_contraction":
        pytest.skip("Skipping fused TP for symmetric contraction: unsupported")
    if (
        method == "fused_tp"
        and math_dtype in [torch.float32, None]
        and dtype == torch.float64
    ):
        pytest.skip("Skipping fused TP for float32 math_dtype with float64 inputs")

    run_segmented_polynomial_test(
        name,
        polynomial,
        method,
        dtype,
        math_dtype,
        batch_size,
        mode,
        grad,
        backward,
        indexing,
        tmp_path,
        device,
    )


# Testing export modes, only using one option for each to save time
# We also have to test naive method explicitly because the reference is not exported
@pytest.mark.parametrize("name, polynomial", SEGMENTED_POLYNOMIALS[:1])
@pytest.mark.parametrize("method", METHODS + ["naive"])
@pytest.mark.parametrize("dtype, math_dtype", DATA_TYPES_IN_MATH[:1])
@pytest.mark.parametrize("batch_size", BATCH_SIZE[1:])
@pytest.mark.parametrize("mode", EXPORT_MODES)
@pytest.mark.parametrize("grad", GRAD[1:])
@pytest.mark.parametrize("backward", BACKWARD[1:])
@pytest.mark.parametrize("indexing", SHORT_INDEXING[:1])
def test_segmented_polynomial_export(
    name,
    polynomial,
    method,
    dtype,
    math_dtype,
    batch_size,
    mode,
    grad,
    backward,
    indexing,
    tmp_path,
):
    # Skip export mode for naive method for issues with testing
    if method == "naive" and mode == "export":
        pytest.skip("Skipping export mode for naive method")

    run_segmented_polynomial_test(
        name,
        polynomial,
        method,
        dtype,
        math_dtype,
        batch_size,
        mode,
        grad,
        backward,
        indexing,
        tmp_path,
    )


@pytest.mark.parametrize("method", METHODS + ["indexed_linear"])
@pytest.mark.parametrize("dtype, math_dtype", DATA_TYPES_IN_MATH[2:6])
@pytest.mark.parametrize("batch_size", BATCH_SIZE[1:])
@pytest.mark.parametrize("mode", EXPORT_MODES[:1])
@pytest.mark.parametrize("grad", GRAD)
@pytest.mark.parametrize("backward", BACKWARD)
def test_segmented_polynomial_indexed_linear(
    method,
    dtype,
    math_dtype,
    batch_size,
    mode,
    grad,
    backward,
    tmp_path,
):
    name = "indexed_linear"
    polynomial = cue.descriptors.linear(
        cue.Irreps("O3", "16x0e + 16x1o + 16x2e"),
        cue.Irreps("O3", "16x0e + 16x1o + 16x2e"),
    ).polynomial

    indexing = {"input": ("first", "indexed"), "output": ("all", "batch")}

    # Unsupported combinations
    if method == "uniform_1d":
        pytest.skip("Linear is not supported for uniform_1d")
    if (
        method == "fused_tp"
        and math_dtype in [torch.float32, None]
        and dtype == torch.float64
    ):
        pytest.skip("Skipping fused TP for float32 math_dtype with float64 inputs")
    if method == "indexed_linear" and math_dtype not in [None]:
        pytest.skip("indexed_linear does not support math_dtype")

    run_segmented_polynomial_test(
        name,
        polynomial,
        method,
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
