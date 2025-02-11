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
from functools import partial

import jax
import jax.core
import jax.extend
import jax.lax
import jax.numpy as jnp
from jax.interpreters import ad, batching, mlir, xla

import cuequivariance as cue

logger = logging.getLogger(__name__)


def tensor_product(
    descriptors: list[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    inputs: list[jax.Array],
    outputs_shape_dtype: list[jax.ShapeDtypeStruct],
    indices: list[jax.Array | None] | None = None,
    *,
    math_dtype: jnp.dtype | None = None,
    # precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    # algorithm: str = "sliced",
    # use_custom_primitive: bool = True,
    # use_custom_kernels: bool | None = False,
    name: str | None = None,
    # **options,
) -> jax.Array:
    if name is None:
        name = "tensor_product"

    buffers = inputs + outputs_shape_dtype

    for i, buffer in enumerate(buffers):
        assert buffer.ndim == 2, (
            f"Expected buffer {i} to have 2 dimensions, got {buffer.shape}"
        )

    if math_dtype is None:
        math_dtype = jnp.result_type(*buffers)
        if math_dtype not in (jnp.float32, jnp.float64):
            math_dtype = jnp.float32

    assert math_dtype in (jnp.float32, jnp.float64), (
        f"math_dtype must be float32 or float64, got {math_dtype}"
    )

    if indices is None:
        indices = [None] * len(buffers)

    buffer_index = []
    unique_indices = []
    for i, index in enumerate(indices):
        if index is None:
            buffer_index.append(-1)
        else:
            found = False
            for j, unique_index in enumerate(unique_indices):
                if index is unique_index:
                    buffer_index.append(j)
                    found = True
                    break
            if not found:
                buffer_index.append(len(unique_indices))
                unique_indices.append(index)

    return tensor_product_prim(
        inputs,
        outputs_shape_dtype,
        unique_indices,
        buffer_index,
        descriptors,
        math_dtype,
        name,
    )


tensor_product_p = jax.extend.core.Primitive("tensor_product")
tensor_product_p.multiple_results = True


def tensor_product_prim(
    inputs: list[jax.Array],  # inputs
    outputs_shape_dtype: list[jax.ShapeDtypeStruct],  # output shapes and dtypes
    unique_indices: list[jax.Array],
    buffer_index: list[int],
    descriptors: list[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    math_dtype: jnp.dtype,
    name: str,
) -> tuple[jax.Array, ...]:  # output buffers
    return tensor_product_p.bind(
        *inputs,
        *unique_indices,
        buffer_index=tuple(buffer_index),
        outputs_shape_dtype=tuple(outputs_shape_dtype),
        descriptors=frozenset(descriptors),
        math_dtype=math_dtype,
        name=name,
    )


def clean_inputs(
    inputs: list[jax.Array], operations: list[cue.Operation]
) -> tuple[list[jax.Array], list[cue.Operation]]:
    num_inputs = len(inputs)

    in_buffers = set()
    for ope in operations:
        in_buffers.update(ope.input_buffers(num_inputs))
    in_buffers = sorted(in_buffers)

    # remove unused inputs
    inputs = [inputs[i] for i in in_buffers]
    operations = [
        cue.Operation(
            [
                in_buffers.index(i) if i < num_inputs else i - num_inputs + len(inputs)
                for i in ope.buffers
            ]
        )
        for ope in operations
    ]
    num_inputs = len(inputs)

    # remove duplicate inputs
    unique_inputs = []
    for x in inputs:
        if id(x) not in map(id, unique_inputs):
            unique_inputs.append(x)
    unique_ids = list(map(id, unique_inputs))
    operations = [
        cue.Operation(
            [
                unique_ids.index(id(inputs[i]))
                if i < num_inputs
                else i - num_inputs + len(unique_inputs)
                for i in ope.buffers
            ]
        )
        for ope in operations
    ]

    return unique_inputs, operations


def map_indices(
    old_indices: list[jax.Array], old_buffer_index: list[int], mapping: list[int]
) -> tuple[list[jax.Array], list[int]]:
    new_indices = []
    new_buffer_index = []

    for new_i, old_i in enumerate(mapping):
        if old_buffer_index[old_i] >= 0:
            idx = old_indices[old_buffer_index[old_i]]
            found = False
            for i, new_idx in enumerate(new_indices):
                if idx is new_idx:
                    new_buffer_index.append(i)
                    found = True
                    break
            if not found:
                new_buffer_index.append(len(new_indices))
                new_indices.append(idx)
        else:
            new_buffer_index.append(-1)
    return new_indices, new_buffer_index


def tensor_product_abstract_eval(
    *inputs_and_indices: jax.core.ShapedArray,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    descriptors: frozenset[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    math_dtype: jnp.dtype,
    name: str,
) -> tuple[jax.core.ShapedArray, ...]:
    return tuple(
        jax.core.ShapedArray(out.shape, out.dtype) for out in outputs_shape_dtype
    )


def tensor_product_impl(
    platform: str | None,
    *inputs_and_indices: jax.Array,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    descriptors: frozenset[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    math_dtype: jnp.dtype,
    name: str,
) -> tuple[jax.Array, ...]:
    print(platform, name)

    num_inputs = len(buffer_index) - len(outputs_shape_dtype)
    inputs, indices = inputs_and_indices[:num_inputs], inputs_and_indices[num_inputs:]
    del inputs_and_indices

    buffers = list(inputs) + list(outputs_shape_dtype)

    def reshape(x, shape):
        if isinstance(x, jax.Array):
            return jnp.reshape(x, shape)
        else:
            return jax.core.ShapedArray(shape, x.dtype)

    new_descriptors: list[tuple[cue.Operation, cue.SegmentedTensorProduct]] = []
    for ope, stp in descriptors:
        ope: cue.Operation
        stp: cue.SegmentedTensorProduct
        for set_of_operands in ope.operands_with_identical_buffers():
            stp = stp.sort_indices_for_identical_operands(set_of_operands)
        new_descriptors.append((ope, stp))

        for i, operand in zip(ope.buffers, stp.operands):
            b = buffers[i]
            buffers[i] = reshape(
                b, (b.shape[0], operand.num_segments, operand.segment_size)
            )

        print(stp)
        print(ope.to_string(num_inputs))

    from cuequivariance_ops_jax import (
        Operation,
        Path,
        tensor_product_uniform_1d_jit,
    )

    operations = []
    paths = []

    for ope, stp in new_descriptors:
        operations.append(Operation(ope.buffers, len(paths), stp.num_paths))
        for path in stp.paths:
            paths.append(Path(path.indices, path.coefficients.item()))

    outputs = tensor_product_uniform_1d_jit(
        buffers[:num_inputs],
        buffers[num_inputs:],
        indices,
        buffer_index,
        operations=operations,
        paths=paths,
        math_dtype=math_dtype,
        name=name,
    )
    return [jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])) for x in outputs]


def tensor_product_jvp(
    primals_and_indices: tuple[jax.Array, ...],
    tangents_and_zeros: tuple[jax.Array | ad.Zero, ...],
    *,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    descriptors: frozenset[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    math_dtype: jnp.dtype,
    name: str,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array | ad.Zero, ...]]:
    num_inputs = len(buffer_index) - len(outputs_shape_dtype)

    primals, tangents = (
        primals_and_indices[:num_inputs],
        tangents_and_zeros[:num_inputs],
    )
    indices = primals_and_indices[num_inputs:]
    assert all(isinstance(t, ad.Zero) for t in tangents_and_zeros[num_inputs:])
    del primals_and_indices, tangents_and_zeros

    out_primals = tensor_product_prim(
        primals,
        outputs_shape_dtype,
        indices,
        buffer_index,
        descriptors,
        math_dtype,
        name,
    )

    jvp_indices, jvp_buffer_index = map_indices(
        indices,
        buffer_index,
        [i for i, x in enumerate(primals)]
        + [i for i, x in enumerate(tangents) if not isinstance(x, ad.Zero)]
        + [num_inputs + i for i, x in enumerate(outputs_shape_dtype)],
    )

    jvp_descriptors = []
    for ope, stp in descriptors:
        jvps = ope.jvp([not isinstance(t, ad.Zero) for t in tangents])
        permutations: list[tuple[int, ...]] = stp.symmetries()
        for multiplicator, ope in cue.Operation.group_by_operational_symmetries(
            permutations, jvps
        ):
            jvp_descriptors.append((ope, multiplicator * stp))

    out_tangents = tensor_product_prim(
        list(primals) + [t for t in tangents if not isinstance(t, ad.Zero)],
        outputs_shape_dtype,
        jvp_indices,
        jvp_buffer_index,
        jvp_descriptors,
        math_dtype,
        name + "_jvp",
    )

    return out_primals, out_tangents


def tensor_product_transpose(
    cotangents: tuple[jax.Array | ad.Zero, ...],
    *inputs_and_indices: jax.Array | ad.UndefinedPrimal,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    descriptors: frozenset[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    math_dtype: jnp.dtype,
    name: str,
) -> tuple[jax.Array | ad.Zero | None, ...]:
    num_inputs = len(buffer_index) - len(outputs_shape_dtype)
    inputs, indices = inputs_and_indices[:num_inputs], inputs_and_indices[num_inputs:]
    assert all(not ad.is_undefined_primal(idx) for idx in indices)
    del inputs_and_indices

    # The cotangents replace the outputs as inputs
    # The undefined primal inputs become outputs

    tr_indices, tr_buffer_index = map_indices(
        indices,
        buffer_index,
        [i for i, x in enumerate(inputs) if not ad.is_undefined_primal(x)]
        + [
            num_inputs + i
            for i, x in enumerate(cotangents)
            if not isinstance(x, ad.Zero)
        ]
        + [i for i, x in enumerate(inputs) if ad.is_undefined_primal(x)],
    )

    tr_descriptors = []
    for ope, stp in descriptors:
        ope = ope.transpose(
            [ad.is_undefined_primal(x) for x in inputs],
            [not isinstance(x, ad.Zero) for x in cotangents],
        )
        if ope is not None:
            tr_descriptors.append((ope, stp))

    tmp = tensor_product_prim(
        [x for x in inputs if not ad.is_undefined_primal(x)]
        + [x for x in cotangents if not isinstance(x, ad.Zero)],  # inputs
        [
            jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype)
            for x in inputs
            if ad.is_undefined_primal(x)
        ],
        tr_indices,
        tr_buffer_index,
        tr_descriptors,
        math_dtype,
        name + "_transpose",
    )

    outputs = [None] * len(inputs)
    i = 0
    for b, input in enumerate(inputs):
        if ad.is_undefined_primal(input):
            outputs[b] = tmp[i]
            i += 1
    return tuple(outputs)


def tensor_product_batching(
    batched_inputs_and_indices: tuple[jax.Array, ...],
    batch_axes: tuple[int | None, ...],
    *,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    descriptors: frozenset[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    math_dtype: jnp.dtype,
    name: str,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    num_inputs = len(buffer_index) - len(outputs_shape_dtype)

    batched_inputs, batched_indices = (
        batched_inputs_and_indices[:num_inputs],
        batched_inputs_and_indices[num_inputs:],
    )

    assert len(batched_indices) == 0, "Batching not supported with indexed buffers"

    for i, batch_axis in zip(buffer_index, batch_axes[:num_inputs]):
        if i >= 0 and batch_axis is not None:
            raise ValueError("Batching an indexed buffer is not supported.")

    new_dim = {
        input.shape[axis]
        for input, axis in zip(batched_inputs_and_indices, batch_axes)
        if axis is not None
    }
    assert len(new_dim) == 1, "Expected all batched inputs to have the same size"
    new_dim = new_dim.pop()

    def prepare(input: jax.Array, axis: int | None) -> jax.Array:
        if axis is None:
            return jnp.expand_dims(input, 0)
        else:
            return jnp.moveaxis(input, axis, 0)

    assert len(batched_inputs) == len(batch_axes)
    batched_inputs = [
        prepare(input, axis) for input, axis in zip(batched_inputs, batch_axes)
    ]

    max_m = max(x.shape[0] for x in batched_inputs)
    max_n = max(x.shape[1] for x in batched_inputs)

    def fn(x: jax.Array) -> jax.Array:
        assert x.ndim == 3
        m, n, d = x.shape
        if (m, n) == (1, 1):
            return jnp.reshape(x, (1, d))
        x = jnp.broadcast_to(x, (max_m, max_n, d))
        return jnp.reshape(x, (max_m * max_n, d))

    batched_inputs = [fn(x) for x in batched_inputs]

    new_outputs_shape_dtype = [
        jax.ShapeDtypeStruct((max_m * max_n, *out.shape[1:]), out.dtype)
        for out in outputs_shape_dtype
    ]

    outputs = tensor_product_p.bind(
        *batched_inputs,
        buffer_index=buffer_index,
        outputs_shape_dtype=new_outputs_shape_dtype,
        descriptors=descriptors,
        math_dtype=math_dtype,
        name=name + "_batching",
    )
    outputs = tuple(jnp.reshape(x, (max_m, max_n, *x.shape[1:])) for x in outputs)
    outputs = tuple(
        jnp.sum(x, axis=1, keepdims=True) if y.shape[0] == 1 else x
        for x, y in zip(outputs, outputs_shape_dtype)
    )
    return outputs, (0,) * len(outputs)


tensor_product_p.def_abstract_eval(tensor_product_abstract_eval)
tensor_product_p.def_impl(partial(xla.apply_primitive, tensor_product_p))
mlir.register_lowering(
    tensor_product_p,
    mlir.lower_fun(
        partial(tensor_product_impl, "cuda"), tensor_product_p.multiple_results
    ),
    "cuda",
)
mlir.register_lowering(
    tensor_product_p,
    mlir.lower_fun(
        partial(tensor_product_impl, None), tensor_product_p.multiple_results
    ),
    None,
)
ad.primitive_jvps[tensor_product_p] = tensor_product_jvp
ad.primitive_transposes[tensor_product_p] = tensor_product_transpose
batching.primitive_batchers[tensor_product_p] = tensor_product_batching
