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
from cuequivariance_jax.primitives.primitives_utils import reshape
from cuequivariance_jax.primitives.segmented_polynomial_ops_impl import (
    segmented_polynomial_ops_impl,
)
from cuequivariance_jax.primitives.segmented_polynomial_vanilla_impl import (
    segmented_polynomial_vanilla_impl,
)

logger = logging.getLogger(__name__)


def segmented_polynomial(
    polynomial: cue.SegmentedPolynomial,
    inputs: list[jax.Array],
    outputs_shape_dtype: list[jax.ShapeDtypeStruct],
    indices: list[jax.Array | None] | None = None,
    *,
    math_dtype: jnp.dtype | None = None,
    name: str | None = None,
    impl: str = "auto",
) -> list[jax.Array]:
    """Compute a segmented polynomial.

    This function evaluates a segmented polynomial using either CUDA or JAX implementation.
    The implementation choice is determined by the input characteristics and availability
    of CUDA support.

    Args:
        polynomial: The segmented polynomial to compute.
        inputs: List of input buffers as JAX arrays.
        outputs_shape_dtype: List of output shapes and dtypes specifications.
        indices: Optional list of indices for inputs and outputs. If None, no indexing
            is applied. Defaults to None.
        math_dtype: Data type for computational operations. If None, automatically
            determined from input types, defaulting to float32 if no float64 inputs
            are present. Defaults to None.
        name: Optional name for the operation. Defaults to None.
        impl: Implementation to use, one of ["auto", "cuda", "jax"]. If "auto",
            uses CUDA when available, falling back to JAX otherwise. Defaults to "auto".

    Returns:
        List of JAX arrays containing the computed tensor product results.

    Features:
        - CUDA kernel activation conditions:
            - STPs have a single mode which is a multiple of 32 (e.g. channelwise
              tensor product with subscripts ``u,u,,u`` where u=128)
            - Math data type is float32 or float64
            - Input/output data types can be float32, float64, float16, or bfloat16
            - Indices must be int32
        - Supports infinite derivatives through JVP and transpose rules
        - Limited batching support:
            - Cannot batch buffers with indices
            - Non-trivial batching may impact performance
        - Automatic optimizations:
            - Based on STP symmetries
            - Based on input buffer repetition patterns
        - Automatic pruning of unused buffers and indices

    Note:
        The function automatically determines the best implementation based on the
        input characteristics when impl="auto". For maximum performance with CUDA-capable
        hardware, ensure inputs match the CUDA kernel activation conditions.
    """

    if name is None:
        name = "segmented_polynomial"

    assert len(inputs) == polynomial.num_inputs
    assert len(outputs_shape_dtype) == polynomial.num_outputs
    buffers = list(inputs) + list(outputs_shape_dtype)

    if indices is None:
        indices = [None] * len(buffers)

    if len(indices) != len(buffers):
        raise ValueError(
            f"Expected {len(buffers)} indices, got {len(indices)}. "
            "Please provide an index for each buffer. "
            "If a buffer does not have an index, please set it to None."
        )

    def fn(
        buffer: jax.Array | jax.ShapeDtypeStruct, idx: jax.Array | None
    ) -> jax.Array | jax.ShapeDtypeStruct:
        if buffer.ndim == 1 and idx is None:
            return reshape(buffer, (1, buffer.shape[0]))
        return buffer

    buffers = list(map(fn, buffers, indices))

    for i, buffer in enumerate(buffers):
        assert buffer.ndim == 2, (
            f"Expected buffer {i} to have 2 dimensions, got {buffer.shape}"
        )
    for i, idx in enumerate(indices):
        assert idx is None or idx.ndim == 1, (
            f"Expected index {i} to have 1 dimension, got {idx.shape}"
        )

    if math_dtype is None:
        math_dtype = jnp.result_type(*buffers)
        if math_dtype not in (jnp.float32, jnp.float64):
            math_dtype = jnp.float32

    assert math_dtype in (jnp.float32, jnp.float64), (
        f"math_dtype must be float32 or float64, got {math_dtype}"
    )

    buffer_index = []
    unique_indices = []
    for idx in indices:
        if idx is None:
            buffer_index.append(-1)
        else:
            found = False
            for j, uidx in enumerate(unique_indices):
                if idx is uidx:
                    buffer_index.append(j)
                    found = True
                    break
            if not found:
                buffer_index.append(len(unique_indices))
                unique_indices.append(idx)

    kwargs = dict(
        inputs=buffers[: len(inputs)],
        outputs_shape_dtype=buffers[len(inputs) :],
        indices=unique_indices,
        buffer_index=buffer_index,
        polynomial=polynomial,
        math_dtype=math_dtype,
        name=name,
    )

    if impl == "naive_jax":
        outputs = segmented_polynomial_vanilla_impl(**kwargs)
    else:
        outputs = segmented_polynomial_prim(**kwargs, impl=impl)

    def fn(x: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jnp.reshape(x, shape)

    return list(map(fn, outputs, [out.shape for out in outputs_shape_dtype]))


segmented_polynomial_p = jax.extend.core.Primitive("segmented_polynomial")
segmented_polynomial_p.multiple_results = True


def segmented_polynomial_prim(
    inputs: list[jax.Array],  # input buffers
    outputs_shape_dtype: list[jax.ShapeDtypeStruct],  # output shapes and dtypes
    indices: list[jax.Array],  # index buffers
    buffer_index: list[int],  # maps: buffer index -> unique indices index
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
    impl: str = "auto",
    return_none_if_empty: bool = False,
) -> tuple[jax.Array, ...]:  # output buffers
    """
    - Filters out unused buffers and indices
    - Calls the tensor product primitive
    - Maps the outputs back to the original output buffers
    """
    assert len(inputs) + len(outputs_shape_dtype) == len(buffer_index)
    assert max(buffer_index) < len(indices)

    outputs_shape_dtype = [
        jax.ShapeDtypeStruct(x.shape, x.dtype) for x in outputs_shape_dtype
    ]

    polynomial = polynomial.consolidate()
    used_buffers = polynomial.used_buffers()
    polynomial = polynomial.remove_unused_buffers()

    used_indices = sorted(
        {buffer_index[i] for i in used_buffers if buffer_index[i] >= 0}
    )

    used_outputs = segmented_polynomial_p.bind(
        *[inputs[i] for i in used_buffers[: polynomial.num_inputs]],
        *[indices[i] for i in used_indices],
        buffer_index=tuple(
            used_indices.index(buffer_index[i]) if buffer_index[i] >= 0 else -1
            for i in used_buffers
        ),
        outputs_shape_dtype=tuple(
            outputs_shape_dtype[i - len(inputs)]
            for i in used_buffers[polynomial.num_inputs :]
        ),
        polynomial=polynomial,
        math_dtype=jnp.dtype(math_dtype),
        name=str(name),
        impl=impl,
    )

    if return_none_if_empty:
        outputs = [None] * len(outputs_shape_dtype)
    else:
        outputs = [jnp.zeros(out.shape, out.dtype) for out in outputs_shape_dtype]

    for i, output in zip(used_buffers[polynomial.num_inputs :], used_outputs):
        outputs[i - len(inputs)] = output

    return tuple(outputs)


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


def segmented_polynomial_dce(
    *inputs_and_indices: jax.Array,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
    impl: str,
) -> tuple[jax.Array, ...]:
    def fn(inputs_and_indices: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        outputs = segmented_polynomial_p.bind(
            *inputs_and_indices,
            buffer_index=buffer_index,
            outputs_shape_dtype=outputs_shape_dtype,
            polynomial=polynomial,
            math_dtype=math_dtype,
            name=name,
            impl=impl,
        )
        assert isinstance(outputs, (tuple, list))
        return tuple(outputs)

    try:
        from jax.experimental.custom_dce import custom_dce
    except ImportError:
        return fn(inputs_and_indices)
    else:
        fn_dce = custom_dce(fn)

        @fn_dce.def_dce
        def fn_dce_rule(
            used_outputs: list[bool], inputs_and_indices
        ) -> tuple[jax.Array | None, ...]:
            assert not all(used_outputs)

            num_inputs = len(buffer_index) - len(outputs_shape_dtype)
            inputs, indices = (
                inputs_and_indices[:num_inputs],
                inputs_and_indices[num_inputs:],
            )
            return segmented_polynomial_prim(
                inputs,
                outputs_shape_dtype,
                indices,
                buffer_index,
                polynomial.compute_only(used_outputs),
                math_dtype,
                name,
                impl,
                return_none_if_empty=True,
            )

        return fn_dce(inputs_and_indices)


def segmented_polynomial_abstract_eval(
    *inputs_and_indices: jax.core.ShapedArray,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
    impl: str,
) -> tuple[jax.core.ShapedArray, ...]:
    return tuple(
        jax.core.ShapedArray(out.shape, out.dtype) for out in outputs_shape_dtype
    )


def segmented_polynomial_impl(
    platform: str | None,
    *inputs_and_indices: jax.Array,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
    impl: str,
) -> tuple[jax.Array, ...]:
    num_inputs = len(buffer_index) - len(outputs_shape_dtype)
    inputs, indices = inputs_and_indices[:num_inputs], inputs_and_indices[num_inputs:]
    del inputs_and_indices

    polynomial = polynomial.sort_indices_for_identical_operands()

    outputs = None
    kwargs = dict(
        inputs=inputs,
        outputs_shape_dtype=outputs_shape_dtype,
        indices=indices,
        buffer_index=buffer_index,
        polynomial=polynomial,
        math_dtype=math_dtype,
        name=name,
    )

    assert impl in ("auto", "cuda", "jax")

    if platform == "cuda" and impl in ("auto", "cuda"):
        outputs, msg = segmented_polynomial_ops_impl(**kwargs)
    else:
        msg = f"{platform=}, {impl=}"

    if impl == "cuda" and outputs is None:
        raise RuntimeError(f"Failed to use CUDA implementation: {msg}")

    if outputs is None:
        outputs = segmented_polynomial_vanilla_impl(**kwargs)

    assert outputs is not None
    return outputs


def segmented_polynomial_jvp(
    primals_and_indices: tuple[jax.Array, ...],
    tangents_and_zeros: tuple[jax.Array | ad.Zero, ...],
    *,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
    impl: str,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array | ad.Zero, ...]]:
    num_inputs = len(buffer_index) - len(outputs_shape_dtype)

    primals, tangents = (
        primals_and_indices[:num_inputs],
        tangents_and_zeros[:num_inputs],
    )
    indices = primals_and_indices[num_inputs:]
    assert all(isinstance(t, ad.Zero) for t in tangents_and_zeros[num_inputs:])
    del primals_and_indices, tangents_and_zeros

    out_primals = segmented_polynomial_prim(
        primals,
        outputs_shape_dtype,
        indices,
        buffer_index,
        polynomial,
        math_dtype,
        name,
        impl=impl,
    )

    jvp_indices, jvp_buffer_index = map_indices(
        indices,
        buffer_index,
        [i for i, x in enumerate(primals)]
        + [i for i, x in enumerate(tangents) if not isinstance(x, ad.Zero)]
        + [num_inputs + i for i, x in enumerate(outputs_shape_dtype)],
    )

    out_tangents = segmented_polynomial_prim(
        list(primals) + [t for t in tangents if not isinstance(t, ad.Zero)],
        outputs_shape_dtype,
        jvp_indices,
        jvp_buffer_index,
        polynomial.jvp([not isinstance(t, ad.Zero) for t in tangents]),
        math_dtype,
        name + "_jvp",
        impl=impl,
    )

    return out_primals, out_tangents


def segmented_polynomial_transpose(
    cotangents: tuple[jax.Array | ad.Zero, ...],
    *inputs_and_indices: jax.Array | ad.UndefinedPrimal,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
    impl: str,
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

    tmp = segmented_polynomial_prim(
        [x for x in inputs if not ad.is_undefined_primal(x)]
        + [x for x in cotangents if not isinstance(x, ad.Zero)],  # inputs
        [
            jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype)
            for x in inputs
            if ad.is_undefined_primal(x)
        ],
        tr_indices,
        tr_buffer_index,
        polynomial.transpose(
            [ad.is_undefined_primal(x) for x in inputs],
            [not isinstance(x, ad.Zero) for x in cotangents],
        ),
        math_dtype,
        name + "_transpose",
        impl=impl,
        return_none_if_empty=True,
    )

    outputs = [None] * (len(inputs) + len(indices))
    i = 0
    for b, input in enumerate(inputs):
        if ad.is_undefined_primal(input):
            outputs[b] = tmp[i] if tmp[i] is not None else ad.Zero(input.aval)
            i += 1
    return tuple(outputs)


def segmented_polynomial_batching(
    batched_inputs_and_indices: tuple[jax.Array, ...],
    batch_axes_of_inputs_and_indices: tuple[int | None, ...],
    *,
    buffer_index: tuple[int, ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
    impl: str,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    num_inputs = len(buffer_index) - len(outputs_shape_dtype)

    batched_inputs, batched_indices = (
        batched_inputs_and_indices[:num_inputs],
        batched_inputs_and_indices[num_inputs:],
    )
    del batched_inputs_and_indices
    batch_axes_of_inputs, batch_axes_of_indices = (
        batch_axes_of_inputs_and_indices[:num_inputs],
        batch_axes_of_inputs_and_indices[num_inputs:],
    )
    del batch_axes_of_inputs_and_indices

    for i in buffer_index[num_inputs:]:
        if i >= 0:
            raise ValueError("Batching is not supported when outputs have indices")
    for i, axis in zip(buffer_index[:num_inputs], batch_axes_of_inputs):
        if i >= 0 and axis is not None:
            raise ValueError("Batching is not supported for inputs that have indices")

    def prepare(input: jax.Array, axis: int | None) -> jax.Array:
        if axis is None:
            return jnp.expand_dims(input, 0)
        else:
            return jnp.moveaxis(input, axis, 0)

    batched_inputs = [
        input if i >= 0 else prepare(input, axis)
        for i, input, axis in zip(buffer_index, batched_inputs, batch_axes_of_inputs)
    ]
    batched_indices = [
        prepare(input, axis)
        for input, axis in zip(batched_indices, batch_axes_of_indices)
    ]

    # possible input buffer shapes:
    #  - (new_dim | 1, batch_size | 1, size)
    #  - (max_index, size)
    # possible indices shapes:
    #  - (new_dim | 1, batch_size)
    new_dim = 1
    batch_size = 1
    for x in batched_inputs:
        if x.ndim == 3:
            if x.shape[0] != 1:
                new_dim = x.shape[0]
            if x.shape[1] != 1:
                batch_size = x.shape[1]
    for x in batched_indices:
        if x.shape[0] != 1:
            new_dim = x.shape[0]
        if x.shape[1] != 1:
            batch_size = x.shape[1]

    def flatten_input(x: jax.Array) -> jax.Array:
        m, n, d = x.shape
        if (m, n) == (1, 1):
            return jnp.reshape(x, (1, d))
        x = jnp.broadcast_to(x, (new_dim, batch_size, d))
        return jnp.reshape(x, (new_dim * batch_size, d))

    batched_inputs = [flatten_input(x) if x.ndim == 3 else x for x in batched_inputs]

    def flatten_index(x: jax.Array) -> jax.Array:
        x = jnp.broadcast_to(x, (new_dim, batch_size))
        return jnp.reshape(x, (new_dim * batch_size))

    batched_indices = [flatten_index(x) for x in batched_indices]

    new_outputs_shape_dtype = tuple(
        jax.ShapeDtypeStruct((new_dim * batch_size, *out.shape[1:]), out.dtype)
        for out in outputs_shape_dtype
    )

    outputs = segmented_polynomial_p.bind(
        *batched_inputs,
        *batched_indices,
        buffer_index=buffer_index,
        outputs_shape_dtype=new_outputs_shape_dtype,
        polynomial=polynomial,
        math_dtype=math_dtype,
        name=name + "_batching",
        impl=impl,
    )
    outputs = tuple(
        jnp.reshape(x, (new_dim, batch_size, *x.shape[1:])) for x in outputs
    )
    outputs = tuple(
        jnp.sum(x, axis=1, keepdims=True) if y.shape[0] == 1 else x
        for x, y in zip(outputs, outputs_shape_dtype)
    )
    return outputs, (0,) * len(outputs)


segmented_polynomial_p.def_abstract_eval(segmented_polynomial_abstract_eval)
segmented_polynomial_p.def_impl(partial(xla.apply_primitive, segmented_polynomial_p))
mlir.register_lowering(
    segmented_polynomial_p,
    mlir.lower_fun(
        partial(segmented_polynomial_impl, "cuda"),
        segmented_polynomial_p.multiple_results,
    ),
    "cuda",
)
mlir.register_lowering(
    segmented_polynomial_p,
    mlir.lower_fun(
        partial(segmented_polynomial_impl, None),
        segmented_polynomial_p.multiple_results,
    ),
    None,
)
ad.primitive_jvps[segmented_polynomial_p] = segmented_polynomial_jvp
ad.primitive_transposes[segmented_polynomial_p] = segmented_polynomial_transpose
batching.primitive_batchers[segmented_polynomial_p] = segmented_polynomial_batching
