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
import logging
import math
import os
import warnings

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from cuequivariance_jax.segmented_polynomials.indexing_mode import IndexingMode
from cuequivariance_jax.segmented_polynomials.segmented_polynomial_indexed_linear import (
    execute_indexed_linear,
)
from cuequivariance_jax.segmented_polynomials.utils import (
    batch_size,
    indexing,
    math_dtype_for_naive_method,
)

import cuequivariance as cue

logger = logging.getLogger(__name__)


def execute_naive(
    inputs: list[jax.Array],  # shape (*batch_sizes, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    index_configuration: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
) -> list[jax.Array]:  # output buffers
    if any(mode == IndexingMode.REPEATED for modes in index_mode for mode in modes):
        return execute_indexed_linear(
            inputs,
            outputs_shape_dtype,
            indices,
            index_configuration,
            index_mode,
            polynomial,
            math_dtype,
            name,
            run_kernel=False,
        )

    compute_dtype, precision = math_dtype_for_naive_method(
        jnp.result_type(
            *[x.dtype for x in inputs] + [x.dtype for x in outputs_shape_dtype]
        ),
        math_dtype,
    )
    del math_dtype

    num_inputs = len(index_configuration) - len(outputs_shape_dtype)

    io_buffers = list(inputs) + [
        jnp.zeros(out.shape, out.dtype) for out in outputs_shape_dtype
    ]
    index_configuration = np.array(index_configuration, dtype=np.int32)
    num_batch_axes = index_configuration.shape[1]
    batch_sizes = [
        batch_size(
            [
                x.shape[i]
                for x, idx in zip(io_buffers, index_configuration[:, i])
                if idx < 0
            ],
        )
        for i in range(num_batch_axes)
    ]

    def scatter(i: int) -> jax.Array:
        idx = indexing(index_configuration[i], io_buffers[i].shape, indices)
        return inputs[i][idx]

    def gather(i: int, x: jax.Array) -> jax.Array:
        idx = indexing(index_configuration[i], io_buffers[i].shape, indices)
        return io_buffers[i].at[idx].add(x)

    for operation, d in polynomial.operations:
        ope_out, b_out = operation.output_operand_buffer(num_inputs)

        out = outputs_shape_dtype[b_out - num_inputs]
        out = jax.ShapeDtypeStruct(
            tuple(
                b if i >= 0 else s
                for i, b, s in zip(index_configuration[b_out], batch_sizes, out.shape)
            )
            + out.shape[-1:],
            out.dtype,
        )

        output_segments: list[list[jax.Array]] = tp_list_list(
            *[scatter(i) for i in operation.input_buffers(num_inputs)],
            out_batch_shape=out.shape[:-1],
            d=d.move_operand_last(ope_out),
            output_dtype=out.dtype,
            compute_dtype=compute_dtype,
            precision=precision,
            algorithm="compact_stacked" if d.all_same_segment_shape() else "sliced",
        )
        out = sum_cat_list_list(
            d.operands[ope_out],
            output_segments,
            out.shape[:-1],
            out.dtype,
        )
        io_buffers[b_out] = gather(b_out, out)

    return tuple(io_buffers[num_inputs:])


def flatten(x: jax.Array, axis: int) -> jax.Array:
    return jnp.reshape(x, x.shape[:axis] + (math.prod(x.shape[axis:]),))


def sum_cat_list_list(
    operand: cue.SegmentedOperand,
    list_list: list[list[jax.Array]] | jax.Array,
    batch_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    if isinstance(list_list, jax.Array):
        x = list_list
        out = flatten(x, len(batch_shape))
        assert out.shape == batch_shape + (operand.size,)
        return out

    for sid, segments in enumerate(list_list):
        for x in segments:
            target_shape = batch_shape + operand[sid]
            assert jnp.broadcast_shapes(x.shape, target_shape) == target_shape
            assert x.dtype == dtype

    def sum(segments: list[jax.Array], size: int) -> jax.Array:
        if len(segments) == 0:
            return jnp.zeros(batch_shape + (size,), dtype)
        elif len(segments) == 1:
            return flatten(segments[0], len(batch_shape))
        else:
            return jnp.sum(
                jnp.stack([flatten(seg, len(batch_shape)) for seg in segments]), axis=0
            )

    out = jnp.concatenate(
        [
            sum(segments, math.prod(operand[sid]))
            for sid, segments in enumerate(list_list)
        ],
        axis=-1,
    )
    out = jnp.broadcast_to(out, batch_shape + (operand.size,))
    assert out.shape == batch_shape + (operand.size,)
    return out


def tp_list_list(
    *inputs: jax.Array,
    out_batch_shape: tuple[int, ...],
    d: cue.SegmentedTensorProduct,
    output_dtype: jnp.dtype,
    compute_dtype: jnp.dtype,
    precision: jax.lax.Precision,
    algorithm: str,
) -> list[list[jax.Array]]:
    broadcasted_batch_shape = jnp.broadcast_shapes(
        *[input.shape[:-1] for input in inputs]
    )
    out_batch_shape = tuple(
        min(b, c) for b, c in zip(broadcasted_batch_shape, out_batch_shape)
    )

    NAME = "CUEQUIVARIANCE_MAX_PATH_UNROLL"
    threshold_num_paths = int(os.environ.get(NAME, "1000"))
    if d.num_paths > threshold_num_paths and algorithm in ["sliced", "stacked"]:
        if d.all_same_segment_shape():
            warnings.warn(
                f"{d} has more than {threshold_num_paths} paths "
                f"(environment variable {NAME}), "
                f"switching algorithm from {algorithm} to compact_stacked."
            )
            algorithm = "compact_stacked"
        else:
            warnings.warn(
                f"{d} has more than {threshold_num_paths} paths "
                f"(environment variable {NAME})"
            )

    for ope, input in zip(d.operands, inputs):
        assert input.ndim == len(out_batch_shape) + 1
        assert input.shape[-1] == ope.size

    d = d.sort_paths(-1)
    pids = d.compressed_path_segment(-1)
    ope_out = d.operands[-1]
    ss_out = d.subscripts.operands[-1]

    def ein(
        coefficients: jax.Array, segments: list[jax.Array], mode: str = "normal"
    ) -> jax.Array:
        assert mode in ["normal", "accumulated", "vectorized"]
        if mode == "accumulated":
            path_in, path_out = "P", ""
        elif mode == "vectorized":
            path_in, path_out = "P", "P"
        else:
            path_in, path_out = "", ""

        batch_modes = "ABCDEFGHIJKLMNOQRSTUVWXYZ"[: len(out_batch_shape)]
        terms_in = [batch_modes + path_in + ss for ss in d.subscripts.operands[:-1]]
        term_out = (
            "".join(m for m, s in zip(batch_modes, out_batch_shape) if s != 1)
            + path_out
            + ss_out
        )
        terms = [path_in + d.coefficient_subscripts] + terms_in + [term_out]
        formula = ",".join(terms[:-1]) + "->" + terms[-1]
        segments = [x.astype(coefficients.dtype) for x in segments]

        segment = jnp.einsum(formula, coefficients, *segments, precision=precision)
        segment_shape = segment.shape[segment.ndim - len(ss_out) :]

        if mode == "vectorized":
            num_paths = coefficients.shape[0]
            output_segment_shape = out_batch_shape + (num_paths,) + segment_shape
        else:
            output_segment_shape = out_batch_shape + segment_shape

        segment = jnp.reshape(segment, output_segment_shape)
        return segment.astype(output_dtype)

    def prepare():
        if not d.all_same_segment_shape():
            raise ValueError("all operands must have the same segment shape\n" + str(d))
        reshaped_inputs = [
            jnp.reshape(
                input, input.shape[:-1] + (ope.num_segments,) + ope.segment_shape
            )
            for ope, input in zip(d.operands, inputs)
        ]
        indices = jnp.asarray(d.indices)
        coefficients = jnp.asarray(d.stacked_coefficients, dtype=compute_dtype)
        return reshaped_inputs, indices, coefficients

    if algorithm == "stacked":
        logger.debug(f"{d} with stacked strategy")

        reshaped_inputs, indices, coefficients = prepare()
        return [
            [
                ein(
                    coefficients[pid],
                    [
                        jnp.take(input, indices[pid, oid], axis=len(out_batch_shape))
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                )
                for pid in range(pid_start, pid_end)
            ]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]

    elif algorithm == "compact_stacked":
        logger.debug(f"{d} with compact_stacked strategy")

        reshaped_inputs, indices, coefficients = prepare()
        return [
            [
                ein(
                    coefficients[pid_start:pid_end],
                    [
                        jnp.take(
                            input,
                            indices[pid_start:pid_end, oid],
                            axis=len(out_batch_shape),
                        )
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                    mode="accumulated",
                )
            ]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]

    elif algorithm == "indexed_vmap":
        logger.debug(f"{d} with indexed_vmap strategy")

        reshaped_inputs, indices, coefficients = prepare()
        return (
            jnp.zeros(
                out_batch_shape + (ope_out.num_segments,) + ope_out.segment_shape,
                output_dtype,
            )
            .at[(slice(None),) * len(out_batch_shape) + (indices[:, -1],)]
            .add(
                jax.vmap(ein, (0, len(out_batch_shape)), len(out_batch_shape))(
                    coefficients,
                    [
                        jnp.take(input, indices[:, oid], axis=len(out_batch_shape))
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                ),
                indices_are_sorted=True,
                unique_indices=False,
            )
        )

    elif algorithm == "indexed_compact":
        logger.debug(f"{d} with indexed_compact strategy")

        reshaped_inputs, indices, coefficients = prepare()
        return (
            jnp.zeros(
                out_batch_shape + (ope_out.num_segments,) + ope_out.segment_shape,
                output_dtype,
            )
            .at[(slice(None),) * len(out_batch_shape) + (indices[:, -1],)]
            .add(
                ein(
                    coefficients,
                    [
                        jnp.take(input, indices[:, oid], axis=len(out_batch_shape))
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                    mode="vectorized",
                ),
                indices_are_sorted=True,
                unique_indices=False,
            )
        )

    elif algorithm == "indexed_for_loop":
        logger.debug(f"{d} with indexed_for_loop strategy")
        reshaped_inputs, indices, coefficients = prepare()

        def body(pid: int, output: jax.Array) -> jax.Array:
            return output.at[
                (slice(None),) * len(out_batch_shape) + (indices[pid, -1],)
            ].add(
                ein(
                    coefficients[pid],
                    [
                        jnp.take(input, indices[pid, oid], axis=len(out_batch_shape))
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                )
            )

        return jax.lax.fori_loop(
            0,
            d.num_paths,
            body,
            jnp.zeros(
                out_batch_shape + (ope_out.num_segments,) + ope_out.segment_shape,
                output_dtype,
            ),
        )

    elif algorithm == "sliced":
        logger.debug(f"{d} with sliced strategy")

        slices = [operand.segment_slices() for operand in d.operands]
        return [
            [
                ein(
                    jnp.asarray(path.coefficients, dtype=compute_dtype),
                    [
                        jnp.reshape(
                            jax.lax.slice_in_dim(
                                input,
                                slices[oid][path.indices[oid]].start,
                                slices[oid][path.indices[oid]].stop,
                                axis=len(out_batch_shape),
                            ),
                            input.shape[:-1] + d.get_segment_shape(oid, path),
                        )
                        for oid, input in enumerate(inputs)
                    ],
                )
                for path in d.paths[pid_start:pid_end]
            ]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]

    elif algorithm == "no-op":
        warnings.warn(f"{d} skipping computation!!!")

        dummy = sum([jnp.sum(input) for input in inputs])

        return [
            [
                jnp.zeros(out_batch_shape + d.get_segment_shape(-1, path), output_dtype)
                + dummy
                for path in d.paths[pid_start:pid_end]
            ]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]

    raise NotImplementedError(f"unknown algorithm {algorithm}")
