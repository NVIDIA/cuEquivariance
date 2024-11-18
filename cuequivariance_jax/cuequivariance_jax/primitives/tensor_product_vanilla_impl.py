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
import math
import os
import warnings
from typing import *

import jax
import jax.lax
import jax.numpy as jnp

from cuequivariance import segmented_tensor_product as stp
from cuequivariance.tensor_product_execution import TensorProductExecution

logger = logging.getLogger(__name__)


def tensor_product_vanilla_impl(
    *inputs: jax.Array,  # input buffers
    shapes: tuple[tuple[int, ...], ...],  # shapes of the operands
    d: stp.SegmentedTensorProduct,
    exe: TensorProductExecution,
    **options,
) -> tuple[jax.Array, ...]:  # output buffers
    assert exe.max_out_buffer + 1 == len(exe.out_buffers)

    outputs = [0] * len(exe.out_buffers)

    for c in exe.computations:
        out = sum_cat_list_list(
            d.operands[c.out_operand],
            tp_list_list(
                *c.map_inputs(inputs),
                shape=shapes[c.out_operand],
                d=d.move_operand_last(c.out_operand),
                **options,
            ),
            shapes[c.out_operand],
            options["dtype_output"],
        )
        assert out.shape == shapes[c.out_operand] + (d.operands[c.out_operand].size,)
        outputs[c.out_buffer] += out

    return tuple(outputs)


def flatten(x: jax.Array, axis: int) -> jax.Array:
    return jnp.reshape(x, x.shape[:axis] + (math.prod(x.shape[axis:]),))


def sum_cat_list_list(
    operand: stp.Operand,
    list_list: list[list[jax.Array]] | jax.Array,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    if isinstance(list_list, jax.Array):
        x = list_list
        out = flatten(x, len(shape))
        assert out.shape == shape + (operand.size,)
        return out

    for sid, segments in enumerate(list_list):
        for x in segments:
            assert x.shape == shape + operand[sid]
            assert x.dtype == dtype

    def sum(segments: list[jax.Array], size: int) -> jax.Array:
        if len(segments) == 0:
            return jnp.zeros(shape + (size,), dtype)
        elif len(segments) == 1:
            return flatten(segments[0], len(shape))
        else:
            return jnp.sum(
                jnp.stack([flatten(seg, len(shape)) for seg in segments]), axis=0
            )

    out = jnp.concatenate(
        [
            sum(segments, math.prod(operand[sid]))
            for sid, segments in enumerate(list_list)
        ],
        axis=-1,
    )
    assert out.shape == shape + (operand.size,)
    return out


def tp_list_list(
    *inputs: jax.Array,
    shape: tuple[int, ...],
    d: stp.SegmentedTensorProduct,
    dtype_output: jnp.dtype,
    dtype_math: jnp.dtype,
    precision: jax.lax.Precision,
    algorithm: str,
    **_options,
) -> list[list[jax.Array]]:
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
        assert input.ndim == len(shape) + 1
        assert input.shape[-1] == ope.size

    d = d.sort_paths(-1)
    pids = d.compressed_path_segment(-1)
    ope_out = d.operands[-1]

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

        batch_modes = "ABCDEFGHIJKLMNOQRSTUVWXYZ"[: len(shape)]
        terms_in = [batch_modes + path_in + ss for ss in d.subscripts.operands[:-1]]
        term_out = (
            "".join(m for m, s in zip(batch_modes, shape) if s > 1)
            + path_out
            + ope_out.subscripts
        )
        terms = [path_in + d.coefficient_subscripts] + terms_in + [term_out]
        formula = ",".join(terms[:-1]) + "->" + terms[-1]
        segments = [x.astype(coefficients.dtype) for x in segments]

        segment = jnp.einsum(formula, coefficients, *segments, precision=precision)
        segment_shape = segment.shape[segment.ndim - len(ope_out.subscripts) :]

        if mode == "vectorized":
            num_paths = coefficients.shape[0]
            output_segment_shape = shape + (num_paths,) + segment_shape
        else:
            output_segment_shape = shape + segment_shape

        segment = jnp.reshape(segment, output_segment_shape)
        return segment.astype(dtype_output)

    def prepare():
        if not d.all_same_segment_shape():
            raise ValueError(
                "cuex.tensor_product: all operands must have the same segment shape\n"
                + str(d)
            )
        reshaped_inputs = [
            jnp.reshape(
                input, input.shape[:-1] + (ope.num_segments,) + ope.segment_shape
            )
            for ope, input in zip(d.operands, inputs)
        ]
        indices = jnp.asarray(d.indices)
        coefficients = jnp.asarray(d.stacked_coefficients, dtype=dtype_math)
        return reshaped_inputs, indices, coefficients

    if algorithm == "stacked":
        logger.debug(f"cuex.tensor_product: {d} with stacked strategy")

        reshaped_inputs, indices, coefficients = prepare()
        return [
            [
                ein(
                    coefficients[pid],
                    [
                        jnp.take(input, indices[pid, oid], axis=len(shape))
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                )
                for pid in range(pid_start, pid_end)
            ]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]

    elif algorithm == "compact_stacked":
        logger.debug(f"cuex.tensor_product: {d} with compact_stacked strategy")

        reshaped_inputs, indices, coefficients = prepare()
        return [
            [
                ein(
                    coefficients[pid_start:pid_end],
                    [
                        jnp.take(
                            input, indices[pid_start:pid_end, oid], axis=len(shape)
                        )
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                    mode="accumulated",
                )
            ]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]

    elif algorithm == "indexed_vmap":
        logger.debug(f"cuex.tensor_product: {d} with indexed_vmap strategy")

        reshaped_inputs, indices, coefficients = prepare()
        return (
            jnp.zeros(
                shape + (ope_out.num_segments,) + ope_out.segment_shape, dtype_output
            )
            .at[(slice(None),) * len(shape) + (indices[:, -1],)]
            .add(
                jax.vmap(ein, (0, len(shape)), len(shape))(
                    coefficients,
                    [
                        jnp.take(input, indices[:, oid], axis=len(shape))
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                ),
                indices_are_sorted=True,
                unique_indices=False,
            )
        )

    elif algorithm == "indexed_compact":
        logger.debug(f"cuex.tensor_product: {d} with indexed_compact strategy")

        reshaped_inputs, indices, coefficients = prepare()
        return (
            jnp.zeros(
                shape + (ope_out.num_segments,) + ope_out.segment_shape, dtype_output
            )
            .at[(slice(None),) * len(shape) + (indices[:, -1],)]
            .add(
                ein(
                    coefficients,
                    [
                        jnp.take(input, indices[:, oid], axis=len(shape))
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                    mode="vectorized",
                ),
                indices_are_sorted=True,
                unique_indices=False,
            )
        )

    elif algorithm == "indexed_for_loop":
        logger.debug(f"cuex.tensor_product: {d} with indexed_for_loop strategy")
        reshaped_inputs, indices, coefficients = prepare()

        def body(pid: int, output: jax.Array) -> jax.Array:
            return output.at[(slice(None),) * len(shape) + (indices[pid, -1],)].add(
                ein(
                    coefficients[pid],
                    [
                        jnp.take(input, indices[pid, oid], axis=len(shape))
                        for oid, input in enumerate(reshaped_inputs)
                    ],
                )
            )

        return jax.lax.fori_loop(
            0,
            d.num_paths,
            body,
            jnp.zeros(
                shape + (ope_out.num_segments,) + ope_out.segment_shape, dtype_output
            ),
        )

    elif algorithm == "sliced":
        logger.debug(f"cuex.tensor_product: {d} with sliced strategy")

        slices = [operand.segment_slices() for operand in d.operands]
        return [
            [
                ein(
                    jnp.asarray(path.coefficients, dtype=dtype_math),
                    [
                        jnp.reshape(
                            jax.lax.slice_in_dim(
                                input,
                                slices[oid][path.indices[oid]].start,
                                slices[oid][path.indices[oid]].stop,
                                axis=len(shape),
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
        warnings.warn(f"cuex.tensor_product: {d} skipping computation!!!")

        dummy = sum([jnp.sum(input) for input in inputs])

        return [
            [
                jnp.zeros(shape + d.get_segment_shape(-1, path), dtype_output) + dummy
                for path in d.paths[pid_start:pid_end]
            ]
            for pid_start, pid_end in zip(pids[:-1], pids[1:])
        ]

    raise NotImplementedError(f"cuex.tensor_product: unknown algorithm {algorithm}")
