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
from dataclasses import dataclass

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from cuequivariance_jax.segmented_polynomials.indexing_mode import IndexingMode
from cuequivariance_jax.segmented_polynomials.utils import (
    batch_size,
    indexing,
    math_dtype_for_naive_method,
)

import cuequivariance as cue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Buffer:
    data: jax.Array
    bi: list[int]  # buffer index
    mode: list[IndexingMode]


def execute_indexed_linear(
    inputs: list[jax.Array],  # shape (*batch_sizes, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    index_configuration: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
    run_kernel: bool = True,
) -> list[jax.Array]:  # output buffers
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

    for operation, d in polynomial.operations:
        ope_out, b_out = operation.output_operand_buffer(num_inputs)

        out = outputs_shape_dtype[b_out - num_inputs]

        output_segments: list[list[jax.Array]] = tp_list_list(
            [
                Buffer(inputs[i], index_configuration[i], index_mode[i])
                for i in operation.input_buffers(num_inputs)
            ],
            Buffer(out, index_configuration[b_out], index_mode[b_out]),
            indices,
            batch_sizes=batch_sizes,
            d=d.move_operand_last(ope_out),
            math_dtype=math_dtype,
            run_kernel=run_kernel,
        )
        out = sum_cat_list_list(
            d.operands[ope_out],
            output_segments,
            out.shape[:-1],
            out.dtype,
        )
        io_buffers[b_out] += out

    return tuple(io_buffers[num_inputs:])


def flatten(x: jax.Array, axis: int) -> jax.Array:
    return jnp.reshape(x, x.shape[:axis] + (math.prod(x.shape[axis:]),))


def sum_cat_list_list(
    operand: cue.SegmentedOperand,
    list_list: list[list[jax.Array]],
    batch_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
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
    inputs: list[Buffer],
    output: Buffer,
    indices: list[jax.Array],
    batch_sizes: list[int],
    d: cue.SegmentedTensorProduct,
    math_dtype: str | None,
    run_kernel: bool,
) -> list[list[jax.Array]]:
    num_batch_axes = len(batch_sizes)

    for ope, input in zip(d.operands, inputs):
        assert input.data.ndim == num_batch_axes + 1
        assert input.data.shape[-1] == ope.size

    d = d.sort_paths(-1)
    pids = d.compressed_path_segment(-1)

    slices = [operand.segment_slices() for operand in d.operands]
    return [
        [
            ein(
                path.coefficients,
                [
                    Buffer(
                        jnp.reshape(
                            jax.lax.slice_in_dim(
                                input.data,
                                slices[oid][path.indices[oid]].start,
                                slices[oid][path.indices[oid]].stop,
                                axis=num_batch_axes,
                            ),
                            input.data.shape[:-1] + d.get_segment_shape(oid, path),
                        ),
                        input.bi,
                        input.mode,
                    )
                    for oid, input in enumerate(inputs)
                ],
                Buffer(
                    jnp.zeros(
                        output.data.shape[:-1] + d.get_segment_shape(-1, path),
                        output.data.dtype,
                    ),
                    output.bi,
                    output.mode,
                ),
                indices,
                d.subscripts.operands,
                d.coefficient_subscripts,
                batch_sizes,
                math_dtype,
                run_kernel=run_kernel,
            )
            for path in d.paths[pid_start:pid_end]
        ]
        for pid_start, pid_end in zip(pids[:-1], pids[1:])
    ]


def ein(
    coefficients: np.ndarray,
    segments: list[Buffer],
    output: Buffer,
    indices: list[jax.Array],
    subscripts: list[str],
    coefficient_subscripts: str,
    batch_sizes: list[int],
    math_dtype: str | None,
    run_kernel: bool,
) -> jax.Array:
    num_batch_axes = len(batch_sizes)
    batch_modes = "ABCDEFGHIJKLMNOQRSTUVWXYZ"[:num_batch_axes]
    terms_in = [batch_modes + ss for ss in subscripts[:-1]]
    term_out = (
        "".join(m for m, s in zip(batch_modes, output.data.shape) if s != 1)
        + subscripts[-1]
    )
    terms = [coefficient_subscripts] + terms_in + [term_out]
    formula = ",".join(terms[:-1]) + "->" + terms[-1]
    modes = tuple([x.mode for x in segments] + [output.mode])

    if (
        run_kernel
        and formula in (",Auv,Au->Av", ",Auv,Av->Au")
        and modes
        == (
            (IndexingMode.REPEATED,),
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.BATCHED_OR_SHARED,),
        )
    ):
        from cuequivariance_ops_jax import indexed_linear

        counts = indices[segments[0].bi[0]]
        (C, u, v) = segments[0].data.shape
        (Z, _) = segments[1].data.shape
        assert counts.shape == (C,)
        return indexed_linear(
            segments[0].data,  # Cuv
            segments[1].data,  # Zu or Zv
            output.data,  # Zv or Zu
            counts,
            u,
            v,
            C,
            Z,
            {",Auv,Au->Av": ("uv", "u", "v"), ",Auv,Av->Au": ("uv", "v", "u")}[formula],
            coefficients.item(),
            math_dtype,
        )

    elif (
        run_kernel
        and formula in (",Auv,Awu->Awv", ",Auv,Awv->Awu")
        and modes
        == (
            (IndexingMode.REPEATED,),
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.BATCHED_OR_SHARED,),
        )
    ):
        from cuequivariance_ops_jax import indexed_linear

        counts = indices[segments[0].bi[0]]
        (C, u, v) = segments[0].data.shape
        (Z, w, _) = segments[1].data.shape
        assert counts.shape == (C,)
        return indexed_linear(
            segments[0].data,  # Cuv
            segments[1].data,  # Zwu or Zwv
            output.data,  # Zwv or Zwu
            counts * w,
            u,
            v,
            C,
            Z * w,
            {",Auv,Awu->Awv": ("uv", "u", "v"), ",Auv,Awv->Awu": ("uv", "v", "u")}[
                formula
            ],
            coefficients.item(),
            math_dtype,
        )

    elif (
        run_kernel
        and formula in (",Au,Auv->Av", ",Au,Avu->Av")
        and modes
        == (
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.REPEATED,),
            (IndexingMode.BATCHED_OR_SHARED,),
        )
    ):
        from cuequivariance_ops_jax import indexed_linear

        counts = indices[segments[1].bi[0]]
        (Z, u) = segments[0].data.shape
        (C, _, _) = segments[1].data.shape
        (Z, v) = output.data.shape
        assert counts.shape == (C,)
        return indexed_linear(
            segments[0].data,  # Zu
            segments[1].data,  # Cuv or Cvu
            output.data,  # Zv
            counts,
            u,
            v,
            C,
            Z,
            {",Au,Auv->Av": ("u", "uv", "v"), ",Au,Avu->Av": ("u", "vu", "v")}[formula],
            coefficients.item(),
            math_dtype,
        )

    elif (
        run_kernel
        and formula in (",Auv,Awv->Auw", ",Auv,Avw->Auw")
        and modes
        == (
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.REPEATED,),
            (IndexingMode.BATCHED_OR_SHARED,),
        )
    ):
        from cuequivariance_ops_jax import indexed_linear

        counts = indices[segments[1].bi[0]]
        (Z, w, u) = segments[0].data.shape
        (C, _, _) = segments[1].data.shape
        (Z, w, v) = output.data.shape
        assert counts.shape == (C,)
        return indexed_linear(
            segments[0].data,  # Zwu
            segments[1].data,  # Cvu
            output.data,  # Zwv
            counts * w,
            u,
            v,
            C,
            Z * w,
            {",Auv,Awv->Auw": ("u", "vu", "v"), ",Auv,Avw->Auw": ("u", "uv", "v")}[
                formula
            ],
            coefficients.item(),
            math_dtype,
        )

    elif (
        run_kernel
        and formula in (",Au,Av->Avu", ",Au,Av->Auv")
        and modes
        == (
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.REPEATED,),
        )
    ):
        from cuequivariance_ops_jax import indexed_linear

        counts = indices[output.bi[0]]
        (Z, u) = segments[0].data.shape
        (Z, v) = segments[1].data.shape
        (C, _, _) = output.data.shape
        assert counts.shape == (C,), f"{counts.shape=}, {C=}"
        return indexed_linear(
            segments[0].data,  # Zu
            segments[1].data,  # Zv
            output.data,  # Cvu or Cuv
            counts,
            u,
            v,
            C,
            Z,
            {",Au,Av->Avu": ("u", "v", "vu"), ",Au,Av->Auv": ("u", "v", "uv")}[formula],
            coefficients.item(),
            math_dtype,
        )

    elif (
        run_kernel
        and formula in (",Auv,Auw->Awv", ",Auv,Auw->Avw")
        and modes
        == (
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.BATCHED_OR_SHARED,),
            (IndexingMode.REPEATED,),
        )
    ):
        from cuequivariance_ops_jax import indexed_linear

        counts = indices[output.bi[0]]
        (Z, w, u) = segments[0].data.shape
        (Z, w, v) = segments[1].data.shape
        (C, _, _) = output.data.shape
        assert counts.shape == (C,)
        return indexed_linear(
            segments[0].data,  # Zwu
            segments[1].data,  # Zwv
            output.data,  # Cuv or Cvu
            counts * w,
            u,
            v,
            C,
            Z * w,
            {",Auv,Auw->Awv": ("u", "v", "vu"), ",Auv,Auw->Avw": ("u", "v", "uv")}[
                formula
            ],
            coefficients.item(),
            math_dtype,
        )

    if run_kernel:
        raise ValueError("It was not possible to execute the method indexed_linear.")

    compute_dtype, precision = math_dtype_for_naive_method(
        jnp.result_type(*[x.data.dtype for x in segments], output.data.dtype),
        math_dtype,
    )

    segments_data = [
        scatter(x.data, x.bi, x.mode, indices, batch_sizes) for x in segments
    ]
    coeffs = jnp.array(coefficients, dtype=compute_dtype)
    segments_data = [x.astype(compute_dtype) for x in segments_data]
    segment = jnp.einsum(formula, coeffs, *segments_data, precision=precision)
    segment = segment.astype(output.data.dtype)
    return gather(output.data, segment, output.bi, output.mode, indices)


def scatter(
    x: jax.Array,
    bi: list[int],
    modes: list[IndexingMode],
    indices: list[jax.Array],
    batch_sizes: list[int],
) -> jax.Array:
    if all(i < 0 for i in bi):
        return x

    if modes == (IndexingMode.REPEATED,):
        counts = indices[bi[0]]
        return jnp.repeat(x, counts, axis=0, total_repeat_length=batch_sizes[0])

    assert all(
        mode in [IndexingMode.BATCHED_OR_SHARED, IndexingMode.INDEXED] for mode in modes
    )
    idx = indexing(bi, x.shape, indices)
    return x[idx]


def gather(
    output: jax.Array,
    x: jax.Array,
    bi: list[int],
    modes: list[IndexingMode],
    indices: list[jax.Array],
) -> jax.Array:
    if all(i < 0 for i in bi):
        return x
    if modes == (IndexingMode.REPEATED,):
        counts = indices[bi[0]]
        i = jnp.cumsum(jnp.append(0, counts[:-1]))
        return jnp.add.reduceat(x, i)

    assert all(
        mode in [IndexingMode.BATCHED_OR_SHARED, IndexingMode.INDEXED] for mode in modes
    )
    idx = indexing(bi, x.shape, indices)
    return output.at[idx].add(x)
