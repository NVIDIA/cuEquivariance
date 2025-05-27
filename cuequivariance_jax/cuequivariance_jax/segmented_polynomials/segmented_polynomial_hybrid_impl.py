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

import cuequivariance as cue
from cuequivariance_jax.segmented_polynomials.indexing_mode import IndexingMode
from cuequivariance_jax.segmented_polynomials.utils import batch_size, indexing

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Buffer:
    data: jax.Array
    bi: list[int]  # buffer index
    mode: list[IndexingMode]


def segmented_polynomial_hybrid_impl(
    inputs: list[jax.Array],  # shape (*batch_sizes, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    buffer_index: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    impl: str,
    name: str,
) -> list[jax.Array]:  # output buffers
    assert impl in ("cuda", "auto", "jax")

    num_inputs = len(buffer_index) - len(outputs_shape_dtype)

    io_buffers = list(inputs) + [
        jnp.zeros(out.shape, out.dtype) for out in outputs_shape_dtype
    ]
    buffer_index = np.array(buffer_index, dtype=np.int32)
    num_batch_axes = buffer_index.shape[1]
    batch_sizes = [
        batch_size(
            [x.shape[i] for x, idx in zip(io_buffers, buffer_index[:, i]) if idx < 0],
        )
        for i in range(num_batch_axes)
    ]

    for operation, d in polynomial.operations:
        ope_out, b_out = operation.output_operand_buffer(num_inputs)

        out = outputs_shape_dtype[b_out - num_inputs]

        output_segments: list[list[jax.Array]] = tp_list_list(
            [
                Buffer(inputs[i], buffer_index[i], index_mode[i])
                for i in operation.input_buffers(num_inputs)
            ],
            Buffer(out, buffer_index[b_out], index_mode[b_out]),
            indices,
            batch_sizes=batch_sizes,
            d=d.move_operand_last(ope_out),
            math_dtype=math_dtype,
            precision=jax.lax.Precision.HIGHEST,
            impl=impl,
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
    math_dtype: jnp.dtype,
    precision: jax.lax.Precision,
    impl: str,
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
                precision,
                math_dtype,
                impl=impl,
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
    precision: jax.lax.Precision,
    math_dtype: jnp.dtype,
    impl: str,
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

    if modes == (
        (IndexingMode.BATCHED_OR_SHARED,),
        (IndexingMode.REPEATED,),
        (IndexingMode.BATCHED_OR_SHARED,),
    ):
        return stuff1(
            formula,
            coefficients.item(),
            segments[0].data,
            segments[1].data,
            indices[segments[1].bi[0]],
        )
    if modes == (
        (IndexingMode.REPEATED,),
        (IndexingMode.BATCHED_OR_SHARED,),
        (IndexingMode.BATCHED_OR_SHARED,),
    ):
        assert num_batch_axes == 1
        assert coefficient_subscripts == ""
        b, a, c = subscripts
        [a, b, c] = (
            cue.segmented_polynomials.Subscripts.from_operands([a, b, c])
            .canonicalize()
            .operands
        )
        return stuff1(
            f",A{a},A{b}->A{c}",
            coefficients.item(),
            segments[1].data,
            segments[0].data,
            indices[segments[0].bi[0]],
        )
    if modes == (
        (IndexingMode.BATCHED_OR_SHARED,),
        (IndexingMode.BATCHED_OR_SHARED,),
        (IndexingMode.REPEATED,),
    ):
        return stuff2(
            formula,
            coefficients.item(),
            segments[0].data,
            segments[1].data,
            indices[output.bi[0]],
        )

    segments_data = [
        scatter(x.data, x.bi, x.mode, indices, batch_sizes) for x in segments
    ]
    coeffs = jnp.array(coefficients, dtype=math_dtype)
    segments_data = [x.astype(math_dtype) for x in segments_data]
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


def ragged_dot_transpose(a, c, i):
    """
    Transpose of jax.lax.ragged_dot(a, b, i) == c w.r.t. b.

    a: (batch_size, m)
    c: (batch_size, n)
    i: (num_groups,)
    b: (num_groups, m, n)
    """
    dn = jax.lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(([0], [0]), ([], [])),
        lhs_ragged_dimensions=[0],
        rhs_group_dimensions=[],
    )
    return jax.lax.ragged_dot_general(a, c, i, dn)


def stuff1(
    formula: str, co: float, a: jax.Array, b: jax.Array, i: jax.Array
) -> jax.Array:
    print("hello from stuff1")
    # b is the set of matrices
    F = jax.lax.ragged_dot
    tr = lambda x: jnp.transpose(x, (0, 2, 1))  # noqa

    if formula == ",Au,Auv->Av":  # 1
        return co * F(a, b, i)
    if formula == ",Au,Avu->Av":  # 2
        return co * F(a, tr(b), i)
    if formula == ",Auv,Avw->Auw":  # 3, same as 1 with an extra u
        (A, u, v) = a.shape
        (_, v, w) = b.shape
        return co * F(a.reshape(A * u, v), b, i * u).reshape(A, u, w)
    if formula == ",Auv,Awv->Auw":
        (A, u, v) = a.shape
        (_, w, v) = b.shape
        return co * F(a.reshape(A * u, v), tr(b), i * u).reshape(A, u, w)

    raise NotImplementedError(
        f"Unsupported formula: {formula} with co={co}, a.shape={a.shape}, b.shape={b.shape}, i.shape={i.shape}"
    )


def stuff2(formula: str, co: float, a: jax.Array, b: jax.Array, i: jax.Array):
    print("hello from stuff2")
    # output is the set of matrices
    F = ragged_dot_transpose
    tr = lambda x: jnp.transpose(x, (0, 2, 1))  # noqa

    if formula == ",Au,Av->Auv":
        return co * F(a, b, i)
    if formula == ",Au,Av->Avu":
        return co * tr(F(a, b, i))
    if formula == ",Auv,Auw->Avw":
        (A, u, v) = a.shape
        (A, u, w) = b.shape
        return co * F(a.reshape(A * u, v), b.reshape(A * u, w), i * u)
    if formula == ",Auv,Auw->Awv":
        (A, u, v) = a.shape
        (A, u, w) = b.shape
        return co * tr(F(a.reshape(A * u, v), b.reshape(A * u, w), i * u))

    raise NotImplementedError(
        f"Unsupported formula: {formula} with co={co}, a.shape={a.shape}, b.shape={b.shape}, i.shape={i.shape}"
    )
