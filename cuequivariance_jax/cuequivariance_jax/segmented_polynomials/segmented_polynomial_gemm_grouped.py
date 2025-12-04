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
import math

import jax
import jax.numpy as jnp
import numpy as np

import cuequivariance as cue


def _prepand_batch_modes(operand_subscript_pair):
    array, subscript = operand_subscript_pair
    batch_shape = array.shape[: -len(subscript)] if subscript else array.shape

    batch_modes = ""
    for i, size in enumerate(batch_shape):
        if size == 1:
            batch_modes += "1"
        else:
            # Use letters A, B, C, ... for batch dimensions
            batch_modes += chr(ord("A") + i)

    return array, batch_modes + subscript


def _squeeze_modes(operand_subscript_pair):
    array, subscript = operand_subscript_pair

    # Find positions of '1' in the subscript
    squeeze_axes = []
    new_subscript = ""

    for i, char in enumerate(subscript):
        if char == "1":
            squeeze_axes.append(i)
        else:
            new_subscript += char

    # Squeeze the array at the identified axes
    squeezed_array = array
    for axis in reversed(squeeze_axes):  # Reverse to maintain correct indices
        squeezed_array = jnp.squeeze(squeezed_array, axis=axis)

    return squeezed_array, new_subscript


def _consolidate_pairs(operands):
    if not operands:
        return operands

    # Find all consecutive character pairs across all subscripts
    all_pairs = set()
    for _, subscript in operands:
        for i in range(len(subscript) - 1):
            all_pairs.add(subscript[i : i + 2])

    # Find a pair that can be consolidated (appears in all relevant subscripts)
    for pair in all_pairs:
        char1, char2 = pair
        if all(
            pair in sub or (char1 not in sub and char2 not in sub)
            for _, sub in operands
        ):
            # Consolidate this pair
            new_operands = []
            for array, subscript in operands:
                if pair in subscript:
                    pos = subscript.index(pair)
                    # Combine dimensions at pos and pos+1
                    new_shape = list(array.shape)
                    new_shape[pos] *= new_shape[pos + 1]
                    new_shape.pop(pos + 1)
                    array = jnp.reshape(array, new_shape)
                    subscript = subscript.replace(pair, char1)
                new_operands.append((array, subscript))
            return _consolidate_pairs(new_operands)

    return operands


def execute_gemm_grouped(
    inputs: list[jax.Array],  # shape (*batch_sizes, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    index_configuration: tuple[tuple[int, ...], ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
) -> list[jax.Array]:
    index_configuration = np.array(index_configuration)
    num_batch_axes = index_configuration.shape[1]
    assert (
        polynomial.num_inputs + len(outputs_shape_dtype) == index_configuration.shape[0]
    )
    assert polynomial.num_outputs == len(outputs_shape_dtype)

    assert math_dtype is None

    if not all(x.dtype in {jnp.int32, jnp.int64} for x in indices):
        raise ValueError("All indices must have dtype int32 or int64")

    from cuequivariance_ops_jax import gemm_grouped

    # index_configuration = np.concatenate(
    #     [index_configuration, np.full((len(indices), num_batch_axes), -1, np.int32)]
    # )

    if not np.all(index_configuration == -1):
        raise ValueError("method 'gemm_grouped' does not support indices (yet)")
    if len(indices) != 0:
        raise ValueError("method 'gemm_grouped' does not support indices (yet)")

    gemms = []

    nin = polynomial.num_inputs
    for ope, stp in polynomial.operations:
        assert stp.num_operands == 3, (
            f"method 'gemm_grouped' requires STPs with 3 operands, got {stp.num_operands} for {ope}"
        )
        assert stp.coefficient_subscripts == "", (
            f"method 'gemm_grouped' requires STPs without coefficient subscripts, got {stp.coefficient_subscripts} for {ope}"
        )
        oid, i = ope.output_operand_buffer(nin)
        [AA, BB] = [inputs[i] for i in ope.input_buffers(nin)]
        CC = outputs_shape_dtype[i - nin]
        stp = stp.move_operand_last(oid)

        Aslices = stp.operands[0].segment_slices()
        Bslices = stp.operands[1].segment_slices()

        for path in stp.paths:
            A = AA[..., Aslices[path.indices[0]]]
            B = BB[..., Bslices[path.indices[1]]]

            A = jnp.reshape(A, A.shape[:-1] + stp.operands[0].segments[path.indices[0]])
            B = jnp.reshape(B, B.shape[:-1] + stp.operands[1].segments[path.indices[1]])
            C_shape = CC.shape[:-1] + stp.operands[2].segments[path.indices[2]]
            C = jnp.zeros(C_shape, dtype=CC.dtype)

            sa, sb, sc = stp.subscripts.operands
            assert A.ndim == num_batch_axes + len(sa)
            assert B.ndim == num_batch_axes + len(sb)
            assert C.ndim == num_batch_axes + len(sc)

            operands = [(A, sa), (B, sb), (C, sc)]
            operands = list(map(_prepand_batch_modes, operands))
            operands = list(map(_squeeze_modes, operands))
            operands = _consolidate_pairs(operands)

            [(A, sa), (B, sb), (C, sc)] = operands

            if len(sc) >= 2:
                u, v = sc[-2:]
                if u in sb and v in sa:
                    [(A, sa), (B, sb)] = [(B, sb), (A, sa)]
            if len(sc) == 1:
                if len(sa) == 2 and len(sb) == 1:
                    [(A, sa), (B, sb)] = [(B, sb), (A, sa)]

            [sa, sb, sc] = (
                cue.segmented_polynomials.Subscripts.from_operands([sa, sb, sc])
                .canonicalize()
                .operands
            )
            contr = f"{sa},{sb}->{sc}"

            gemm = None

            if contr == "uvw,uav->uwa":
                gemm = (A, B, True, True)
            if contr == "uvw,uwa->uva":
                gemm = (A, B, False, False)

            if contr == "uv,vw->uw":
                gemm = (A, B, False, False)
            if contr == "uv,wv->uw":
                gemm = (A, B, False, True)
            if contr == "uv,uw->vw":
                gemm = (A, B, True, False)
            if contr == "uv,wu->vw":
                gemm = (A, B, True, True)

            if contr == "u,uv->v":
                gemm = (A[None, :], B, False, False)
            if contr == "u,vu->v":
                gemm = (A[None, :], B, False, True)

            if contr == "u,v->uv":
                gemm = (A[:, None], B[None, :], False, False)

            if gemm is None:
                raise ValueError(
                    f"gemm_grouped does not support: {A.shape} @ {B.shape} -> {C.shape} with contraction {sa},{sb}->{sc}"
                )
            gemms.append(gemm + (path.coefficients.item(),))

    num_batch_axes = {A.ndim - 2 for A, _, _, _, _ in gemms}
    assert len(num_batch_axes) == 1
    num_batch_axes = num_batch_axes.pop()
    gemm_outs = gemm_grouped(
        gemms,
        [],
        np.full((2 * len(gemms), num_batch_axes), -1, np.int32),
        use_tf32=False,
    )
    outputs = [jnp.zeros(x.shape, dtype=x.dtype) for x in outputs_shape_dtype]

    for ope, stp in polynomial.operations:
        oid, i = ope.output_operand_buffer(nin)
        slices = stp.operands[oid].segment_slices()
        segments = stp.operands[oid].segments

        for path in stp.paths:
            sid = path.indices[oid]
            acc = outputs[i - nin]
            outputs[i - nin] = acc.at[..., slices[sid]].add(
                jnp.reshape(
                    gemm_outs.pop(0), acc.shape[:-1] + (math.prod(segments[sid]),)
                )
            )
    return outputs
