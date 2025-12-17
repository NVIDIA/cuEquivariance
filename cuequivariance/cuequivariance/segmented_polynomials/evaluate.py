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
from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

import cuequivariance as cue


def compute_last_operand(
    descriptor: cue.SegmentedTensorProduct,
    *inputs: np.ndarray,
    segment_axes: Union[int, list[int]] = -1,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    r"""
    Compute the last operand of a segmented tensor product.

    Args:
        descriptor (SegmentedTensorProduct): The descriptor of the segmented tensor product.
        inputs (np.ndarray): The operands of the segmented tensor product.
        segment_axes (int or list of int, optional): The axes along which to segment the inputs, by default -1.

    Returns:
        np.ndarray: The last operand of the segmented tensor product.
    """
    if len(inputs) != descriptor.num_operands - 1:
        raise ValueError(
            f"Expected {descriptor.num_operands - 1} inputs, got {len(inputs)}."
        )

    if isinstance(segment_axes, int):
        segment_axes = [segment_axes] * descriptor.num_operands

    segment_axes = [k % input.ndim for k, input in zip(segment_axes, inputs)] + [
        segment_axes[-1]
    ]
    assert len(segment_axes) == descriptor.num_operands

    for i, (operand, input, k) in enumerate(
        zip(descriptor.operands[:-1], inputs, segment_axes)
    ):
        if operand.size != input.shape[k]:
            raise ValueError(
                f"Expected operand {i} to have size {operand.size}, got {input.shape[k]}."
            )

    batch_shapes = [
        input.shape[:k] + input.shape[k + 1 :] for input, k in zip(inputs, segment_axes)
    ]
    output_batch_shape = np.broadcast_shapes(*batch_shapes)

    segment_slices = [operand.segment_slices() for operand in descriptor.operands]
    return primitive_compute_last_operand(
        descriptor.subscripts.operands,
        descriptor.subscripts.coefficients,
        [operand.segments for operand in descriptor.operands],
        output_batch_shape,
        [[slice_.start for slice_ in segment] for segment in segment_slices],
        [
            [slice_.stop - slice_.start for slice_ in segment]
            for segment in segment_slices
        ],
        [path.indices for path in descriptor.paths],
        [path.coefficients for path in descriptor.paths],
        segment_axes,
        dtype or np.result_type(*inputs),
        *inputs,
    )


def primitive_compute_last_operand(
    operand_subscripts: list[str],  # ['uvw', 'iu', 'jv', 'kw']
    coefficient_subscripts: str,  # 'ijk'
    segment_shapes: list[list[tuple[int, ...]]],  # segment_shapes[oid][sid] = shape
    output_batch_shape: tuple[int, ...],
    segment_offsets: list[list[slice]],  # segment_offsets[oid][sid] = offset
    segment_sizes: list[list[int]],  # segment_sizes[oid][sid] = size
    indices: list[tuple[int, ...]],  # indices[pid][oid] = sid
    coefficients: Union[list[np.ndarray], np.ndarray],  # coefficients[pid] = coeff
    segment_axes: list[int],  # segment_axes[oid] = axis
    dtype,
    *inputs: np.ndarray,
) -> np.ndarray:
    num_operands = len(operand_subscripts)
    inputs_subscripts = ",".join("..." + s for s in operand_subscripts[:-1])
    if inputs_subscripts:
        formula = (
            f"{coefficient_subscripts},{inputs_subscripts}->...{operand_subscripts[-1]}"
        )
    else:
        formula = f"{coefficient_subscripts}->{operand_subscripts[-1]}"

    output = np.zeros(
        output_batch_shape + (sum(segment_sizes[-1]),),
        dtype=dtype,
    )

    inputs = [np.moveaxis(input, k, -1) for input, k in zip(inputs, segment_axes)]

    for idxx, coeff in zip(indices, coefficients):
        slices = [
            slice(
                segment_offsets[oid][idxx[oid]],
                segment_offsets[oid][idxx[oid]] + segment_sizes[oid][idxx[oid]],
            )
            for oid in range(num_operands)
        ]
        output[..., slices[-1]] += np.reshape(
            np.einsum(
                formula,
                coeff,
                *[
                    np.reshape(
                        inputs[oid][..., slices[oid]],
                        inputs[oid].shape[:-1] + segment_shapes[oid][idxx[oid]],
                    )
                    for oid in range(num_operands - 1)
                ],
            ),
            output_batch_shape + (math.prod(segment_shapes[-1][idxx[-1]]),),
        )

    return np.moveaxis(output, -1, segment_axes[-1])
