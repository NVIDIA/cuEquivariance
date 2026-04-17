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
from __future__ import annotations

import dataclasses
import itertools

import cuequivariance as cue


def split_polynomial_by_irreps(
    polynomial: cue.SegmentedPolynomial,
    operand_id: int,
    irreps: cue.Irreps,
) -> cue.SegmentedPolynomial:
    """Split a polynomial operand according to irreps boundaries.

    Each ``(mul, ir)`` block in the irreps becomes a separate operand
    in the resulting polynomial.

    Args:
        polynomial: The polynomial to split.
        operand_id: Index of the operand to split (negative indices supported).
        irreps: Irreps describing the operand's structure.

    Returns:
        A new :class:`~cuequivariance.SegmentedPolynomial` with the specified
        operand split into one operand per ``(mul, ir)`` block.
    """
    offsets = list(
        itertools.accumulate((mul * ir.dim for mul, ir in irreps), initial=0)
    )
    return polynomial.split_operand_by_size(operand_id, offsets)


@dataclasses.dataclass(init=False, frozen=True)
class IrDictPolynomial:
    """A segmented polynomial with per-operand irreps metadata for the ``ir_dict`` workflow.

    This class pairs a :class:`~cuequivariance.SegmentedPolynomial` (already split
    by irrep) with the :class:`~cuequivariance.Irreps` that describe each operand group.

    Each :class:`~cuequivariance.Irreps` in ``input_irreps`` and ``output_irreps``
    corresponds to a logical operand group (e.g. weights, node features, spherical
    harmonics, output features). Within each group, every ``(mul, ir)`` block maps
    to one polynomial operand.

    Contract:
        - The polynomial is already split by irrep: each operand corresponds to
          exactly one ``(mul, ir)`` block.
        - The ``(mul, ir)`` blocks in ``input_irreps`` and ``output_irreps``
          are in the same order as the polynomial's input and output operands.
        - For each ``(mul, ir)`` block, the corresponding polynomial operand
          has size ``mul * ir.dim``.

    Args:
        polynomial: The underlying polynomial, already split by irrep.
        input_irreps: One :class:`~cuequivariance.Irreps` per input group.
        output_irreps: One :class:`~cuequivariance.Irreps` per output group.
    """

    polynomial: cue.SegmentedPolynomial
    input_irreps: tuple[cue.Irreps, ...]
    output_irreps: tuple[cue.Irreps, ...]

    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        input_irreps: list[cue.Irreps] | tuple[cue.Irreps, ...],
        output_irreps: list[cue.Irreps] | tuple[cue.Irreps, ...],
    ):
        object.__setattr__(self, "polynomial", polynomial)
        object.__setattr__(self, "input_irreps", tuple(input_irreps))
        object.__setattr__(self, "output_irreps", tuple(output_irreps))

        expected_inputs = sum(len(irreps) for irreps in self.input_irreps)
        if expected_inputs != polynomial.num_inputs:
            raise ValueError(
                f"input_irreps describe {expected_inputs} operands, "
                f"but polynomial has {polynomial.num_inputs} inputs"
            )

        expected_outputs = sum(len(irreps) for irreps in self.output_irreps)
        if expected_outputs != polynomial.num_outputs:
            raise ValueError(
                f"output_irreps describe {expected_outputs} operands, "
                f"but polynomial has {polynomial.num_outputs} outputs"
            )

        operand_idx = 0
        for irreps in self.input_irreps:
            for mul, ir in irreps:
                actual_size = polynomial.inputs[operand_idx].size
                expected_size = mul * ir.dim
                if expected_size != actual_size:
                    raise ValueError(
                        f"Input operand {operand_idx} ({mul}x{ir}): "
                        f"expected size {expected_size}, "
                        f"got {actual_size}"
                    )
                operand_idx += 1

        operand_idx = 0
        for irreps in self.output_irreps:
            for mul, ir in irreps:
                actual_size = polynomial.outputs[operand_idx].size
                expected_size = mul * ir.dim
                if expected_size != actual_size:
                    raise ValueError(
                        f"Output operand {operand_idx} ({mul}x{ir}): "
                        f"expected size {expected_size}, "
                        f"got {actual_size}"
                    )
                operand_idx += 1

    def __repr__(self):
        labels = []
        for irreps in self.input_irreps:
            for mul, ir in irreps:
                labels.append(f"{mul}x{ir}" if mul > 1 else f"{ir}")
        for irreps in self.output_irreps:
            for mul, ir in irreps:
                labels.append(f"{mul}x{ir}" if mul > 1 else f"{ir}")
        return self.polynomial.to_string(labels)
