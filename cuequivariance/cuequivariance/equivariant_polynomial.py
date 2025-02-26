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

import dataclasses

import cuequivariance as cue


@dataclasses.dataclass(init=False, frozen=True)
class EquivariantPolynomial:
    operands: tuple[cue.Rep, ...]
    polynomial: cue.SegmentedPolynomial

    def __init__(self, operands: list[cue.Rep], polynomial: cue.SegmentedPolynomial):
        assert isinstance(polynomial, cue.SegmentedPolynomial)
        object.__setattr__(self, "operands", tuple(operands))
        object.__setattr__(self, "polynomial", polynomial)
        assert (
            len(self.operands)
            == self.polynomial.num_inputs + self.polynomial.num_outputs
        )
        for rep, size in zip(self.operands, self.polynomial.buffer_sizes):
            assert size is None or size == rep.dim

    def __repr__(self):
        return self.polynomial.to_string([f"{rep}" for rep in self.operands])

    def __hash__(self) -> int:
        return hash((self.operands, self.polynomial))

    def __mul__(self, factor: float) -> EquivariantPolynomial:
        return EquivariantPolynomial(self.operands, self.polynomial * factor)

    def __rmul__(self, factor: float) -> EquivariantPolynomial:
        return self.__mul__(factor)

    @property
    def num_operands(self) -> int:
        return len(self.operands)

    @property
    def num_inputs(self) -> int:
        return self.polynomial.num_inputs

    @property
    def num_outputs(self) -> int:
        return self.polynomial.num_outputs

    @property
    def inputs(self) -> tuple[cue.Rep, ...]:
        return self.operands[: self.num_inputs]

    @property
    def outputs(self) -> tuple[cue.Rep, ...]:
        return self.operands[self.num_inputs :]

    @classmethod
    def stack(
        cls, polys: list[EquivariantPolynomial], stacked: list[bool]
    ) -> EquivariantPolynomial:
        assert len(polys) > 0
        num_operands = polys[0].num_operands

        assert all(pol.num_operands == num_operands for pol in polys)
        assert len(stacked) == num_operands

        operands = []
        for oid in range(num_operands):
            if stacked[oid]:
                for pol in polys:
                    if not isinstance(pol.operands[oid], cue.IrrepsAndLayout):
                        raise ValueError(
                            f"Cannot stack operand {oid} of type {type(pol.operands[oid])}"
                        )
                operands.append(cue.concatenate([pol.operands[oid] for pol in polys]))
            else:
                ope = polys[0].operands[oid]
                for pol in polys:
                    if pol.operands[oid] != ope:
                        raise ValueError(
                            f"Operand {oid} must be the same for all polynomials."
                            f" Found {ope} and {pol.operands[oid]}"
                        )
                operands.append(ope)

        return cls(
            operands,
            cue.SegmentedPolynomial.stack([pol.polynomial for pol in polys], stacked),
        )
