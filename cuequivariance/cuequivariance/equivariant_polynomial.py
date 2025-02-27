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

import numpy as np

import cuequivariance as cue


@dataclasses.dataclass(init=False, frozen=True)
class EquivariantPolynomial:
    operands: tuple[cue.Rep, ...]
    polynomial: cue.SegmentedPolynomial

    def __init__(self, operands: list[cue.Rep], polynomial: cue.SegmentedPolynomial):
        assert isinstance(polynomial, cue.SegmentedPolynomial)
        object.__setattr__(self, "operands", tuple(operands))
        object.__setattr__(self, "polynomial", polynomial)
        if (
            len(self.operands)
            != self.polynomial.num_inputs + self.polynomial.num_outputs
        ):
            raise ValueError(
                f"Number of operands {len(self.operands)} must equal the number of inputs"
                f" {self.polynomial.num_inputs} plus the number of outputs {self.polynomial.num_outputs}"
            )
        for rep, size in zip(self.operands, self.polynomial.buffer_sizes):
            assert size is None or size == rep.dim

    def __hash__(self) -> int:
        return hash((self.operands, self.polynomial))

    def __eq__(self, value) -> bool:
        assert isinstance(value, EquivariantPolynomial)
        return self.operands == value.operands and self.polynomial == value.polynomial

    def __lt__(self, value) -> bool:
        assert isinstance(value, EquivariantPolynomial)
        return (
            self.num_inputs,
            self.num_outputs,
            self.operands,
            self.polynomial,
        ) < (
            value.num_inputs,
            value.num_outputs,
            value.operands,
            value.polynomial,
        )

    def __mul__(self, factor: float) -> EquivariantPolynomial:
        return EquivariantPolynomial(self.operands, self.polynomial * factor)

    def __rmul__(self, factor: float) -> EquivariantPolynomial:
        return self.__mul__(factor)

    def __repr__(self):
        return self.polynomial.to_string([f"{rep}" for rep in self.operands])

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        return self.polynomial(*inputs)

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

    def consolidate(self) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            self.operands,
            self.polynomial.consolidate(),
        )

    def buffer_used(self) -> list[bool]:
        return self.polynomial.buffer_used()

    def remove_unused_buffers(self) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            [rep for u, rep in zip(self.buffer_used(), self.operands) if u],
            self.polynomial.remove_unused_buffers(),
        )

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

        poly = cls(
            operands,
            cue.SegmentedPolynomial.stack([pol.polynomial for pol in polys], stacked),
        )
        return poly.consolidate()

    def squeeze_modes(self) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            self.operands,
            self.polynomial.squeeze_modes(),
        )

    def flatten_coefficient_modes(self) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            self.operands,
            self.polynomial.flatten_coefficient_modes(),
        )

    def jvp(self, has_tangent: list[bool]) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            list(self.inputs)
            + [x for has, x in zip(has_tangent, self.inputs) if has]
            + list(self.outputs),
            self.polynomial.jvp(has_tangent),
        )

    def transpose(
        self,
        is_undefined_primal: list[bool],
        has_cotangent: list[bool],
    ) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            # defined inputs
            [
                x
                for is_undefined, x in zip(is_undefined_primal, self.inputs)
                if not is_undefined
            ]
            # cotangent outputs
            + [x for has, x in zip(has_cotangent, self.outputs) if has]
            # undefined inputs
            + [
                x
                for is_undefined, x in zip(is_undefined_primal, self.inputs)
                if is_undefined
            ],
            self.polynomial.transpose(is_undefined_primal, has_cotangent),
        )

    def backward(
        self, requires_gradient: list[bool], has_cotangent: list[bool]
    ) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            list(self.inputs)
            + [x for has, x in zip(has_cotangent, self.outputs) if has]
            + [x for req, x in zip(requires_gradient, self.inputs) if req],
            self.polynomial.backward(requires_gradient, has_cotangent),
        )
