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
        object.__setattr__(self, "operands", tuple(operands))
        object.__setattr__(self, "polynomial", polynomial)
        assert len(self.operands) == len(self.polynomial)
        for rep, size in zip(self.operands, self.polynomial.buffer_sizes):
            assert size is None or size == rep.dim

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
    def inputs(self) -> tuple[cue.Rep, ...]:
        return self.operands[: self.num_inputs]

    @property
    def outputs(self) -> tuple[cue.Rep, ...]:
        return self.operands[self.num_inputs :]
