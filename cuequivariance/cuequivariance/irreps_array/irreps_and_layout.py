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

from dataclasses import dataclass, field

import numpy as np

import cuequivariance as cue


@dataclass(init=False, frozen=True)
class IrrepsAndLayout(cue.Rep):
    irreps: cue.Irreps = field()
    layout: cue.IrrepsLayout = field()

    def __init__(
        self, irreps: cue.Irreps | str, layout: cue.IrrepsLayout | None = None
    ):
        irreps = cue.Irreps(irreps)
        if layout is None:
            layout = cue.get_layout_scope()

        object.__setattr__(self, "irreps", irreps)
        object.__setattr__(self, "layout", layout)

    def __repr__(self):
        return f"{self.irreps}"

    def _dim(self) -> int:
        return self.irreps.dim

    def algebra(self) -> np.ndarray:
        return self.irreps.irrep_class.algebra()

    def continuous_generators(self) -> np.ndarray:
        if self.layout == cue.mul_ir:
            return block_diag(
                [np.kron(np.eye(mul), ir.X) for mul, ir in self.irreps], (self.lie_dim,)
            )
        if self.layout == cue.ir_mul:
            return block_diag(
                [np.kron(ir.X, np.eye(mul)) for mul, ir in self.irreps], (self.lie_dim,)
            )

    def discrete_generators(self) -> np.ndarray:
        if self.layout == cue.mul_ir:
            return block_diag(
                [np.kron(np.eye(mul), ir.H) for mul, ir in self.irreps], (self.lie_dim,)
            )
        if self.layout == cue.ir_mul:
            return block_diag(
                [np.kron(ir.H, np.eye(mul)) for mul, ir in self.irreps], (self.lie_dim,)
            )

    def trivial(self) -> cue.Rep:
        ir = self.irreps.irrep_class.trivial()
        return IrrepsAndLayout(
            cue.Irreps(self.irreps.irrep_class, [ir]),
            self.layout,
        )

    def is_scalar(self) -> bool:
        return self.irreps.is_scalar()

    def __eq__(self, other: cue.Rep) -> bool:
        if isinstance(other, IrrepsAndLayout):
            return self.irreps == other.irreps and (
                self.irreps.layout_insensitive() or self.layout == other.layout
            )
        return cue.Rep.__eq__(self, other)


def block_diag(entries: list[np.ndarray], leading_shape: tuple[int, ...]) -> np.ndarray:
    if len(entries) == 0:
        return np.zeros(leading_shape + (0, 0))

    A = entries[0]
    assert A.shape[:-2] == leading_shape

    if len(entries) == 1:
        return A

    B = entries[1]
    assert B.shape[:-2] == leading_shape

    i, m = A.shape[-2:]
    j, n = B.shape[-2:]

    C = np.block(
        [[A, np.zeros(leading_shape + (i, n))], [np.zeros(leading_shape + (j, m)), B]]
    )
    return block_diag([C] + entries[2:], leading_shape)
