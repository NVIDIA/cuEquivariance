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
from cuequivariance.group_theory.representations import Rep


@dataclass(init=False, frozen=True)
class IrrepsAndLayout(Rep):
    r"""
    A group representation (:class:`Rep`) made from the combination of :class:`Irreps` and :class:`IrrepsLayout` into a single object.

    This class inherits from :class:`Rep`::

        Rep                  <--- Base class for all representations
        ├── Irrep            <--- Base class for all irreducible representations
            ├── SU2
            ├── SO3
            ├── O3
        ├── IrrepsAndLayout  <--- This class

        IrrepsLayout         <--- Enum class with two values: mul_ir and ir_mul

        Irreps               <--- Collection of Irrep with multiplicities

    Args:
        irreps (Irreps or str): Irreducible representations and their multiplicities.
        layout (optional, IrrepsLayout): The data layout (``mul_ir`` or ``ir_mul``).

    Examples:
        Let's create rotations matrices for a 2x1 representation of SO(3) using two different layouts:

        >>> angles = np.array([np.pi, 0, 0])

        Here we use the ``ir_mul`` layout:

        >>> with cue.assume("SO3", cue.ir_mul):
        ...     rep = cue.IrrepsAndLayout("2x1")
        >>> R_ir_mul = rep.exp_map(angles, np.array([]))

        Here we use the ``mul_ir`` layout:

        >>> with cue.assume("SO3", cue.mul_ir):
        ...     rep = cue.IrrepsAndLayout("2x1")
        >>> R_mul_ir = rep.exp_map(angles, np.array([]))

        Let's see the difference between the two layouts:

        >>> R_ir_mul.round(1) + 0.0
        array([[ 1.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.],
               [ 0.,  0., -1.,  0.,  0.,  0.],
               [ 0.,  0.,  0., -1.,  0.,  0.],
               [ 0.,  0.,  0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  0.,  0., -1.]])

        >>> R_mul_ir.round(1) + 0.0
        array([[ 1.,  0.,  0.,  0.,  0.,  0.],
               [ 0., -1.,  0.,  0.,  0.,  0.],
               [ 0.,  0., -1.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  0.,  0., -1.]])
    """

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
        num_H = self.irreps.irrep_class.trivial().H.shape[0]

        if self.layout == cue.mul_ir:
            return block_diag(
                [np.kron(np.eye(mul), ir.H) for mul, ir in self.irreps], (num_H,)
            )
        if self.layout == cue.ir_mul:
            return block_diag(
                [np.kron(ir.H, np.eye(mul)) for mul, ir in self.irreps], (num_H,)
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
    assert A.shape[:-2] == leading_shape, (A.shape, leading_shape)

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
