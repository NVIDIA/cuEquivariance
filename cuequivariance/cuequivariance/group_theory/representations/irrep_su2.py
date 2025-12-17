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

import itertools
import re
from dataclasses import dataclass
from math import factorial
from typing import Iterator

import numpy as np

from cuequivariance.group_theory.representations import Irrep


# This class is an adaptation of https://github.com/lie-nn/lie-nn/blob/70adebce44e3197ee17f780585c6570d836fc2fe/lie_nn/_src/irreps/su2.py
@dataclass(frozen=True)
class SU2(Irrep):
    r"""Subclass of :class:`Irrep`, irreducible representations of the Lie group :math:`SU(2)`.

    Each representation is labeled by a non-negative half-integer :math:`j`.

    Examples:
        >>> SU2(0)
        0
        >>> SU2(1/2)
        1/2
        >>> SU2(1/2).dim
        2
        >>> SU2.from_string("1")
        1
    """

    j: float

    def __post_init__(self):
        assert isinstance(self.j, (float, int))
        assert self.j >= 0
        assert round(2 * self.j) == 2 * self.j

    @classmethod
    def regexp_pattern(cls) -> re.Pattern:
        return re.compile(r"(\d+|\d+/2)")

    @classmethod
    def from_string(cls, string: str) -> SU2:
        string = string.strip()
        if string.endswith("/2"):
            j = int(string[:-2]) / 2
        else:
            j = int(string)
        return cls(j=j)

    def __repr__(rep: SU2) -> str:
        if round(rep.j) == rep.j:
            return f"{int(rep.j)}"
        return f"{int(2 * rep.j)}/2"

    def __mul__(rep1: SU2, rep2: SU2) -> Iterator[SU2]:
        rep2 = rep1._from(rep2)
        return [
            SU2(j=j / 2)
            for j in range(
                abs(int(2 * rep1.j) - int(2 * rep2.j)),
                int(2 * rep1.j + 2 * rep2.j) + 1,
                2,
            )
        ]

    @classmethod
    def clebsch_gordan(cls, rep1: SU2, rep2: SU2, rep3: SU2) -> np.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

        if rep3 in rep1 * rep2:
            return clebsch_gordanSU2mat(rep1.j, rep2.j, rep3.j)[None]
        else:
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: SU2) -> int:
        return int(2 * rep.j + 1)

    def is_scalar(rep: SU2) -> bool:
        return rep.j == 0

    def __lt__(rep1: SU2, rep2: SU2) -> bool:
        rep2 = rep1._from(rep2)
        return rep1.j < rep2.j

    @classmethod
    def iterator(cls) -> Iterator[SU2]:
        for j in itertools.count(0):
            yield SU2(j=j / 2)

    def discrete_generators(rep: SU2) -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: SU2) -> np.ndarray:
        j = rep.j
        m = np.arange(-j, j)
        raising = np.diag(-np.sqrt(j * (j + 1) - m * (m + 1)), k=-1)

        m = np.arange(-j + 1, j + 1)
        lowering = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), k=1)

        m = np.arange(-j, j + 1)
        return np.stack(
            [
                0.5 * (raising + lowering),  # x (usually)
                np.diag(1j * m),  # z (usually)
                -0.5j * (raising - lowering),  # -y (usually)
            ],
            axis=0,
        )

    def algebra(rep=None) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return np.array(
            [
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ],
                [
                    [0, 0, -1],
                    [0, 0, 0],
                    [1, 0, 0],
                ],
                [
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0.0],
                ],
            ]
        )


# Taken from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################


def clebsch_gordanSU2mat(j1, j2, j3):
    """Calculates the Clebsch-Gordon matrix
    for SU(2) coupling j1 and j2 to give j3.
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    Returns
    -------
    cg_matrix : numpy.array
        Requested Clebsch-Gordan matrix.
    """
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = (
                        clebsch_gordanSU2coeffs((j1, m1), (j2, m2), (j3, m1 + m2))
                    )
    # in e3nn it was divided by np.sqrt(2 * j3 + 1) to make the CG symmetric wrt j1,j2,j3
    # but if we say that j3 is always the output, it makes sense to not divide
    return mat  # / np.sqrt(2 * j3 + 1)


def clebsch_gordanSU2coeffs(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient
    for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    m1 : float
        z-component of angular momentum 1.
    m2 : float
        z-component of angular momentum 2.
    m3 : float
        z-component of angular momentum 3.
    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    """
    from fractions import Fraction

    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    j1 = Fraction(int(2 * j1), 2)
    j2 = Fraction(int(2 * j2), 2)
    j3 = Fraction(int(2 * j3), 2)
    m1 = Fraction(int(2 * m1), 2)
    m2 = Fraction(int(2 * m2), 2)
    m3 = Fraction(int(2 * m3), 2)

    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        assert n == round(n)
        return factorial(round(n))

    C = (2 * j3 + 1) * Fraction(
        f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3),
        f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
    )

    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1) ** (v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )

    if S > 0:
        return (C * S**2) ** 0.5
    else:
        return -((C * S**2) ** 0.5)
