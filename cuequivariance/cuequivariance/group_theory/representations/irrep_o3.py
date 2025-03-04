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

import itertools
import re
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from cuequivariance.group_theory.representations import SO3, Irrep


# This class is an adaptation of https://github.com/lie-nn/lie-nn/blob/70adebce44e3197ee17f780585c6570d836fc2fe/lie_nn/_src/irreps/o3_real.py
@dataclass(frozen=True)
class O3(Irrep):
    r"""Subclass of :class:`Irrep`, real irreducible representations of the 3D rotation group :math:`O(3)`.

    Each representation is labeled by a non-negative integer :math:`l` and a parity :math:`p = \pm 1`.

    Examples:
        >>> O3(0, 1)
        0e
        >>> O3(1, -1)
        1o
        >>> O3(1, -1).dim
        3
        >>> O3.from_string("2o")
        2o
    """

    l: int  # non-negative integer # noqa: E741
    p: int  # 1 or -1

    @classmethod
    def regexp_pattern(cls) -> re.Pattern:
        return re.compile(r"(\d+)([eo])")

    @classmethod
    def from_string(cls, s: str) -> O3:
        s = s.strip()
        ell = int(s[:-1])
        p = {"e": 1, "o": -1}[s[-1]]
        return cls(l=ell, p=p)

    def __repr__(rep: O3) -> str:
        return f"{rep.l}{['e', 'o'][rep.p < 0]}"

    def __mul__(rep1: O3, rep2: O3) -> Iterator[O3]:
        rep2 = rep1._from(rep2)
        p = rep1.p * rep2.p
        return [
            O3(l=ell, p=p)
            for ell in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)
        ]

    @classmethod
    def clebsch_gordan(cls, rep1: O3, rep2: O3, rep3: O3) -> np.ndarray:
        rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

        if rep1.p * rep2.p == rep3.p:
            return SO3.clebsch_gordan(rep1.l, rep2.l, rep3.l)
        else:
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: O3) -> int:
        return 2 * rep.l + 1

    def is_scalar(rep: O3) -> bool:
        return rep.l == 0 and rep.p == 1

    def __lt__(rep1: O3, rep2: O3) -> bool:
        rep2 = rep1._from(rep2)
        return (rep1.l, -rep1.p * (-1) ** rep1.l) < (rep2.l, -rep2.p * (-1) ** rep2.l)

    @classmethod
    def iterator(cls) -> Iterator[O3]:
        for ell in itertools.count(0):
            yield O3(l=ell, p=1 * (-1) ** ell)
            yield O3(l=ell, p=-1 * (-1) ** ell)

    def continuous_generators(rep: O3) -> np.ndarray:
        return SO3(l=rep.l).continuous_generators()

    def discrete_generators(rep: O3) -> np.ndarray:
        return rep.p * np.eye(rep.dim)[None]

    def algebra(rep=None) -> np.ndarray:
        return SO3.algebra()

    def rotation(rep: O3, axis: np.ndarray, angle: float) -> np.ndarray:
        return SO3(l=rep.l).rotation(axis, angle)

    # def exp_map(
    #     self, continuous_params: np.ndarray, discrete_params: np.ndarray
    # ) -> np.ndarray:
    #     I = (-1) ** discrete_params[0]
    #     R = SO3(l=self.l).exp_map(continuous_params, [])
    #     return I * R
