# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# MIT License
# Copyright (c) 2023 lie-nn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import *

import numpy as np

from cuequivariance.representation import SO3, Irrep


@dataclass(frozen=True)
class O3(Irrep):
    r"""Subclass of :class:`Irrep`, real irreducible representations of the 3D rotation group :math:`O(3)`.

    Each representation is labeled by a non-negative integer :math:`l` and a parity :math:`p = \pm 1`.
    """

    l: int  # non-negative integer
    p: int  # 1 or -1

    @classmethod
    def regexp_pattern(cls) -> re.Pattern:
        return re.compile(r"(\d+)([eo])")

    @classmethod
    def from_string(cls, s: str) -> O3:
        s = s.strip()
        l = int(s[:-1])
        p = {"e": 1, "o": -1}[s[-1]]
        return cls(l=l, p=p)

    def __repr__(rep: O3) -> str:
        return f"{rep.l}{['e', 'o'][rep.p < 0]}"

    def __mul__(rep1: O3, rep2: O3) -> Iterator[O3]:
        rep2 = rep1._from(rep2)
        p = rep1.p * rep2.p
        return [
            O3(l=l, p=p) for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)
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
        for l in itertools.count(0):
            yield O3(l=l, p=1 * (-1) ** l)
            yield O3(l=l, p=-1 * (-1) ** l)

    def continuous_generators(rep: O3) -> np.ndarray:
        return SO3(l=rep.l).continuous_generators()

    def discrete_generators(rep: O3) -> np.ndarray:
        return rep.p * np.eye(rep.dim)[None]

    def algebra(rep=None) -> np.ndarray:
        return SO3.algebra()

    def rotation(rep: O3, axis: np.ndarray, angle: float) -> np.ndarray:
        return SO3(l=rep.l).rotation(axis, angle)
