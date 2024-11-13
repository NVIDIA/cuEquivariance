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

import dataclasses
import itertools
import re
from typing import *

import numpy as np

from cuequivariance.representation import Rep


@dataclasses.dataclass(frozen=True)
class Irrep(Rep):
    r"""Subclass of :class:`Rep` for an irreducible representation of a Lie group.

    It extends the base class by adding:

    - A regular expression pattern for parsing the string representation.
    - The selection rule for the tensor product of two irreps.
    - An ordering relation for sorting the irreps.
    - A Clebsch-Gordan method for computing the Clebsch-Gordan coefficients.
    """

    @classmethod
    def regexp_pattern(cls) -> re.Pattern:
        """Regular expression pattern for parsing the string representation."""
        raise NotImplementedError

    @classmethod
    def from_string(cls, string: str) -> Irrep:
        """Create an instance from the string representation."""
        raise NotImplementedError

    @classmethod
    def _from(cls, *args) -> Irrep:
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, cls):
                return arg
            if isinstance(arg, str):
                return cls.from_string(arg)
            if isinstance(arg, Iterable):  # if isinstance(arg, tuple):
                return cls(
                    *iter(arg)
                )  # the iter is needed for compatibility with e3nn.o3.Irrep
        return cls(*args)

    def __repr__(rep: Irrep) -> str:
        raise NotImplementedError

    def __mul__(rep1: Irrep, rep2: Irrep) -> Iterable[Irrep]:
        """Selection rule for the tensor product of two irreps."""
        raise NotImplementedError

    def __lt__(rep1: Irrep, rep2: Irrep) -> bool:
        """This is required for sorting the irreps.

        * the dimension is the first criterion (ascending order)
        """
        if rep1 == rep2 or (rep1.dim != rep2.dim):
            return rep1.dim < rep2.dim
        raise NotImplementedError

    @classmethod
    def iterator(cls) -> Iterable[Irrep]:
        r"""Iterator over all irreps of the Lie group.

        * the first element is the trivial irrep
        * the elements respect the partial order defined by ``__lt__``
        """
        raise NotImplementedError

    @classmethod
    def trivial(cls) -> Irrep:
        rep: Irrep = cls.iterator().__next__()
        assert rep.is_trivial(), "problem with the iterator"
        return rep

    @classmethod
    def clebsch_gordan(cls, rep1: Irrep, rep2: Irrep, rep3: Irrep) -> np.ndarray:
        """Clebsch-Gordan coefficients tensor.

        The shape is ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)`` and rep3 is the output irrep.
        """
        raise NotImplementedError


def clebsch_gordan(rep1: Irrep, rep2: Irrep, rep3: Irrep) -> np.ndarray:
    r"""
    Compute the Clebsch-Gordan coefficients.

    The Clebsch-Gordan coefficients are used to decompose the tensor product of two irreducible representations
    into a direct sum of irreducible representations. This method computes the Clebsch-Gordan coefficients
    for the given input representations and returns an array of shape (num_solutions, dim1, dim2, dim3),
    where num_solutions is the number of solutions, dim1 is the dimension of rep1, dim2 is the dimension of rep2,
    and dim3 is the dimension of rep3.

    The Clebsch-Gordan coefficients satisfy the following equation:

    .. math::

        C_{ljk} X^1_{li} + C_{ilk} X^2_{lj} = X^3_{kl} C_{ijl}

    Args:
        rep1 (Irrep): The first irreducible representation (input).
        rep2 (Irrep): The second irreducible representation (input).
        rep3 (Irrep): The third irreducible representation (output).

    Returns:
        np.ndarray: An array of shape (num_solutions, dim1, dim2, dim3) containing the Clebsch-Gordan coefficients.
    """
    return rep1.clebsch_gordan(rep1, rep2, rep3)


def selection_rule_product(
    irs1: Union[Irrep, Sequence[Irrep], None],
    irs2: Union[Irrep, Sequence[Irrep], None],
) -> Optional[FrozenSet[Irrep]]:
    if irs1 is None or irs2 is None:
        return None

    if isinstance(irs1, Irrep):
        irs1 = [irs1]
    if isinstance(irs2, Irrep):
        irs2 = [irs2]
    irs1, irs2 = list(irs1), list(irs2)
    assert all(isinstance(x, Irrep) for x in irs1)
    assert all(isinstance(x, Irrep) for x in irs2)

    out = set()
    for ir1, ir2 in itertools.product(irs1, irs2):
        out = out.union(ir1 * ir2)
    return frozenset(out)


def selection_rule_power(
    irrep_class: Type[Irrep], irs: Union[Irrep, Sequence[Irrep]], n: int
) -> FrozenSet[Irrep]:
    if isinstance(irs, Irrep):
        irs = [irs]
    irs = list(irs)
    assert all(isinstance(x, Irrep) for x in irs)

    out = frozenset([irrep_class.trivial()])
    for _ in range(n):
        out = selection_rule_product(out, irs)
    return out
