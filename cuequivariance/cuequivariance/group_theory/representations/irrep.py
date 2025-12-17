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
import itertools
import re
from typing import FrozenSet, Iterable, Optional, Sequence, Type, Union

import numpy as np

import cuequivariance as cue  # noqa: F401
from cuequivariance.group_theory.representations import Rep


# This class is inspired from https://github.com/lie-nn/lie-nn/blob/70adebce44e3197ee17f780585c6570d836fc2fe/lie_nn/_src/irrep.py
@dataclasses.dataclass(frozen=True)
class Irrep(Rep):
    r"""
    Subclass of :class:`Rep` for an irreducible representation of a Lie group.

    It extends the base class by adding:

    - A regular expression pattern for parsing the string representation.
    - The selection rule for the tensor product of two irreps.
    - An ordering relation for sorting the irreps.
    - A Clebsch-Gordan method for computing the Clebsch-Gordan coefficients.

    .. _custom-irreps:

    Simple example of defining custom irreps
    ----------------------------------------

    In some cases, you may want to define a custom set of irreducible representations of a group.
    Here is a simple example of how to define the irreps of the group :math:`Z_2`. For this we need to define a class that inherits from :class:`cue.Irrep <cuequivariance.Irrep>` and implement the required methods.

    .. jupyter-execute::

        from __future__ import annotations
        import re
        from typing import Iterator
        import numpy as np
        import cuequivariance as cue


        class Z2(cue.Irrep):
            odd: bool

            def __init__(rep: Z2, odd: bool):
                rep.odd = odd

            @classmethod
            def regexp_pattern(cls) -> re.Pattern:
                return re.compile(r"(odd|even)")

            @classmethod
            def from_string(cls, string: str) -> Z2:
                return cls(odd=string == "odd")

            def __repr__(rep: Z2) -> str:
                return "odd" if rep.odd else "even"

            def __mul__(rep1: Z2, rep2: Z2) -> Iterator[Z2]:
                return [Z2(odd=rep1.odd ^ rep2.odd)]

            @classmethod
            def clebsch_gordan(cls, rep1: Z2, rep2: Z2, rep3: Z2) -> np.ndarray:
                if rep3 in rep1 * rep2:
                    return np.array(
                        [[[[1]]]]
                    )  # (number_of_paths, rep1.dim, rep2.dim, rep3.dim)
                else:
                    return np.zeros((0, 1, 1, 1))

            @property
            def dim(rep: Z2) -> int:
                return 1

            def __lt__(rep1: Z2, rep2: Z2) -> bool:
                # False < True
                return rep1.odd < rep2.odd

            @classmethod
            def iterator(cls) -> Iterator[Z2]:
                for odd in [False, True]:
                    yield Z2(odd=odd)

            def discrete_generators(rep: Z2) -> np.ndarray:
                if rep.odd:
                    return -np.ones((1, 1, 1))  # (number_of_generators, rep.dim, rep.dim)
                else:
                    return np.ones((1, 1, 1))

            def continuous_generators(rep: Z2) -> np.ndarray:
                return np.zeros((0, rep.dim, rep.dim))  # (lie_dim, rep.dim, rep.dim)

            def algebra(self) -> np.ndarray:
                return np.zeros((0, 0, 0))  # (lie_dim, lie_dim, lie_dim)


        cue.Irreps(Z2, "13x odd + 6x even")

    .. rubric:: Methods to implement
    """

    @classmethod
    def regexp_pattern(cls) -> re.Pattern:
        """
        Regular expression pattern for parsing the string representation.
        """
        raise NotImplementedError

    @classmethod
    def from_string(cls, string: str) -> Irrep:
        """
        Create an instance from the string representation.
        """
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
        """
        This is required for sorting the irreps.

        - the dimension is the first criterion (ascending order)
        """
        if rep1 == rep2 or (rep1.dim != rep2.dim):
            return rep1.dim < rep2.dim
        raise NotImplementedError

    @classmethod
    def iterator(cls) -> Iterable[Irrep]:
        r"""
        Iterator over all irreps of the Lie group.

        - the first element is the trivial irrep
        - the elements respect the partial order defined by ``__lt__``
        """
        raise NotImplementedError

    @classmethod
    def clebsch_gordan(cls, rep1: Irrep, rep2: Irrep, rep3: Irrep) -> np.ndarray:
        """
        Clebsch-Gordan coefficients tensor.

        The shape is ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)`` and rep3 is the output irrep.

        See also:
            :func:`clebsch_gordan`.
        """
        raise NotImplementedError

    @classmethod
    def trivial(cls) -> Irrep:
        """
        Return the trivial irrep.

        Implemented by returning the first element of the iterator.
        """
        rep: Irrep = cls.iterator().__next__()
        assert rep.is_trivial(), "problem with the iterator"
        return rep


def clebsch_gordan(rep1: Irrep, rep2: Irrep, rep3: Irrep) -> np.ndarray:
    r"""
    Compute the Clebsch-Gordan coefficients.

    The Clebsch-Gordan coefficients are used to decompose the tensor product of two irreducible representations
    into a direct sum of irreducible representations. This method computes the Clebsch-Gordan coefficients
    for the given input representations and returns an array of shape ``(num_solutions, dim1, dim2, dim3)``,
    where num_solutions is the number of solutions, ``dim1`` is the dimension of ``rep1``, ``dim2`` is the
    dimension of ``rep2``, and ``dim3`` is the dimension of ``rep3``.

    The Clebsch-Gordan coefficients satisfy the following equation:

    .. math::

        C_{ljk} X^1_{li} + C_{ilk} X^2_{lj} = X^3_{kl} C_{ijl}

    Args:
        rep1 (Irrep): The first irreducible representation (input).
        rep2 (Irrep): The second irreducible representation (input).
        rep3 (Irrep): The third irreducible representation (output).

    Returns:
        np.ndarray: An array of shape ``(num_solutions, dim1, dim2, dim3)``.

    Examples:
        >>> rep1 = cue.SO3(1)
        >>> rep2 = cue.SO3(1)
        >>> rep3 = cue.SO3(2)
        >>> C = clebsch_gordan(rep1, rep2, rep3)
        >>> C.shape
        (1, 3, 3, 5)
        >>> C
        array([[[[ 0.  ...]]]])

        If there is no solution, the output is an empty array.

        >>> C = clebsch_gordan(cue.SO3(1), cue.SO3(1), cue.SO3(3))
        >>> C.shape
        (0, 3, 3, 7)
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
