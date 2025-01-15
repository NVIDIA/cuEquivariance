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

import dataclasses
import re
from typing import Any, Callable, NamedTuple, Optional, Sequence, Type, Union

import cuequivariance as cue


@dataclasses.dataclass(frozen=True)
class MulIrrep:
    mul: int
    ir: cue.Irrep

    def __repr__(self):
        if self.mul == 1:
            return f"{self.ir}"
        return f"{self.mul}x{self.ir}"

    def __iter__(self):
        return iter((self.mul, self.ir))


# This class is inspired from https://github.com/e3nn/e3nn-jax/blob/245e17eb23deaccad9f2c9cfd40fe40515e3c074/e3nn_jax/_src/irreps.py
@dataclasses.dataclass(init=False, frozen=True)
class Irreps:
    """
    Direct sum of irreducible representations with multiplicities.

    For more information, see :ref:`tuto_irreps`.

    Args:
        irrep_class: Class of the irreducible representations
            (e.g., :class:`SU2 <cuequivariance.SU2>`, :class:`SO3 <cuequivariance.SO3>`, :class:`O3 <cuequivariance.O3>`).
        input: A description of multiplicities of each irreducible representation.

    Examples:
        >>> Irreps("SO3", "16x0 + 4x1")
        16x0+4x1

        >>> Irreps(cue.SO3, "16x0 + 4x1")
        16x0+4x1

        >>> Irreps(cue.SO3, [(16, 0), (4, 1)])
        16x0+4x1

        >>> with cue.assume("SO3"):
        ...     Irreps("16x0 + 4x1")
        16x0+4x1

    .. rubric:: Methods
    """

    irrep_class: Type[cue.Irrep]
    _mulirreps: tuple[MulIrrep, ...]

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Irreps):
            args = (args[0].irrep_class, args[0]._mulirreps)

        if len(args) == 1 and isinstance(args[0], cue.Irrep):
            args = (type(args[0]), [(1, args[0])])

        if len(args) == 1:
            args = (cue.get_irrep_scope(), args[0])

        if len(args) != 2:
            raise ValueError(
                "Irreps requires two arguments: irrep_collection and input."
                f" Got {len(args)} arguments."
            )

        irrep_class: Any = args[0]
        if isinstance(irrep_class, str):
            irrep_class = getattr(cue, irrep_class)
        if isinstance(irrep_class, cue.Irreps):
            irrep_class = irrep_class.irrep_class
        if isinstance(irrep_class, cue.Irrep):
            irrep_class = type(irrep_class)
        assert isinstance(irrep_class, type) and issubclass(irrep_class, cue.Irrep)

        irrep_class: Type[cue.Irrep]
        input: Union[Irreps, str, Sequence[MulIrrep, cue.Irrep]] = args[1]

        mulreps = []

        if isinstance(input, str):
            for mul_rep_str in [] if input.strip() == "" else input.split("+"):
                if "x" in mul_rep_str:
                    mul, rep_str = mul_rep_str.split("x")
                    mul = int(mul)
                    rep_str = rep_str.strip()
                else:
                    mul = 1
                    rep_str = mul_rep_str.strip()

                if not re.match(irrep_class.regexp_pattern(), rep_str):
                    raise ValueError(
                        f"Invalid representation string: {rep_str}, expected pattern: {irrep_class.regexp_pattern().pattern}"
                    )

                rep = irrep_class.from_string(rep_str)
                mulreps.append(MulIrrep(mul=mul, ir=rep))

        elif isinstance(input, Irreps):
            if irrep_class != input.irrep_class:
                raise ValueError(
                    f"Invalid representation: {input.irrep_class}, expected {irrep_class}"
                )
            mulreps = input._mulirreps
        else:
            for x in input:
                if isinstance(x, MulIrrep):
                    if not isinstance(x.ir, irrep_class):
                        raise ValueError(
                            f"Invalid representation: {x.ir}, expected subclass of {irrep_class}"
                        )
                    mulreps.append(x)
                elif isinstance(x, irrep_class):
                    mulreps.append(MulIrrep(mul=1, ir=x))
                else:
                    try:
                        mul, ir = x
                    except ValueError:
                        raise ValueError(
                            f"Invalid representation: {x}, expected (mul, irrep)"
                        )

                    if not isinstance(mul, int):
                        raise ValueError(f"Invalid multiplicity: {mul}, expected int")

                    try:
                        ir = irrep_class._from(ir)
                    except ValueError:
                        raise ValueError(
                            f"Invalid representation: {ir}, expected {irrep_class}"
                        )

                    mulreps.append(MulIrrep(mul=mul, ir=ir))

        object.__setattr__(self, "irrep_class", irrep_class)
        object.__setattr__(self, "_mulirreps", tuple(mulreps))

    def __len__(self):
        return len(self._mulirreps)

    def __iter__(self):
        return iter(self._mulirreps)

    def __getitem__(self, index: Union[int, slice]) -> Union[MulIrrep, Irreps]:
        x = self._mulirreps[index]

        if isinstance(index, slice):
            return Irreps(self.irrep_class, x)
        return x

    def __contains__(self, rep: Union[str, cue.Irrep]) -> bool:
        # This function does not check the multiplicity of the representation!
        rep = self.irrep_class._from(rep)
        return any(mulrep.ir == rep for mulrep in self)

    def __repr__(self):
        return "+".join(f"{mulrep}" for mulrep in self)

    def new_scalars(self, mul: int) -> Irreps:
        """
        Return a representation with all scalar representations.

        Examples:
            >>> Irreps("SO3", "32x1").new_scalars(2)
            2x0
        """
        return Irreps(self.irrep_class, [(mul, self.irrep_class.trivial())])

    def count(self, rep: Union[str, cue.Irrep]) -> int:
        """
        Count the total multiplicity of a representation.

        Examples:
            >>> Irreps("SO3", "100x0 + 20x1 + 10x0").count("0")
            110
        """
        rep = self.irrep_class._from(rep)
        return sum(mul for mul, r in self if r == rep)

    @property
    def dim(self) -> int:
        """
        Total dimension of the representation.

        Examples:
            >>> Irreps("SO3", "100x0 + 10x1").dim
            130
        """
        return sum(mul * rep.dim for mul, rep in self)

    @property
    def num_irreps(self) -> int:
        """
        Return the number of irreducible representations.

        Examples:
            >>> Irreps("SO3", "100x0 + 10x1").num_irreps
            110
        """
        return sum(mul for mul, _ in self)

    @property
    def muls(self) -> list[int]:
        """
        List of multiplicities.

        Examples:
            >>> Irreps("SO3", "100x0 + 10x1").muls
            [100, 10]
        """
        return [mul for mul, _ in self]

    def slices(self) -> list[slice]:
        """
        List of slices for each segment of the representation.

        Examples:
            >>> Irreps("SO3", "100x0 + 10x1").slices()
            [slice(0, 100, None), slice(100, 130, None)]
        """
        s = []
        i = 0
        for mul, rep in self:
            d = mul * rep.dim
            s.append(slice(i, i + d))
            i += d
        return s

    def is_scalar(self) -> bool:
        """
        All representations are scalar.

        Note:
            This function does not check the multiplicity of the representation.

        Examples:
            >>> Irreps("SO3", "100x0 + 0x1").is_scalar()
            False

            >>> Irreps("SO3", "100x0").is_scalar()
            True
        """
        # Does not check the multiplicity of the representation!
        return all(rep.is_scalar() for mul, rep in self)

    def __add__(self, other):
        other = Irreps(self.irrep_class, other)
        return Irreps(self.irrep_class, self._mulirreps + other._mulirreps)

    def __radd__(self, other):
        return Irreps(self.irrep_class, other) + self

    def __mul__(self, other):
        if isinstance(other, int):
            return Irreps(
                self.irrep_class, [MulIrrep(mul * other, rep) for mul, rep in self]
            )
        return NotImplemented  # pragma: no cover

    def __rmul__(self, other):
        return self * other

    def __floordiv__(self, other):
        if isinstance(other, int):
            return Irreps(
                self.irrep_class, [MulIrrep(mul // other, rep) for mul, rep in self]
            )
        return NotImplemented  # pragma: no cover

    def __eq__(self, other):
        try:
            other = Irreps(self.irrep_class, other)
        except ValueError:
            return False
        return (
            self.irrep_class == other.irrep_class
            and self._mulirreps == other._mulirreps
        )

    def merge_consecutive(self) -> Irreps:
        """
        Merge consecutive segments with the same representation.

        Examples:
            >>> Irreps("SO3", "1 + 1 + 0 + 1").merge_consecutive()
            2x1+0+1
        """
        out = []
        for mul, rep in self:
            if out and out[-1][1] == rep:
                out[-1] = (out[-1][0] + mul, rep)
            else:
                out.append((mul, rep))
        return Irreps(self.irrep_class, out)

    def remove_zero_multiplicities(self) -> Irreps:
        """
        Remove zero multiplicities.

        Examples:
            >>> Irreps("SO3", "1 + 0x2 + 1").remove_zero_multiplicities()
            1+1
        """
        return Irreps(self.irrep_class, [(mul, rep) for mul, rep in self if mul != 0])

    def simplify(self) -> Irreps:
        """
        Simplify the representation by removing zero multiplicities and merging consecutive tuples.

        Examples:
            >>> Irreps("SO3", "1 + 0x2 + 1").simplify()
            2x1
        """
        return self.remove_zero_multiplicities().merge_consecutive()

    def sort(self) -> SortResult:
        """
        Sort the representation.

        Returns:
            SortResult: The sorted representation and associated permutations.

        Examples:
            >>> Irreps("SO3", "1 + 2 + 0 + 1").sort()
            SortResult(irreps=0+1+1+2, perm=(1, 3, 0, 2), inv=(2, 0, 3, 1))
        """

        def inverse(p):
            return tuple(p.index(i) for i in range(len(p)))

        out = sorted([(rep, i, mul) for i, (mul, rep) in enumerate(self)])
        inv = tuple(i for rep, i, mul in out)
        perm = inverse(inv)
        irreps = Irreps(self.irrep_class, [(mul, rep) for rep, i, mul in out])
        return SortResult(irreps, perm, inv)

    def regroup(self) -> Irreps:
        """
        Regroup the representation by sorting and simplifying.

        Examples:
            >>> Irreps("SO3", "1 + 2 + 0 + 1").regroup()
            0+2x1+2
        """
        return self.sort().irreps.simplify()

    def set_mul(self, mul: int) -> Irreps:
        """
        Set the multiplicity of all segments.

        Examples:
            >>> Irreps("SO3", "3x0 + 2x0 + 4x1").set_mul(2)
            2x0+2x0+2x1
        """
        return Irreps(self.irrep_class, [(mul, rep) for _, rep in self])

    def filter(
        self,
        *,
        keep: Union[str, Sequence[cue.Irrep], Callable[[MulIrrep], bool], None] = None,
        drop: Union[str, Sequence[cue.Irrep], Callable[[MulIrrep], bool], None] = None,
        mask: Optional[Sequence[bool]] = None,
    ) -> Irreps:
        """
        Filter the representation.

        Args:
            keep (str, list of Irrep, callable, optional): Keep only the specified representations.
            drop (str, list of Irrep, callable, optional): Drop the specified representations.

        Examples:
            >>> Irreps("SO3", "4x0 + 4x1 + 2x2").filter(keep="0 + 1")
            4x0+4x1

            >>> Irreps("SO3", "4x0 + 4x1 + 2x2").filter(drop="0 + 1")
            2x2
        """
        if mask is None:
            mask = self.filter_mask(keep=keep, drop=drop)
        return Irreps(
            self.irrep_class, [mulrep for keep, mulrep in zip(mask, self) if keep]
        )

    def filter_mask(
        self,
        *,
        keep: Union[str, Sequence[cue.Irrep], Callable[[MulIrrep], bool], None] = None,
        drop: Union[str, Sequence[cue.Irrep], Callable[[MulIrrep], bool], None] = None,
    ) -> list[bool]:
        if keep is not None:
            if drop is not None:
                raise ValueError(
                    "Only one of `keep` or `drop` must be defined."
                )  # pragma: no cover
            else:
                return self._filter_keep(keep)
        else:
            if drop is not None:
                return self._filter_drop(drop)
            else:
                raise ValueError(
                    "One of `keep` or `drop` must be defined."
                )  # pragma: no cover

    def _filter_keep(
        self, keep: Union[str, Sequence[cue.Irrep], Callable[[MulIrrep], bool]]
    ):
        if callable(keep):
            return [keep(mulrep) for mulrep in self]

        from .irrep_utils import into_list_of_irrep

        keep = into_list_of_irrep(self.irrep_class, keep)

        if not all(isinstance(rep, cue.Irrep) for rep in keep):
            raise ValueError(f"Invalid `keep` {keep}, expected a list of Irrep")

        return [mulrep.ir in keep for mulrep in self]

    def _filter_drop(
        self, drop: Union[str, Sequence[cue.Irrep], Callable[[MulIrrep], bool]]
    ):
        if callable(drop):
            return [not drop(mulrep) for mulrep in self]

        from .irrep_utils import into_list_of_irrep

        drop = into_list_of_irrep(self.irrep_class, drop)

        if not all(isinstance(rep, cue.Irrep) for rep in drop):
            raise ValueError(f"Invalid `drop` {drop}, expected a list of Irrep")

        return [mulrep.ir not in drop for mulrep in self]

    def layout_insensitive(self) -> bool:
        """
        True if the representation is layout insensitive.

        Examples:
            >>> Irreps("SO3", "100x0 + 1x1 + 1x2").layout_insensitive()
            True

            >>> Irreps("SO3", "100x0 + 2x1").layout_insensitive()
            False
        """
        for mul, ir in self:
            if mul > 1 and ir.dim > 1:
                return False
        return True


class SortResult(NamedTuple):
    irreps: Irreps
    perm: tuple[int, ...]
    inv: tuple[int, ...]
