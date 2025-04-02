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

import re
from typing import Union

SEP = ","


class Subscripts(str):
    """
    Represent the subscripts of a Segmented Tensor Product.

    Examples:
        - "ui,uv,vi" could be the subscripts of an equivariant linear layer with shared weights.
        - "ui,vj,uvk+ijk" could be the subscripts of a full tensor product (without weights but with Clebsch-Gordan coefficients).
    """

    def __new__(cls, subscripts: Union[str, list[str]]):
        if not isinstance(subscripts, str):
            subscripts = SEP.join(subscripts)
        subscripts = subscripts.replace("_", ",")
        if not Subscripts.is_valid(subscripts):
            raise ValueError(f"invalid subscripts {subscripts}.")
        return super().__new__(cls, subscripts)

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def is_valid(subscripts: str) -> bool:
        """
        Verify if a given string is a valid tensor product subscripts.

        Args:
            subscripts (str): The subscripts string.

        Returns:
            bool: Whether the subscripts is valid.
        """
        if not isinstance(subscripts, str):
            return False
        mode = r"[a-z*]"
        if re.match(rf"^{mode}*({SEP}{mode}*)*(\+{mode}*)?$", subscripts) is None:
            return False
        operands_and_coefficients = re.split(rf"[{SEP}+]", subscripts)
        for x in operands_and_coefficients:
            if len(set(x)) < len(x):
                return False
        return True

    @classmethod
    def from_operands(cls, operands: list[str], coefficients: str = "") -> Subscripts:
        """
        Create a subscripts from a list of operands.

        Args:
            operands (list of str): The list of operands.
            coefficients (str, optional): The coefficients, by default "".

        Returns:
            Subscripts: The subscripts.

        Examples:
            >>> Subscripts.from_operands(["ui", "uv", "vi"], "ijk")
            'ui,uv,vi+ijk'
        """
        if coefficients:
            return cls(SEP.join(operands) + "+" + coefficients)
        return cls(SEP.join(operands))

    @classmethod
    def complete_wildcards(
        cls, subscripts_with_wildcards: Subscripts, reference: Subscripts
    ) -> Subscripts:
        subscripts_with_wildcards = Subscripts(subscripts_with_wildcards)
        reference = Subscripts(reference)

        if "*" in reference:
            raise ValueError("reference subscripts must not contain wildcards.")

        if subscripts_with_wildcards.num_operands != reference.num_operands:
            raise ValueError(
                f"expected the same number of operands for both subscripts, {subscripts_with_wildcards} and {reference}."
            )

        new_subscripts = []
        for x, y in zip(
            reference.operands_and_coefficients,
            subscripts_with_wildcards.operands_and_coefficients,
        ):
            if "*" in y:
                assert y.count("*") == 1
                i = y.index("*")
                n = len(x) - len(y) + 1
                y = y[:i] + x[i : i + n] + y[i + 1 :]
            new_subscripts.append(y)
        return cls.from_operands(new_subscripts[:-1], new_subscripts[-1])

    def modes(self) -> list[str]:
        """
        Return the list of modes in the subscripts.

        Returns:
            list of str: The list of modes.

        Examples:
            >>> Subscripts("ui,uv,vi").modes()
            ['u', 'i', 'v']
        """
        modes = []
        for m in self:
            if m.isalpha() and m not in modes:
                modes.append(m)
        return modes

    def remove_mode(self, mode: str) -> Subscripts:
        """
        Remove a mode from the subscripts.

        Args:
            mode (str): The mode to remove.

        Returns:
            Subscripts: The subscripts without the mode.

        Examples:
            >>> Subscripts("ui,uv,vi").remove_mode("u")
            'i,v,vi'
        """
        assert len(mode) == 1
        return Subscripts(str(self).replace(mode, ""))

    def canonicalize(self) -> Subscripts:
        """
        Return the canonical form of the subscripts.

        Returns:
            Subscripts: The canonical form of the subscripts.

        Examples:
            >>> Subscripts("ab,b,a").canonicalize()
            'uv,v,u'

            >>> Subscripts("ab,aj,bi+ij").canonicalize()
            'uv,ui,vj+ji'
        """
        canonical_modes = "uvwabcdefghxyz"
        canonical_coeff = "ijklmnopqrst"

        mapping = dict()
        for m in self.modes():
            if m not in mapping:
                if m in self.coefficients.modes():
                    mapping[m] = canonical_coeff[0]
                    canonical_coeff = canonical_coeff[1:]
                else:
                    mapping[m] = canonical_modes[0]
                    canonical_modes = canonical_modes[1:]

        letters = sorted({m for m in self if m.isalpha()})

        subscripts = str(self)
        for i, m in enumerate(letters):
            subscripts = subscripts.replace(m, f".{i}")

        for i, m in enumerate(letters):
            subscripts = subscripts.replace(f".{i}", mapping[m])

        if subscripts.endswith("+"):
            subscripts = subscripts[:-1]

        return Subscripts(subscripts)

    def is_equivalent(self, other: Subscripts) -> bool:
        """
        Check if two subscripts are equivalent.

        Args:
            other (Subscripts): The other subscripts.

        Returns:
            bool: Whether the two subscripts are equivalent.

        Examples:
            >>> Subscripts("ui,uv,vi").is_equivalent("aj,ab,bj")
            True
        """
        return self.canonicalize() == Subscripts(other).canonicalize()

    @property
    def coefficients(self) -> Subscripts:
        """
        Return the coefficients part of the subscripts.

        Returns:
            Subscripts: The coefficients part of the subscripts.

        Examples:
            >>> Subscripts("ui_uv_vj+ij").coefficients
            'ij'
        """
        if "+" in self:
            return Subscripts(self.split("+")[1])
        return Subscripts("")

    @property
    def operands(self) -> tuple[Subscripts, ...]:
        """
        Return the subscripts of the operands.

        Returns:
            tuple of Subscripts: The subscripts of the operands.

        Examples:
            >>> Subscripts("ui,uv,vj+ij").operands
            ('ui', 'uv', 'vj')
        """
        x = self
        if "+" in x:
            x = x.split("+")[0]
        return tuple(map(Subscripts, x.split(SEP)))

    @property
    def operands_and_coefficients(self) -> tuple[Subscripts, ...]:
        """
        Return the operands and coefficients of the subscripts.

        Returns:
            tuple of Subscripts: The operands and coefficients of the subscripts.

        Examples:
            >>> Subscripts("ui,uv,vj+ij").operands_and_coefficients
            ('ui', 'uv', 'vj', 'ij')
        """
        return self.operands + (self.coefficients,)

    @property
    def num_operands(self) -> int:
        """
        Return the number of operands.

        Returns:
            int: The number of operands.

        Examples:
            >>> Subscripts("ui,uv,vj+ij").num_operands
            3
        """
        return len(self.operands)

    def is_subset_of(self, other: Subscripts) -> list[dict[str, str]]:
        """
        Check if the subscripts is a subset of another subscripts.

        Args:
            other (Subscripts): The other subscripts.

        Returns:
            list of dict of str, str: A list of dictionaries mapping characters from the subset to the other subscripts.

        Examples:
            >>> Subscripts("a_a").is_subset_of("u,u")
            [{'a': 'u'}]

            >>> Subscripts("a_a").is_subset_of("uv_vu")
            [{'a': 'v'}, {'a': 'u'}]

            >>> Subscripts("ab_b_a").is_subset_of("uv_u_v")
            []
        """
        other = Subscripts(other)

        if self.num_operands != other.num_operands:
            return []

        if len(self.coefficients) > len(other.coefficients):
            return []

        for i in range(self.num_operands):
            if len(self.operands[i]) > len(other.operands[i]):
                return []

        if self.is_equivalent(other):
            return [{a: b for a, b in zip(self.modes(), other.modes())}]

        return [
            mapping
            for m in other.modes()
            if m not in other.coefficients
            for mapping in self.is_subset_of(other.remove_mode(m))
        ]

    def __mul__(self, other: Subscripts) -> Subscripts:
        """
        Compute the product of two subscripts.

        Args:
            other (Subscripts): The other subscripts.

        Returns:
            Subscripts: The product of the two subscripts.

        Examples:
            >>> Subscripts("c,c") * Subscripts("ab,ba")
            'cab,cba'

        Raises:
            ValueError: If the subscripts do not have the same number of operands.
        """
        other = Subscripts(other)

        if self.num_operands != other.num_operands:
            raise ValueError("subscripts must have the same number of operands.")

        return Subscripts.from_operands(
            [a + b for a, b in zip(self.operands, other.operands)],
            self.coefficients + other.coefficients,
        )

    def __rmul__(self, other: Subscripts) -> Subscripts:
        return Subscripts(other) * self

    def flattenable_powerset(self) -> list[frozenset[str]]:
        # uv_vu -> [uv]
        candidates = {
            frozenset(operand[:i])
            for operand in self.operands
            for i in range(1, len(operand) + 1)
        }
        results = []
        for candidate in candidates:
            modes = "".join(candidate)
            pattern = re.compile(rf"^([{modes}]*)([^{modes}]*)$")
            if all(pattern.match(operand) for operand in self.operands):
                results.append(candidate)
        return sorted(results, key=lambda x: len(x))

    def modes_on_the_left(self, mode: str) -> list[str]:
        out = ""
        for operand in self.operands:
            out += operand.split(mode)[0]
        return [m for m in self.modes() if m in out]
