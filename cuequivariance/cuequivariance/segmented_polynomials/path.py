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

import numpy as np


def np_asarray_with_copy(arr):
    """Create a NumPy array with a copy in a version-compatible way."""
    dtype = np.float64
    order = "C"
    try:
        return np.asarray(arr, dtype=dtype, order=order, copy=True)
    except TypeError:
        return np.asarray(arr, dtype=dtype, order=order).copy()


@dataclasses.dataclass(init=False, frozen=True)
class Path:
    """
    A tensor product path links segments from different operands and contains coefficients.

    Attributes:
        indices (tuple of int): One index per operand, pointing to the segment.
        coefficients (np.ndarray): Coefficients of the path.

    Examples:
        >>> Path((12, 44), 2.0)
        op0[12]*op1[44]*2.
    """

    indices: tuple[int, ...]  # One index per operand, pointing to the segment
    coefficients: np.ndarray

    def __init__(self, indices, coefficients):
        super().__setattr__("indices", tuple(int(i) for i in indices))
        super().__setattr__("coefficients", np_asarray_with_copy(coefficients))

    def assert_valid(self):
        """Assert that the path is valid."""
        if not isinstance(self.indices, tuple):
            raise ValueError(f"indices {self.indices} are not valid.")

        if not all(isinstance(i, int) and i >= 0 for i in self.indices):
            raise ValueError(f"indices {self.indices} are not valid.")

        if not self.coefficients.dtype == np.float64:
            raise ValueError(f"coefficients {self.coefficients} are not valid.")

    def __repr__(self) -> str:
        indices_txt = "*".join(f"op{i}[{s}]" for i, s in enumerate(self.indices))
        txt = np.array2string(
            self.coefficients, separator=" ", precision=2, suppress_small=True
        )
        if "\n" not in txt:
            return f"{indices_txt}*{txt}"
        return f"{indices_txt}*c c.shape={self.coefficients.shape} c.nnz={np.count_nonzero(self.coefficients)}"

    @property
    def num_operands(self) -> int:
        """The number of operands."""
        return len(self.indices)

    def __hash__(self) -> int:
        return hash((self.indices, self.coefficients.tobytes()))

    def __eq__(self, other: Path) -> bool:
        return self.indices == other.indices and np.array_equal(
            self.coefficients, other.coefficients
        )

    def __lt__(self, other: Path) -> bool:
        k1 = (self.indices, self.coefficients.shape)
        k2 = (other.indices, other.coefficients.shape)
        if k1 != k2:
            return k1 < k2
        return tuple(self.coefficients.flatten()) < tuple(other.coefficients.flatten())

    def permute_operands(self, perm: tuple[int, ...]) -> Path:
        """
        Apply a permutation to the operands.

        Args:
            perm (tuple of int): The permutation of the operands.

        Returns:
            Path: A new path with the operands permuted.
        """
        return Path(
            indices=tuple(self.indices[i] for i in perm),
            coefficients=self.coefficients,
        )

    def move_operand(self, operand: int, new_index: int) -> Path:
        """Move an operand to a new position."""
        perm = list(range(self.num_operands))
        perm.remove(operand)
        perm.insert(new_index, operand)
        return self.permute_operands(perm)

    def move_operand_first(self, operand: int) -> Path:
        """Move an operand to the first position."""
        return self.move_operand(operand, 0)

    def move_operand_last(self, operand: int) -> Path:
        """Move an operand to the last position."""
        return self.move_operand(operand, self.num_operands - 1)
