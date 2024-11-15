# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum, auto
from typing import *

import cuequivariance as cue


class IrrepsLayout(Enum):
    """
    Enum for the possible data layouts.

    Attributes:
        mul_ir: Multiplicity first, then irreducible representation.
            This layout corresponds to the layout used in the library `e3nn`.
        ir_mul: Irreducible representation first, then multiplicity.
            This layout differs from the one used in `e3nn` but can be more convenient in some cases.

    Examples:
        >>> cue.mul_ir
        (mul,irrep)

        >>> cue.ir_mul
        (irrep,mul)

    .. rubric:: Methods
    """

    mul_ir = auto()
    ir_mul = auto()

    def shape(
        self, mulir: Union[cue.MulIrrep, tuple[int, cue.Irrep]]
    ) -> tuple[int, int]:
        """The shape of the tensor for the given layout.

        Examples:
            >>> mulir = cue.MulIrrep(32, cue.O3(2, -1))
            >>> mulir
            32x2o

            >>> cue.mul_ir.shape(mulir)
            (32, 5)

            >>> cue.ir_mul.shape(mulir)
            (5, 32)
        """
        mul, ir = mulir
        if self == IrrepsLayout.mul_ir:
            return (mul, ir.dim)
        return (ir.dim, mul)

    def __repr__(self) -> str:
        if self == IrrepsLayout.mul_ir:
            return "(mul,irrep)"
        if self == IrrepsLayout.ir_mul:
            return "(irrep,mul)"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def as_layout(layout: Union[str, IrrepsLayout, None]) -> IrrepsLayout:
        if isinstance(layout, IrrepsLayout):
            return layout
        if layout is None:
            return cue.get_layout_scope()
        try:
            return cue.IrrepsLayout[layout]
        except KeyError:
            raise ValueError(f"Invalid layout {layout}")
