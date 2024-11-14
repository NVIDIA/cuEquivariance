# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction
import numpy as np

import sympy


def Q_to_sympy(x: float) -> sympy.Expr:
    x = Fraction(x).limit_denominator()
    return sympy.sympify(x)


def sqrtQ_to_sympy(x: float) -> sympy.Expr:
    sign = 1 if x >= 0 else -1
    return sign * sympy.sqrt(Q_to_sympy(x**2))


def sqrtQarray_to_sympy(x: np.ndarray) -> sympy.Array:
    if x.ndim == 0:
        return sqrtQ_to_sympy(x)
    else:
        return sympy.Array([sqrtQarray_to_sympy(row) for row in x])
