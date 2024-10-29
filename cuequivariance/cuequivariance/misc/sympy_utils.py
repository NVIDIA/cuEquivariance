# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright 2023 Mario Geiger
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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
