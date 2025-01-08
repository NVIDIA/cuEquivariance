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
from functools import cache

import sympy as sp

import cuequivariance as cue
from cuequivariance import segmented_tensor_product as stp
from cuequivariance.misc.sympy_utils import sqrtQarray_to_sympy


def spherical_harmonics(
    ir_vec: cue.Irrep, ls: list[int], layout: cue.IrrepsLayout = cue.ir_mul
) -> cue.EquivariantTensorProduct:
    """
    subscripts: ``vector[],...,vector[],Yl[]``

    Args:
        ir_vec (Irrep): irrep of the input vector, for example ``cue.SO3(1)``.
        ls (list of int): list of spherical harmonic degrees, for example ``[0, 1, 2]``.
        layout (IrrepsLayout, optional): layout of the output. Defaults to ``cue.ir_mul``.

    Returns:
        :class:`cue.EquivariantTensorProduct <cuequivariance.EquivariantTensorProduct>`: The descriptor.

    Examples:
        >>> spherical_harmonics(cue.SO3(1), [0, 1, 2])
        EquivariantTensorProduct((1)^(0..2) -> 0+1+2)
    """
    if len(ls) != 1:
        return cue.EquivariantTensorProduct.stack(
            [spherical_harmonics(ir_vec, [ell], layout) for ell in ls], [False, True]
        )

    [ell] = ls
    ir, formula = sympy_spherical_harmonics(ir_vec, ell)

    assert ir_vec.dim == 3
    d = stp.SegmentedTensorProduct.empty_segments([3] * ell + [ir.dim])
    for i in range(ir.dim):
        for degrees, coeff in sp.Poly(formula[i], sp.symbols("x:3")).as_dict().items():
            indices = poly_degrees_to_path_indices(degrees)
            d.add_path(*indices, i, c=coeff)

    return cue.EquivariantTensorProduct(
        [d],
        [
            cue.IrrepsAndLayout(cue.Irreps(ir_vec), cue.ir_mul),
            cue.IrrepsAndLayout(cue.Irreps(ir), cue.ir_mul),
        ],
    )


def poly_degrees_to_path_indices(degrees: tuple[int, ...]) -> tuple[int, ...]:
    # (1, 0, 3) -> (0, 2, 2, 2)
    # (3, 0, 1) -> (0, 0, 0, 2)
    return sum(((i,) * d for i, d in enumerate(degrees)), ())


# The function sympy_spherical_harmonics below is a 1:1 adaptation of https://github.com/e3nn/e3nn-jax/blob/c1a1adda485b8de756df56c656ce1d0cece73b64/e3nn_jax/_src/spherical_harmonics/recursive.py
@cache
def sympy_spherical_harmonics(
    ir_vec: cue.Irrep, ell: int
) -> tuple[cue.Irrep, sp.Array]:
    if ell == 0:
        return ir_vec.trivial(), sp.Array([1])

    if ell == 1:
        assert ir_vec.dim == 3
        x = sp.symbols("x:3")
        return ir_vec, sp.sqrt(3) * sp.Array([x[0], x[1], x[2]])

    l2 = ell // 2
    l1 = ell - l2
    ir1, yl1 = sympy_spherical_harmonics(ir_vec, l1)
    ir2, yl2 = sympy_spherical_harmonics(ir_vec, l2)
    ir = sorted(cue.selection_rule_product(ir1, ir2))[-1]

    def sh_var(ir: cue.Irrep, ell: int) -> list[sp.Symbol]:
        return [sp.symbols(f"sh{ell}_{m}") for m in range(ir.dim)]

    cg = sqrtQarray_to_sympy(ir_vec.clebsch_gordan(ir1, ir2, ir).squeeze(0))
    yl = sp.Array(
        [
            sum(
                sh_var(ir1, l1)[i] * sh_var(ir2, l2)[j] * cg[i, j, k]
                for i in range(ir1.dim)
                for j in range(ir2.dim)
            )
            for k in range(ir.dim)
        ]
    )

    y = yl.subs(zip(sh_var(ir1, l1), yl1)).subs(zip(sh_var(ir2, l2), yl2))

    cst = y.subs({"x0": 0, "x1": 1, "x2": 0})
    norm = sp.sqrt(sum(cst.applyfunc(lambda x: x**2)))

    y = sp.sqrt(sp.Integer(ir.dim)) * y / norm
    return ir, sp.simplify(y)
