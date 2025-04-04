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
from functools import cache

import sympy

import cuequivariance as cue
from cuequivariance.etc.sympy_utils import sqrtQarray_to_sympy


def spherical_harmonics(
    ir_vec: cue.Irrep, ls: list[int], layout: cue.IrrepsLayout = cue.ir_mul
) -> cue.EquivariantPolynomial:
    """Polynomial descriptor for the spherical harmonics.

    Subscripts: ``vector[],...,vector[],Yl[]``

    Args:
        ir_vec (Irrep): irrep of the input vector, for example ``cue.SO3(1)``.
        ls (list of int): list of spherical harmonic degrees, for example ``[0, 1, 2]``.
        layout (IrrepsLayout, optional): layout of the output. Defaults to ``cue.ir_mul``.

    Returns:
        :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: The descriptor.

    Example:
        >>> spherical_harmonics(cue.SO3(1), [0, 1, 2])
        ╭ a=1 -> B=0+1+2
        │  []➜B[] ───────── num_paths=1
        │  []·a[]➜B[] ───── num_paths=3
        ╰─ []·a[]·a[]➜B[] ─ num_paths=11
    """
    if len(ls) != 1:
        return cue.EquivariantPolynomial.stack(
            [spherical_harmonics(ir_vec, [ell], layout) for ell in ls], [False, True]
        )

    [ell] = ls
    ir, formula = sympy_spherical_harmonics(ir_vec, ell)

    assert ir_vec.dim == 3
    d = cue.SegmentedTensorProduct.empty_segments([3] * ell + [ir.dim])
    for i in range(ir.dim):
        for degrees, coeff in (
            sympy.Poly(formula[i], sympy.symbols("x:3")).as_dict().items()
        ):
            indices = poly_degrees_to_path_indices(degrees)
            d.add_path(*indices, i, c=coeff)

    d = d.symmetrize_operands(range(ell), force=True)

    return cue.EquivariantPolynomial(
        [cue.IrrepsAndLayout(cue.Irreps(ir_vec), cue.ir_mul)],
        [cue.IrrepsAndLayout(cue.Irreps(ir), cue.ir_mul)],
        cue.SegmentedPolynomial(
            [cue.SegmentedOperand([()] * 3)],
            [cue.SegmentedOperand([()] * ir.dim)],
            [(cue.Operation([0] * ell + [1]), d)],
        ),
    )


def poly_degrees_to_path_indices(degrees: tuple[int, ...]) -> tuple[int, ...]:
    # (1, 0, 3) -> (0, 2, 2, 2)
    # (3, 0, 1) -> (0, 0, 0, 2)
    return sum(((i,) * d for i, d in enumerate(degrees)), ())


# The function sympy_spherical_harmonics below is a 1:1 adaptation of https://github.com/e3nn/e3nn-jax/blob/c1a1adda485b8de756df56c656ce1d0cece73b64/e3nn_jax/_src/spherical_harmonics/recursive.py
@cache
def sympy_spherical_harmonics(
    ir_vec: cue.Irrep, ell: int
) -> tuple[cue.Irrep, sympy.Array]:
    if ell == 0:
        return ir_vec.trivial(), sympy.Array([1])

    if ell == 1:
        assert ir_vec.dim == 3
        x = sympy.symbols("x:3")
        return ir_vec, sympy.sqrt(3) * sympy.Array([x[0], x[1], x[2]])

    l2 = ell // 2
    l1 = ell - l2
    ir1, yl1 = sympy_spherical_harmonics(ir_vec, l1)
    ir2, yl2 = sympy_spherical_harmonics(ir_vec, l2)
    ir = sorted(cue.selection_rule_product(ir1, ir2))[-1]

    def sh_var(ir: cue.Irrep, ell: int) -> list[sympy.Symbol]:
        return [sympy.symbols(f"sh{ell}_{m}") for m in range(ir.dim)]

    cg = sqrtQarray_to_sympy(ir_vec.clebsch_gordan(ir1, ir2, ir).squeeze(0))
    yl = sympy.Array(
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
    norm = sympy.sqrt(sum(cst.applyfunc(lambda x: x**2)))

    y = sympy.sqrt(sympy.Integer(ir.dim)) * y / norm
    return ir, sympy.simplify(y)
