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

import cuequivariance as cue


def symmetric_contraction(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    degrees: tuple[int, ...],
) -> cue.EquivariantPolynomial:
    """Construct the descriptor for a symmetric contraction.

    The symmetric contraction is a weighted sum of the input contracted with itself degree times.

    Subscripts: ``weights[u],input[u],output[u]``

    Args:
        irreps_in (Irreps): The input irreps, the multiplicity are treated in parallel.
        irreps_out (Irreps): The output irreps.
        degrees (tuple[int, ...]): List of degrees for the symmetric contractions.

    Returns:
        EquivariantPolynomial: The descriptor of the symmetric contraction.
            The operands are the weights, the input degree times and the output.

    Example:
        >>> cue.descriptors.symmetric_contraction(
        ...    16 * cue.Irreps("SO3", "0 + 1 + 2"),
        ...    16 * cue.Irreps("SO3", "0 + 1"),
        ...    (1, 2, 3)
        ... )
        ╭ a=32x0+80x0+176x0 b=16x0+16x1+16x2 -> C=16x0+16x1
        │  []·a[u]·b[u]➜C[u] ─────────── num_paths=4 u=16
        │  []·a[u]·b[u]·b[u]➜C[u] ────── num_paths=37 u=16
        ╰─ []·a[u]·b[u]·b[u]·b[u]➜C[u] ─ num_paths=437 u=16

        Where ``32x0+80x0+176x0`` are the weights needed for each degree (32 for degree 1, 80 for degree 2, 176 for degree 3).
    """
    return symmetric_contraction_cached(irreps_in, irreps_out, tuple(degrees))


@cache
def symmetric_contraction_cached(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    degrees: tuple[int, ...],
) -> cue.EquivariantPolynomial:
    degrees = list(degrees)
    if len(degrees) != 1:
        return cue.EquivariantPolynomial.stack(
            [
                symmetric_contraction(irreps_in, irreps_out, (degree,))
                for degree in degrees
            ],
            [True, False, False],
        )
    [degree] = degrees
    del degrees

    mul = irreps_in.muls[0]
    assert all(mul == m for m in irreps_in.muls)
    assert all(mul == m for m in irreps_out.muls)
    irreps_in = irreps_in.set_mul(1)
    irreps_out = irreps_out.set_mul(1)

    input_operands = range(1, degree + 1)
    output_operand = degree + 1

    input_operand = cue.SegmentedOperand(ndim=1, segments=[(mul,)] * irreps_in.dim)

    if degree == 0:
        d = cue.SegmentedTensorProduct.from_subscripts("i_i")
        for _, ir in irreps_out:
            if not ir.is_scalar():
                d.add_segment(output_operand, {"i": ir.dim})
            else:
                d.add_path(None, None, c=1, dims={"i": ir.dim})
        d = d.flatten_modes("i")

    else:
        abc = "abcdefgh"[:degree]
        d = cue.SegmentedTensorProduct.from_subscripts(
            f"w_{'_'.join(f'{a}' for a in abc)}_i+{abc}iw"
        )

        for i in input_operands:
            d.add_segment(i, (irreps_in.dim,))

        U = cue.reduced_symmetric_tensor_product_basis(
            irreps_in, degree, keep_ir=irreps_out, layout=cue.ir_mul
        )
        for _, ir in irreps_out:
            u = U.filter(keep=ir)
            if len(u.segments) == 0:
                d.add_segment(output_operand, {"i": ir.dim})
            else:
                [u] = u.segments  # (a, b, c, ..., i, w)
                d.add_path(None, *(0,) * degree, None, c=u)

        d = d.normalize_paths_for_operand(output_operand)
        d = d.flatten_coefficient_modes()

    d = d.append_modes_to_all_operands("u", {"u": mul})
    for i in input_operands:
        assert d.operands[i] == input_operand

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(mul * irreps_in, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(mul * irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial(
            [d.operands[0], input_operand],
            [d.operands[-1]],
            [(cue.Operation([0] + [1] * degree + [2]), d)],
        ),
    )
