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
from typing import *

import cuequivariance as cue
from cuequivariance import segmented_tensor_product as stp


def symmetric_contraction(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    degrees: list[int],
) -> cue.EquivariantTensorProduct:
    r"""
    subscripts: ``weights[u],input[u],output[u]``

    Construct the descriptor for a symmetric contraction.

    The symmetric contraction is a weighted sum of the input contracted with itself degree times.

    Args:
        irreps_in (Irreps): The input irreps, the multiplicity are treated in parallel.
        irreps_out (Irreps): The output irreps.
        degree (int): The degree of the symmetric contraction.

    Returns:
        EquivariantTensorProduct:
            The descriptor of the symmetric contraction.
            The operands are the weights, the input degree times and the output.

    Examples:
        >>> cue.descriptors.symmetric_contraction(
        ...    16 * cue.Irreps("SO3", "0 + 1 + 2"),
        ...    16 * cue.Irreps("SO3", "0 + 1"),
        ...    [1, 2, 3]
        ... )
        EquivariantTensorProduct(32x0+80x0+176x0 x (16x0+16x1+16x2)^(1..3) -> 16x0+16x1)

        Where ``32x0+80x0+176x0`` are the weights needed for each degree (32 for degree 1, 80 for degree 2, 176 for degree 3).
    """
    degrees = list(degrees)
    if len(degrees) != 1:
        return cue.EquivariantTensorProduct.stack(
            [
                symmetric_contraction(irreps_in, irreps_out, [degree])
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

    if degree == 0:
        d = stp.SegmentedTensorProduct.from_subscripts("i_i")
        for _, ir in irreps_out:
            if not ir.is_scalar():
                d.add_segment(output_operand, {"i": ir.dim})
            else:
                d.add_path(None, None, c=1, dims={"i": ir.dim})
        d = d.flatten_modes("i")

    else:
        abc = "abcdefgh"[:degree]
        d = stp.SegmentedTensorProduct.from_subscripts(
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
    return cue.EquivariantTensorProduct(
        [d],
        [irreps_in.new_scalars(d.operands[0].size), mul * irreps_in, mul * irreps_out],
        layout=cue.ir_mul,
    )
