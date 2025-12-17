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
import itertools
from typing import Optional, Sequence

import cuequivariance as cue
from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep


def fully_connected_tensor_product(
    irreps1: cue.Irreps, irreps2: cue.Irreps, irreps3: cue.Irreps
) -> cue.EquivariantPolynomial:
    """
    subscripts: ``weights[uvw],lhs[iu],rhs[jv],output[kw]``

    Construct a fully connected tensor product descriptor.

    The descriptor is constructed by iterating over all possible combinations of irreducible representations
    of the inputs and output irreps and constructing a path for each combination.

    .. currentmodule:: cuequivariance

    Args:
        irreps1 (Irreps): Irreps of the first operand.
        irreps2 (Irreps): Irreps of the second operand.
        irreps3 (Irreps): Irreps of the output.

    Returns:
        :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Descriptor of the fully connected tensor product.

    Examples:
        >>> cue.descriptors.fully_connected_tensor_product(
        ...    16 * cue.Irreps("SO3", "0 + 1 + 2"),
        ...    16 * cue.Irreps("SO3", "0 + 1 + 2"),
        ...    16 * cue.Irreps("SO3", "0 + 1 + 2"),
        ... )
        ╭ a=61440x0 b=16x0+16x1+16x2 c=16x0+16x1+16x2 -> D=16x0+16x1+16x2
        ╰─ [ijk]·a[uvw]·b[iu]·c[jv]➜D[kw] ─ num_paths=15 i={1, 3, 5} j={1, 3, 5} k={1, 3, 5} u=16 v=16 w=16

        Where ``61440x0`` are the 61440 weights needed to mix all the inputs with all the outputs.
    """
    G = irreps1.irrep_class

    d = cue.SegmentedTensorProduct.from_subscripts("uvw,iu,jv,kw+ijk")

    for mul, ir in irreps1:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(2, (ir.dim, mul))
    for mul, ir in irreps3:
        d.add_segment(3, (ir.dim, mul))

    for (i1, (mul1, ir1)), (i2, (mul2, ir2)), (i3, (mul3, ir3)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2), enumerate(irreps3)
    ):
        if ir3 not in ir1 * ir2:
            continue

        # for loop over the different solutions of the Clebsch-Gordan decomposition
        for cg in G.clebsch_gordan(ir1, ir2, ir3):
            d.add_path((mul1, mul2, mul3), i1, i2, i3, c=cg)

    d = d.normalize_paths_for_operand(-1)
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps1.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def full_tensor_product(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3_filter: Optional[Sequence[cue.Irrep]] = None,
) -> cue.EquivariantPolynomial:
    """
    subscripts: ``lhs[iu],rhs[jv],output[kuv]``

    Construct a weightless channelwise tensor product descriptor.

    .. currentmodule:: cuequivariance

    Args:
        irreps1 (Irreps): Irreps of the first operand.
        irreps2 (Irreps): Irreps of the second operand.
        irreps3_filter (sequence of Irrep, optional): Irreps of the output to consider.

    Returns:
        :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Descriptor of the full tensor product.
    """
    G = irreps1.irrep_class

    if irreps3_filter is not None:
        irreps3_filter = into_list_of_irrep(G, irreps3_filter)

    d = cue.SegmentedTensorProduct.from_subscripts("iu,jv,kuv+ijk")

    for mul, ir in irreps1:
        d.add_segment(0, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(1, (ir.dim, mul))

    irreps3 = []

    for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2)
    ):
        for ir3 in ir1 * ir2:
            if irreps3_filter is not None and ir3 not in irreps3_filter:
                continue

            # for loop over the different solutions of the Clebsch-Gordan decomposition
            for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(i1, i2, None, c=cg)

                irreps3.append((mul1 * mul2, ir3))

    irreps3 = cue.Irreps(G, irreps3)
    irreps3, perm, inv = irreps3.sort()
    d = d.permute_segments(2, inv)

    d = d.normalize_paths_for_operand(-1)
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def channelwise_tensor_product(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3_filter=None,
    simplify_irreps3: bool = False,
) -> cue.EquivariantPolynomial:
    """
    subscripts: ``weights[uv],lhs[iu],rhs[jv],output[kuv]``

    Construct a channelwise tensor product descriptor.

    This operation is computationally sparser than the fully connected tensor product.

    .. currentmodule:: cuequivariance

    Args:
        irreps1 (Irreps): Irreps of the first operand.
        irreps2 (Irreps): Irreps of the second operand.
        irreps3_filter (sequence of Irrep, optional): Irreps of the output to consider.
        simplify_irreps3 (bool, optional): If True, the irreps of the output are simplified.

    Returns:
        :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Descriptor of the channelwise tensor product.
    """
    G = irreps1.irrep_class

    if irreps3_filter is not None:
        irreps3_filter = into_list_of_irrep(G, irreps3_filter)

    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,jv,kuv+ijk")

    for mul, ir in irreps1:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(2, (ir.dim, mul))

    irreps3 = []

    for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2)
    ):
        for ir3 in ir1 * ir2:
            if irreps3_filter is not None and ir3 not in irreps3_filter:
                continue

            # for loop over the different solutions of the Clebsch-Gordan decomposition
            for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(None, i1, i2, None, c=cg, dims={"u": mul1, "v": mul2})

                irreps3.append((mul1 * mul2, ir3))

    irreps3 = cue.Irreps(G, irreps3)
    irreps3, perm, inv = irreps3.sort()
    d = d.permute_segments(0, inv)
    d = d.permute_segments(3, inv)

    d = d.normalize_paths_for_operand(-1)

    if simplify_irreps3:
        # We need to flatten the coefficients to do this simplification
        d = d.flatten_coefficient_modes()

        # Sort output segments to regroup them by irreps
        segments = [(ir, k) for _mul, ir in irreps3 for k in range(ir.dim)]
        segments = [(ir, k, sid) for sid, (ir, k) in enumerate(segments)]
        segments = sorted(segments)
        d = d.permute_segments(3, [sid for _, _, sid in segments])
        irreps3 = irreps3.simplify()

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps1.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def _align_two_irreps(
    irreps1: cue.Irreps, irreps2: cue.Irreps, layout: cue.IrrepsLayout
) -> tuple[cue.Irreps, cue.Irreps]:
    assert irreps1.num_irreps == irreps2.num_irreps

    l1 = list(irreps1)
    l2 = list(irreps2)

    i = 0
    while i < min(len(l1), len(l2)):
        mul_1, ir_1 = l1[i]
        mul_2, ir_2 = l2[i]

        if mul_1 < mul_2:
            assert ir_2.dim == 1 or layout == cue.mul_ir
            l2[i] = (mul_1, ir_2)
            l2.insert(i + 1, (mul_2 - mul_1, ir_2))

        if mul_2 < mul_1:
            assert ir_1.dim == 1 or layout == cue.mul_ir
            l1[i] = (mul_2, ir_1)
            l1.insert(i + 1, (mul_1 - mul_2, ir_1))

        i += 1

    assert [mul for mul, _ in l1] == [mul for mul, _ in l2], (l1, l2)
    return cue.Irreps(irreps1.irrep_class, l1), cue.Irreps(irreps2.irrep_class, l2)


def elementwise_tensor_product(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3_filter: Optional[Sequence[cue.Irrep]] = None,
) -> cue.EquivariantPolynomial:
    """
    subscripts: ``lhs[iu],rhs[ju],output[ku]``

    Construct an elementwise tensor product descriptor.

    Args:
        irreps1 (Irreps): Irreps of the first operand.
        irreps2 (Irreps): Irreps of the second operand.
        irreps3_filter (sequence of Irrep, optional): Irreps of the output to consider.

    Returns:
        :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Descriptor of the elementwise tensor product.
    """
    G = irreps1.irrep_class

    if irreps1.num_irreps != irreps2.num_irreps:
        raise ValueError(
            f"The input irreps must have the same number of irreps, got {irreps1} and {irreps2}"
        )

    irreps1_cut, irreps2_cut = _align_two_irreps(irreps1, irreps2, cue.ir_mul)

    d = cue.SegmentedTensorProduct.from_subscripts("iu,ju,ku+ijk")

    irreps3 = []
    for (mul, ir1), (_, ir2) in zip(irreps1_cut, irreps2_cut):
        i1 = d.add_segment(0, (ir1.dim, mul))
        i2 = d.add_segment(1, (ir2.dim, mul))

        for ir3 in ir1 * ir2:
            if irreps3_filter is not None and ir3 not in irreps3_filter:
                continue

            for cg in G.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(i1, i2, None, c=cg)

                irreps3.append((mul, ir3))

    irreps3 = cue.Irreps(G, irreps3)
    d = d.normalize_paths_for_operand(-1)
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def linear(irreps_in: cue.Irreps, irreps_out: cue.Irreps) -> cue.EquivariantPolynomial:
    """
    subscripts: ``weights[uv],input[iu],output[iv]``

    Construct the descriptor of a linear equivariant transformation.

    Args:
        irreps_in (Irreps): Irreps of the input.
        irreps_out (Irreps): Irreps of the output.

    Returns:
        :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Descriptor of the linear transformation.
    """
    d = cue.SegmentedTensorProduct.from_subscripts("uv_iu_iv")
    for mul, ir in irreps_in:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps_out:
        d.add_segment(2, (ir.dim, mul))

    for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps_in), enumerate(irreps_out)
    ):
        if ir1 == ir2:
            d.add_path((mul1, mul2), i1, i2, c=1.0)

    d = d.normalize_paths_for_operand(-1)

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps_in, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )
