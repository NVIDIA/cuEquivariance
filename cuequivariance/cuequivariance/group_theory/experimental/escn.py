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
from typing import Optional

import numpy as np

import cuequivariance as cue


# The function escn_iu_ju_ku below is a 1:1 adaptation of https://github.com/e3nn/e3nn-jax/blob/a2a81ab451b9cd597d7be27b3e1faba79457475d/e3nn_jax/experimental/linear_shtp.py#L38-L165
def escn_tp(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    m_max: Optional[int] = None,
    l_max: Optional[int] = None,
) -> cue.EquivariantPolynomial:
    """
    subsrcipts: ``weights[uv],input[u],output[v]``

    Tensor Product part of the eSCN convolution introduced in https://arxiv.org/pdf/2302.03655.pdf

    Args:
        irreps_in (Irreps): Irreps of the input.
        irreps_out (Irreps): Irreps of the output.
        m_max (int, optional): Maximum angular resolution around the principal axis.
        l_max (int, optional): Maximum angular resolution along the principal axis.

    Returns:
        EquivariantPolynomial:
            Descriptor of the tensor product part of the eSCN convolution.

            - Operand 0: weights
            - Operand 1: input
            - Operand 2: output
    """
    assert irreps_in.irrep_class == irreps_out.irrep_class
    G = irreps_in.irrep_class
    if G not in [cue.SO3, cue.O3]:
        # TODO: we could support SU2 since it shares the same Clebsch-Gordan coefficients as SO3 and O3
        raise NotImplementedError("Only SO3 and O3 are supported")

    if l_max is not None:

        def pr(mul_ir: cue.MulIrrep) -> bool:
            ir = mul_ir.ir

            return any(abs(other.l - ir.l) <= l_max for _, other in irreps_in)

        irreps_out = irreps_out.filter(keep=pr)

    d = cue.SegmentedTensorProduct.from_subscripts("iuv,ju,kv+ijk")

    for mul, ir in irreps_in:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps_out:
        d.add_segment(2, (ir.dim, mul))

    for i1, (mul1, ir1) in enumerate(irreps_in):
        for i2, (mul2, ir2) in enumerate(irreps_out):
            ell = min(ir1.l, ir2.l)

            if l_max is not None:
                if abs(ir1.l - ir2.l) > l_max:
                    continue
            if m_max is not None:
                ell = min(ell, m_max)

            # Scaled rotation (2 degrees of freedom per |m|)
            c = np.zeros((2 * ell + 1, ir1.dim, ir2.dim))
            for m in range(-ell, ell + 1):
                # "cosine" part
                c[ell - abs(m), ir1.l + m, ir2.l + m] = 1.0

                # "sine" part
                if m != 0:
                    c[ell + abs(m), ir1.l - m, ir2.l + m] = 1.0 if m > 0 else -1.0

            if G == cue.SO3:
                pass  # keep all the degrees of freedom
            elif G == cue.O3:
                if (-1) ** (ir1.l + ir2.l) == ir1.p * ir2.p:
                    # Symmetric case: keep only the "cosine" part
                    c = c[: ell + 1]
                elif ell > 0:
                    # Antisymmetric case: keep only the "sine" part
                    c = c[ell + 1 :]
                else:
                    c = None

            if c is not None:
                d.add_path(None, i1, i2, c=c)

    d = d.normalize_paths_for_operand(2)
    d = d.flatten_coefficient_modes()
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps_in, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def escn_tp_compact(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    m_max: Optional[int] = None,
) -> cue.SegmentedPolynomial:
    """
    subsrcipts: ``weights[uv],input[u],output[v]``

    Tensor Product part of the eSCN convolution introduced in https://arxiv.org/pdf/2302.03655.pdf

    This "compact" implementation puts the L index contiguous in memory.
    This allows to create bigger segments and less paths.

    Args:
        irreps_in (Irreps): Irreps of the input.
        irreps_out (Irreps): Irreps of the output.
        m_max (int, optional): Maximum angular resolution around the principal axis.

    Returns:
        SegmentedTensorProduct:
            Descriptor of the tensor product part of the eSCN convolution.

            - Operand 0: weights
            - Operand 1: input
            - Operand 2: output
    """
    assert irreps_in.irrep_class == irreps_out.irrep_class
    G = irreps_in.irrep_class
    if G not in [cue.SO3]:
        raise NotImplementedError("Only SO3 is supported")

    d = cue.SegmentedTensorProduct.from_subscripts("uv,u,v")

    l_max_in = max(ir.l for _, ir in irreps_in)
    for m in range(-l_max_in, l_max_in + 1):
        mulirs = [(mul, ir) for mul, ir in irreps_in if abs(m) <= ir.l]
        mul = sum(mul for mul, _ in mulirs)
        d.add_segment(1, (mul,))

    l_max_out = max(ir.l for _, ir in irreps_out)
    for m in range(-l_max_out, l_max_out + 1):
        mulirs = [(mul, ir) for mul, ir in irreps_out if abs(m) <= ir.l]
        mul = sum(mul for mul, _ in mulirs)
        d.add_segment(2, (mul,))

    if m_max is None:
        m_max = min(l_max_in, l_max_out)

    # m = 0
    d.add_path(None, l_max_in, l_max_out, c=1.0)

    for m in range(1, min(m_max, l_max_in, l_max_out) + 1):
        # "cosine" part
        d.add_path(None, l_max_in - m, l_max_out - m, c=1.0)
        i = d.operands[0].num_segments - 1
        d.add_path(i, l_max_in + m, l_max_out + m, c=1.0)

        # "sine" part
        d.add_path(None, l_max_in + m, l_max_out - m, c=1.0)
        i = d.operands[0].num_segments - 1
        d.add_path(i, l_max_in - m, l_max_out + m, c=-1.0)

    d = d.normalize_paths_for_operand(2)
    # TODO: return an EquivariantPolynomial using SphericalSignal
    return cue.SegmentedPolynomial.eval_last_operand(d)


class SphericalSignal(cue.Rep):
    def __init__(self, mul: int, l_max: int, m_max: int):
        self.mul = mul
        self.l_max = l_max
        self.m_max = m_max

    # TODO
