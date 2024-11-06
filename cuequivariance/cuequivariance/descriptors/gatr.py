# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import cuequivariance as cue


def gatr_linear(mul_in: int, mul_out: int) -> cue.SegmentedTensorProduct:
    """
    subsrcipts: weights[uv],input[iu],output[iv]

    references:
    - https://arxiv.org/pdf/2305.18415
    - https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/3f967c978445648ef83d87190d32176f7fd91565/gatr/primitives/linear.py#L33-L43
    """
    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,iv")

    for ope, mul in [(1, mul_in), (2, mul_out)]:
        one = d.add_segment(ope, (1, mul))
        e0 = d.add_segment(ope, (1, mul))
        ei = d.add_segment(ope, (3, mul))
        e0i = d.add_segment(ope, (3, mul))
        eij = d.add_segment(ope, (3, mul))
        e0ij = d.add_segment(ope, (3, mul))
        e123 = d.add_segment(ope, (1, mul))
        e0123 = d.add_segment(ope, (1, mul))

    for xs in [[one], [e0, ei], [e0i, eij], [e0ij, e123], [e0123]]:
        x, xs = xs[0], xs[1:]
        d.add_path(None, x, x, c=1)
        for y in xs:
            d.add_path(-1, y, y, c=1)

    d.add_path(None, one, e0, c=1)
    d.add_path(None, ei, e0i, c=1)
    d.add_path(None, eij, e0ij, c=1)
    d.add_path(None, e123, e0123, c=1)

    d = d.normalize_paths_for_operand(2)
    return d
