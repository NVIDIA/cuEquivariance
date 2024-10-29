# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import *

import cuequivariance as cue
from cuequivariance import segmented_tensor_product as stp
from cuequivariance import equivariant_tensor_product as etp


def transpose(
    irreps: cue.Irreps, source: cue.IrrepsLayout, target: cue.IrrepsLayout
) -> etp.EquivariantTensorProduct:
    """Transpose the irreps layout of a tensor."""
    d = stp.SegmentedTensorProduct(
        operands=[
            stp.Operand(subscripts="ui" if source == cue.mul_ir else "iu"),
            stp.Operand(subscripts="ui" if target == cue.mul_ir else "iu"),
        ]
    )
    for mul, ir in irreps:
        d.add_path(None, None, c=1, dims={"u": mul, "i": ir.dim})
    return etp.EquivariantTensorProduct(
        d, [etp.Operand(irreps, source), etp.Operand(irreps, target)]
    )
