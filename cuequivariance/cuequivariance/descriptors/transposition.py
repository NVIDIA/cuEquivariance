# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from typing import *

import cuequivariance as cue
from cuequivariance import segmented_tensor_product as stp
from cuequivariance.equivariant_tensor_product import Operand


def transpose(
    irreps: cue.Irreps, source: cue.IrrepsLayout, target: cue.IrrepsLayout
) -> cue.EquivariantTensorProduct:
    """Transpose the irreps layout of a tensor."""
    d = stp.SegmentedTensorProduct(
        operands=[
            stp.Operand(subscripts="ui" if source == cue.mul_ir else "iu"),
            stp.Operand(subscripts="ui" if target == cue.mul_ir else "iu"),
        ]
    )
    for mul, ir in irreps:
        d.add_path(None, None, c=1, dims={"u": mul, "i": ir.dim})
    return cue.EquivariantTensorProduct(
        d, [Operand(irreps, source), Operand(irreps, target)]
    )
