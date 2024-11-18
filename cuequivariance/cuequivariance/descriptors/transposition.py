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
