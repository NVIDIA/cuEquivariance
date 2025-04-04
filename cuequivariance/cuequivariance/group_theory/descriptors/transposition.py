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
import cuequivariance as cue


def transpose(
    irreps: cue.Irreps, source: cue.IrrepsLayout, target: cue.IrrepsLayout
) -> cue.EquivariantPolynomial:
    """Transpose the irreps layout of a tensor."""
    d = cue.SegmentedTensorProduct(
        operands_and_subscripts=[
            (cue.SegmentedOperand(ndim=2), "ui" if source == cue.mul_ir else "iu"),
            (cue.SegmentedOperand(ndim=2), "ui" if target == cue.mul_ir else "iu"),
        ]
    )
    for mul, ir in irreps:
        d.add_path(None, None, c=1, dims={"u": mul, "i": ir.dim})
    return cue.EquivariantPolynomial(
        [cue.IrrepsAndLayout(irreps, source)],
        [cue.IrrepsAndLayout(irreps, target)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )
