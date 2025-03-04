# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from .subscripts import Subscripts
from .path import Path
from .segmented_operand import SegmentedOperand
from .segmented_tensor_product import SegmentedTensorProduct
from .dot import dot, trace

from .evaluate import compute_last_operand, primitive_compute_last_operand
from .dispatch import dispatch

from .operation import Operation
from .segmented_polynomial import SegmentedPolynomial


__all__ = [
    "Subscripts",
    "Path",
    "SegmentedOperand",
    "SegmentedTensorProduct",
    "dot",
    "trace",
    "compute_last_operand",
    "primitive_compute_last_operand",
    "dispatch",
    "Operation",
    "SegmentedPolynomial",
]
