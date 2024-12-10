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
from .context_irrep_class import get_irrep_scope
from .irreps import MulIrrep, Irreps
from .irreps_layout import IrrepsLayout, mul_ir, ir_mul
from .irreps_and_layout import IrrepsAndLayout

from .context_layout import get_layout_scope
from .context_decorator import assume

from .numpy_irreps_array import NumpyIrrepsArray, from_segments, concatenate
from .reduced_tensor_product import (
    reduced_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
    reduced_antisymmetric_tensor_product_basis,
)

__all__ = [
    "get_irrep_scope",
    "MulIrrep",
    "Irreps",
    "IrrepsLayout",
    "mul_ir",
    "ir_mul",
    "IrrepsAndLayout",
    "get_layout_scope",
    "assume",
    "NumpyIrrepsArray",
    "from_segments",
    "concatenate",
    "reduced_tensor_product_basis",
    "reduced_symmetric_tensor_product_basis",
    "reduced_antisymmetric_tensor_product_basis",
]
