# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from .context_irrep_class import get_irrep_scope
from .irreps import MulIrrep, Irreps
from .irreps_layout import IrrepsLayout

mul_ir = IrrepsLayout.mul_ir
ir_mul = IrrepsLayout.ir_mul

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
    "get_layout_scope",
    "assume",
    "NumpyIrrepsArray",
    "from_segments",
    "concatenate",
    "reduced_tensor_product_basis",
    "reduced_symmetric_tensor_product_basis",
    "reduced_antisymmetric_tensor_product_basis",
]
