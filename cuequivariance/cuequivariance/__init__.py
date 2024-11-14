# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
import importlib.resources

__version__ = (
    importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()
)

from cuequivariance.representation import (
    Rep,
    Irrep,
    clebsch_gordan,
    selection_rule_product,
    selection_rule_power,
    SU2,
    SO3,
    O3,
)

from cuequivariance.irreps_array import (
    get_irrep_scope,
    MulIrrep,
    Irreps,
    IrrepsLayout,
    mul_ir,
    ir_mul,
    get_layout_scope,
    assume,
    NumpyIrrepsArray,
    from_segments,
    concatenate,
    reduced_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
    reduced_antisymmetric_tensor_product_basis,
)

from cuequivariance.segmented_tensor_product import SegmentedTensorProduct
from cuequivariance.equivariant_tensor_product import EquivariantTensorProduct
from cuequivariance.tensor_product_execution import TensorProductExecution

from cuequivariance import (
    segmented_tensor_product,
    descriptors,
    tensor_product_execution,
)

__all__ = [
    "Rep",
    "Irrep",
    "clebsch_gordan",
    "selection_rule_product",
    "selection_rule_power",
    "SU2",
    "SO3",
    "O3",
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
    "SegmentedTensorProduct",
    "EquivariantTensorProduct",
    "TensorProductExecution",
    "segmented_tensor_product",
    "descriptors",
    "tensor_product_execution",
]
