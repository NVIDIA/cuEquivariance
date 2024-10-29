# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import importlib.resources

__version__ = (
    importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()
)

from .primitives.tensor_product import TensorProduct
from .primitives.symmetric_tensor_product import (
    SymmetricTensorProduct,
    IWeightedSymmetricTensorProduct,
)
from .primitives.transpose import TransposeSegments, TransposeIrrepsLayout

from .operations.equivariant_tensor_product import EquivariantTensorProduct
from .operations.tp_channel_wise import ChannelWiseTensorProduct
from .operations.tp_fully_connected import FullyConnectedTensorProduct
from .operations.linear import Linear
from .operations.symmetric_contraction import SymmetricContraction
from .operations.rotation import (
    Rotation,
    encode_rotation_angle,
    vector_to_euler_angles,
    Inversion,
)
from .operations.spherical_harmonics import spherical_harmonics

from cuequivariance_torch import layers

__all__ = [
    "TensorProduct",
    "SymmetricTensorProduct",
    "IWeightedSymmetricTensorProduct",
    "TransposeSegments",
    "TransposeIrrepsLayout",
    "EquivariantTensorProduct",
    "ChannelWiseTensorProduct",
    "FullyConnectedTensorProduct",
    "Linear",
    "SymmetricContraction",
    "Rotation",
    "Inversion",
    "encode_rotation_angle",
    "vector_to_euler_angles",
    "spherical_harmonics",
    "layers",
]
