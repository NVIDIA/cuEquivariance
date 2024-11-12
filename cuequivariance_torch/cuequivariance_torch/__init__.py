# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
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

from .primitives.equivariant_tensor_product import EquivariantTensorProduct
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
