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


from .irreps_array.jax_irreps_array import (
    IrrepsArray,
    from_segments,
    vmap,
)
from .irreps_array.utils import concatenate, randn, as_irreps_array

from .primitives.tensor_product import tensor_product
from .primitives.symmetric_tensor_product import symmetric_tensor_product
from .primitives.equivariant_tensor_product import equivariant_tensor_product

from .operations.activation import (
    normalspace,
    normalize_function,
    function_parity,
    scalar_activation,
)
from .operations.spherical_harmonics import spherical_harmonics, normalize, norm

from cuequivariance_jax import flax_linen

__all__ = [
    "IrrepsArray",
    "from_segments",
    "as_irreps_array",
    "vmap",
    "concatenate",
    "randn",
    "tensor_product",
    "symmetric_tensor_product",
    "equivariant_tensor_product",
    "normalspace",
    "normalize_function",
    "function_parity",
    "scalar_activation",
    "spherical_harmonics",
    "normalize",
    "norm",
    "flax_linen",
]
