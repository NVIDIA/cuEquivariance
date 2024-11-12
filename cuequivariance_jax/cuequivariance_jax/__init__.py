# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
import importlib.resources

__version__ = (
    importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()
)


from .irreps_array.jax_irreps_array import (
    IrrepsArray,
    from_segments,
    as_irreps_array,
    vmap,
)
from .irreps_array.concatenate import concatenate, randn

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
