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
import importlib.resources

__version__ = (
    importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()
)


from .rep_array.jax_rep_array import RepArray, from_segments, IrrepsArray
from .rep_array.vmap import vmap
from .rep_array.utils import concatenate, randn, as_irreps_array, clebsch_gordan

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
    "RepArray",
    "from_segments",
    "IrrepsArray",
    "vmap",
    "concatenate",
    "randn",
    "as_irreps_array",
    "clebsch_gordan",
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
