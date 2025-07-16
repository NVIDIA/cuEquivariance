# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from .rep_array.rep_array_ import RepArray, from_segments
from .rep_array.vmap import vmap
from .rep_array.rep_array_utils import (
    concatenate,
    randn,
    as_irreps_array,
    clebsch_gordan,
)

from .segmented_polynomials.utils import Repeats
from .segmented_polynomials.segmented_polynomial import segmented_polynomial
from .equivariant_polynomial import equivariant_polynomial

from .activation import (
    normalspace,
    normalize_function,
    function_parity,
    scalar_activation,
)
from .spherical_harmonics import spherical_harmonics, normalize, norm
from .triangle import (
    triangle_multiplicative_update,
    Precision as TriMulPrecision,
    triangle_attention,
)
from cuequivariance_jax import flax_linen
from cuequivariance_jax import experimental

__all__ = [
    "RepArray",
    "from_segments",
    "vmap",
    "concatenate",
    "randn",
    "as_irreps_array",
    "clebsch_gordan",
    "Repeats",
    "segmented_polynomial",
    "equivariant_polynomial",
    "normalspace",
    "normalize_function",
    "function_parity",
    "scalar_activation",
    "spherical_harmonics",
    "normalize",
    "norm",
    "triangle_multiplicative_update",
    "TriMulPrecision",
    "triangle_attention",
    "flax_linen",
    "experimental",
]
