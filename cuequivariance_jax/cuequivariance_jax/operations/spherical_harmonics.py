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

import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance import descriptors


def spherical_harmonics(
    ls: list[int],
    vector: cuex.RepArray,
    normalize: bool = True,
) -> cuex.RepArray:
    """Compute the spherical harmonics of a vector.

    The spherical harmonics are polynomials of an input vector. This function computes the polynomials of the specified degrees.

    Args:
        ls (list[int]): List of spherical harmonic degrees. Each degree must be non-negative.
        vector (RepArray): Input vector. Must be a single vector (multiplicity 1) with 3 components.
        normalize (bool, optional): Whether to normalize the vector before computing the spherical harmonics. Defaults to True.

    Returns:
        RepArray: The spherical harmonics of the vector, containing the polynomials of each specified degree.

    Example:
        >>> import cuequivariance_jax as cuex
        >>> vector = cuex.randn(jax.random.key(0), cue.Irreps(cue.SO3, "1o"))
        >>> harmonics = spherical_harmonics([0, 1, 2], vector)
    """
    ls = list(ls)
    assert vector.is_irreps_array()
    irreps = vector.irreps
    assert len(irreps) == 1
    mul, ir = irreps[0]
    assert mul == 1
    assert ir.dim == 3
    assert min(ls) >= 0

    if normalize:
        vector = _normalize(vector)

    return cuex.equivariant_polynomial(
        descriptors.spherical_harmonics(ir, ls, vector.layout),
        [vector],
        name="spherical_harmonics",
    )


def normalize(array: cuex.RepArray, epsilon: float = 0.0) -> cuex.RepArray:
    assert array.is_irreps_array()

    match array.layout:
        case cue.ir_mul:
            axis_ir = -2
        case cue.mul_ir:
            axis_ir = -1

    def f(x: jax.Array) -> jax.Array:
        sn = jnp.sum(jnp.conj(x) * x, axis=axis_ir, keepdims=True)
        sn += epsilon
        if epsilon == 0.0:
            sn = jnp.where(sn == 0.0, 1.0, sn)
        return x / jnp.sqrt(sn)

    return cuex.from_segments(
        array.irreps,
        [f(x) for x in array.segments],
        array.shape,
        array.layout,
        array.dtype,
    )


_normalize = normalize


def norm(array: cuex.RepArray, *, squared: bool = False) -> cuex.RepArray:
    """Norm of `RepArray`."""
    assert array.is_irreps_array()

    match array.layout:
        case cue.ir_mul:
            axis_ir = -2
        case cue.mul_ir:
            axis_ir = -1

    def f(x: jax.Array) -> jax.Array:
        sn = jnp.sum(jnp.conj(x) * x, axis=axis_ir, keepdims=True)
        match squared:
            case True:
                return sn
            case False:
                sn_safe = jnp.where(sn == 0.0, 1.0, sn)
                rsn_safe = jnp.sqrt(sn_safe)
                rsn = jnp.where(sn == 0.0, 0.0, rsn_safe)
                return rsn

    return cuex.from_segments(
        cue.Irreps(array.irreps, [(mul, ir.trivial()) for mul, ir in array.irreps]),
        [f(x) for x in array.segments],
        array.shape,
        array.layout,
        array.dtype,
    )
