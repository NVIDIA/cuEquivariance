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
    vector: cuex.IrrepsArray,
    normalize: bool = True,
    algorithm: str = "stacked",
) -> cuex.IrrepsArray:
    ls = list(ls)
    assert isinstance(vector, cuex.IrrepsArray)
    assert vector.is_simple()
    irreps = vector.irreps()
    assert len(irreps) == 1
    mul, ir = irreps[0]
    assert mul == 1
    assert ir.dim == 3
    assert max(ls) > 0
    assert min(ls) >= 0

    if normalize:
        vector = _normalize(vector)

    return cuex.equivariant_tensor_product(
        descriptors.spherical_harmonics(ir, ls, vector.layout),
        vector,
        algorithm=algorithm,
    )


def normalize(array: cuex.IrrepsArray) -> cuex.IrrepsArray:
    assert array.is_simple()

    match array.layout:
        case cue.ir_mul:
            axis_ir = -2
        case cue.mul_ir:
            axis_ir = -1

    def f(x: jax.Array) -> jax.Array:
        sn = jnp.sum(jnp.conj(x) * x, axis=axis_ir, keepdims=True)
        sn_safe = jnp.where(sn == 0.0, 1.0, sn)
        rsn_safe = jnp.sqrt(sn_safe)
        return x / rsn_safe

    return cuex.from_segments(
        array.irreps(),
        [f(x) for x in array.segments()],
        array.shape,
        array.layout,
        array.dtype,
    )


_normalize = normalize


def norm(array: cuex.IrrepsArray, *, squared: bool = False) -> cuex.IrrepsArray:
    """Norm of IrrepsArray."""
    assert array.is_simple()

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
        cue.Irreps(array.irreps(), [(mul, ir.trivial()) for mul, ir in array.irreps()]),
        [f(x) for x in array.segments()],
        array.shape,
        array.layout,
        array.dtype,
    )
