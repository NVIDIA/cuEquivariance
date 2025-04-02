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
from typing import Any

import jax
import jax.numpy as jnp


def reshape(
    x: jax.Array | jax.ShapeDtypeStruct, shape: tuple[int, ...]
) -> jax.Array | jax.ShapeDtypeStruct:
    if isinstance(x, jax.Array):
        return jnp.reshape(x, shape)
    else:
        return jax.ShapeDtypeStruct(shape, x.dtype)


def sanitize_multi_index(indices, ndim: int) -> tuple[Any, ...]:
    if not isinstance(indices, tuple):
        indices = (indices,)

    if Ellipsis in indices:
        assert indices.count(Ellipsis) == 1, "Only one ellipsis allowed"
        i = indices.index(Ellipsis)
        indices = (
            indices[:i] + (slice(None),) * (ndim - len(indices) + 1) + indices[i + 1 :]
        )

    indices = indices + (slice(None),) * (ndim - len(indices))
    return tuple(indices)


def batch_size(sizes: list[int]) -> int:
    batch_size = 1
    for size in sizes:
        if size != 1:
            assert batch_size in {1, size}
            batch_size = size
    return batch_size


def iota(shape, axis):
    i = jnp.arange(shape[axis])
    i = jnp.reshape(i, (1,) * (len(shape) - 1) + (-1,))
    i = jnp.moveaxis(i, -1, axis)
    return i


def indexing(
    bi: list[int], shape: tuple[int, ...], indices: list[jax.Array]
) -> tuple[slice, ...]:
    num_batch_axes = len(bi)
    shape = shape[:num_batch_axes]

    if all(i < 0 for i in bi):
        return tuple(slice(None) for _ in range(num_batch_axes))

    return tuple(
        iota(shape, axis) if i < 0 else indices[i] for axis, i in enumerate(bi)
    )
