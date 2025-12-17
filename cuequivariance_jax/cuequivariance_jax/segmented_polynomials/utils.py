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
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Repeats:
    """
    A class to represent a sequence of repeated elements.

    Example:
        >>> a = Repeats(jnp.array([1, 0, 2]), 3)
        >>> jnp.repeat(
        ...     jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32),
        ...     a.repeats,
        ...     total_repeat_length=a.total_repeat_length,
        ... )
        Array([0.1, 0.3, 0.3], dtype=float32)
    """

    repeats: jax.Array = field()
    total_repeat_length: int = field(default=None)


jax.tree_util.register_pytree_node(
    Repeats,
    lambda x: ((x.repeats,), (x.total_repeat_length,)),
    lambda a, x: Repeats(x[0], a[0]),
)


def math_dtype_for_naive_method(
    io_dtype: jnp.dtype,
    math_dtype: str | None,
) -> tuple[jnp.dtype, jax.lax.Precision]:
    if math_dtype is None:
        return io_dtype, jax.lax.Precision.HIGHEST

    if hasattr(jnp, math_dtype):
        return getattr(jnp, math_dtype), jax.lax.Precision.HIGHEST

    if math_dtype == "tensor_float32":
        return jnp.float32, jax.lax.Precision.HIGH

    raise ValueError(
        f"method='naive' does not support math_dtype '{math_dtype}'. "
        "Supported options are any JAX dtype (e.g., 'float32', 'float64', 'float16', 'bfloat16') or 'tensor_float32'."
    )
