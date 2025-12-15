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


def group_by_index(
    primary_idx: jax.Array,
    secondary_idx: jax.Array,
    max_primary_idx: int,
    axis: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """Group and reorder indices in CSR-like format along a specified axis.

    Groups ``secondary_idx`` values by their corresponding ``primary_idx`` values,
    enabling efficient contiguous access to all elements with the same primary index.

    Args:
        primary_idx: Indices to group by along ``axis``.
        secondary_idx: Indices to reorder. Must have matching size with ``primary_idx`` along ``axis``.
        max_primary_idx: Maximum value in ``primary_idx`` (exclusive).
        axis: Axis along which to perform grouping. Defaults to -1.

    Returns:
        tuple: ``(indptr, reordered_indices)`` where:
            - ``indptr``: Offsets of shape ``(..., max_primary_idx + 1, ...)`` where
              ``reordered_indices[..., indptr[k]:indptr[k+1], ...]`` contains all elements
              with ``primary_idx == k``.
            - ``reordered_indices``: Reordered ``secondary_idx`` with elements grouped by ``primary_idx``.

    Example:
        >>> primary_idx = jnp.array([1, 0, 2, 1, 0])
        >>> secondary_idx = jnp.array([10, 20, 30, 40, 50])
        >>> indptr, reordered = group_by_index(primary_idx, secondary_idx, max_primary_idx=3)
        >>> print(reordered[indptr[0]:indptr[1]])  # Elements where primary_idx == 0
        [20 50]
        >>> print(reordered[indptr[1]:indptr[2]])  # Elements where primary_idx == 1
        [10 40]
    """
    assert primary_idx.ndim == secondary_idx.ndim
    assert primary_idx.shape[axis] == secondary_idx.shape[axis]

    reordered = jnp.take_along_axis(
        secondary_idx, jnp.argsort(primary_idx, axis=axis), axis=axis
    )

    def compute_indptr(p):
        return jnp.append(
            0, jnp.cumsum(jnp.zeros((max_primary_idx,), jnp.int32).at[p].add(1))
        )

    p = jnp.moveaxis(primary_idx, axis, -1)
    indptr = jax.vmap(compute_indptr)(p.reshape(-1, p.shape[-1])).reshape(
        p.shape[:-1] + (max_primary_idx + 1,)
    )
    indptr = jnp.moveaxis(indptr, -1, axis)

    return indptr, reordered
