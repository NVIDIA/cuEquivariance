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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import jax

import cuequivariance as cue
import cuequivariance_jax as cuex

T = TypeVar("T")


def remove_axis(dirreps: dict[int, T], axis: int) -> dict[int, T]:
    assert axis >= 0
    if axis in dirreps:
        raise ValueError(
            f"Cannot vmap over an Irreps axis. {axis} has Irreps {dirreps[axis]}."
        )
    return {
        a - 1 if a > axis else a: irreps for a, irreps in dirreps.items() if a != axis
    }


def add_axis(dirreps: dict[int, T], axis: int) -> dict[int, T]:
    return {a + 1 if a >= axis else a: irreps for a, irreps in dirreps.items()}


def vmap(
    fun: Callable[..., Any],
    in_axes: int | tuple[int, ...] = 0,
    out_axes: int = 0,
) -> Callable[..., Any]:
    """
    Like jax.vmap, but for RepArray.

    Args:
        fun: Callable[..., Any]: Function to vectorize. Can take `RepArray` as input and output.
        in_axes: int | tuple[int, ...]: Axes to vectorize over.
        out_axes: int: Axes to vectorize over.

    Returns:
        Callable[..., Any]: Vectorized function.
    """

    def inside_fun(*args, **kwargs):
        args, kwargs = jax.tree.map(
            lambda x: x.to_array() if isinstance(x, _wrapper) else x,
            (args, kwargs),
            is_leaf=lambda x: isinstance(x, _wrapper),
        )
        out = fun(*args, **kwargs)
        return jax.tree.map(
            lambda x: (
                _wrapper.from_array_add_axis(x, out_axes) if _is_array(x) else x
            ),
            out,
            is_leaf=_is_array,
        )

    def outside_fun(*args, **kwargs):
        if isinstance(in_axes, int):
            in_axes_ = (in_axes,) * len(args)
        else:
            in_axes_ = in_axes

        args = [
            jax.tree.map(
                lambda x: (
                    _wrapper.from_array_remove_axis(x, axis) if _is_array(x) else x
                ),
                arg,
                is_leaf=_is_array,
            )
            for axis, arg in zip(in_axes_, args)
        ]
        kwargs = jax.tree.map(
            lambda x: (_wrapper.from_array_remove_axis(x, 0) if _is_array(x) else x),
            kwargs,
            is_leaf=_is_array,
        )
        out = jax.vmap(inside_fun, in_axes, out_axes)(*args, **kwargs)
        return jax.tree.map(
            lambda x: x.to_array() if isinstance(x, _wrapper) else x,
            out,
            is_leaf=lambda x: isinstance(x, _wrapper),
        )

    return outside_fun


def _is_array(x):
    return isinstance(x, cuex.RepArray)


@dataclass(frozen=True)
class _wrapper:
    reps: dict[int, cue.Rep] = field()
    array: jax.Array = field()

    def to_array(self):
        return cuex.RepArray(self.reps, self.array)

    @classmethod
    def from_array_add_axis(cls, x: cuex.RepArray, axis: int) -> _wrapper:
        return _wrapper(add_axis(x.reps, axis), x.array)

    @classmethod
    def from_array_remove_axis(cls, x: cuex.RepArray, axis: int) -> _wrapper:
        return _wrapper(
            remove_axis(x.reps, axis if axis >= 0 else axis + x.ndim),
            x.array,
        )


jax.tree_util.register_pytree_node(
    _wrapper,
    lambda x: ((x.array,), (x.reps,)),
    lambda static, data: _wrapper(static[0], data[0]),
)
