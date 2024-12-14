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
from typing import Any

import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance.irreps_array.misc_ui import assert_same_group


def concatenate(arrays: list[cuex.RepArray]) -> cuex.RepArray:
    """Concatenate a list of :class:`cuex.RepArray <cuequivariance_jax.RepArray>`

    Args:
        arrays (list of RepArray): List of arrays to concatenate.
        axis (int, optional): Axis along which to concatenate. Defaults to -1.

    Example:

        >>> with cue.assume(cue.SO3, cue.ir_mul):
        ...     x = cuex.IrrepsArray("3x0", jnp.array([1.0, 2.0, 3.0]))
        ...     y = cuex.IrrepsArray("1x1", jnp.array([0.0, 0.0, 0.0]))
        >>> cuex.concatenate([x, y])
        {0: 3x0+1} [1. 2. 3. 0. 0. 0.]
    """
    if len(arrays) == 0:
        raise ValueError(
            "Must provide at least one array to concatenate"
        )  # pragma: no cover
    if not all(a.layout == arrays[0].layout for a in arrays):
        raise ValueError("All arrays must have the same layout")  # pragma: no cover
    if not all(a.ndim == arrays[0].ndim for a in arrays):
        raise ValueError(
            "All arrays must have the same number of dimensions"
        )  # pragma: no cover
    assert_same_group(*[a.irreps for a in arrays])

    irreps = sum(
        (a.irreps for a in arrays), cue.Irreps(arrays[0].irreps.irrep_class, [])
    )
    return cuex.IrrepsArray(
        irreps,
        jnp.concatenate([a.array for a in arrays], axis=-1),
        arrays[0].layout,
    )


def randn(
    key: jax.Array,
    rep: cue.Rep,
    leading_shape: tuple[int, ...] = (),
    dtype: jnp.dtype | None = None,
) -> cuex.RepArray:
    r"""Generate a random :class:`cuex.RepArray <cuequivariance_jax.RepArray>`.

    Args:
        key (jax.Array): Random key.
        rep (Rep): representation.
        leading_shape (tuple[int, ...], optional): Leading shape of the array. Defaults to ().
        dtype (jnp.dtype): Data type of the array.

    Returns:
        RepArray: Random RepArray.

    Example:

        >>> key = jax.random.key(0)
        >>> rep = cue.IrrepsAndLayout(cue.Irreps("O3", "2x1o"), cue.ir_mul)
        >>> cuex.randn(key, rep, ())
        {0: 2x1o} [...]
    """
    return cuex.RepArray(
        rep, jax.random.normal(key, leading_shape + (rep.dim,), dtype=dtype)
    )


def as_irreps_array(
    input: Any,
    layout: cue.IrrepsLayout | None = None,
    like: cuex.RepArray | None = None,
) -> cuex.RepArray:
    """Converts input to an IrrepsArray. Arrays are assumed to be scalars.

    Examples:

        >>> with cue.assume(cue.O3):
        ...     cuex.as_irreps_array([1.0], layout=cue.ir_mul)
        {0: 0e} [1.]
    """
    ir = None

    if like is not None:
        assert layout is None
        assert like.is_irreps_array()

        layout = like.layout
        ir = like.irreps.irrep_class.trivial()
    del like

    if layout is None:
        layout = cue.get_layout_scope()
    if ir is None:
        ir = cue.get_irrep_scope().trivial()

    if isinstance(input, cuex.RepArray):
        assert input.is_irreps_array()

        if input.layout != layout:
            raise ValueError(
                f"as_irreps_array: layout mismatch {input.layout} != {layout}"
            )

        return input

    input: jax.Array = jnp.asarray(input)
    irreps = cue.Irreps(type(ir), [(input.shape[-1], ir)])
    return cuex.IrrepsArray(irreps, input, layout)
