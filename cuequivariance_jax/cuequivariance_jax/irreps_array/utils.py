# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from typing import *

import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance.equivariant_tensor_product import Operand
from cuequivariance.irreps_array.misc_ui import assert_same_group


def concatenate(arrays: list[cuex.IrrepsArray], axis: int = -1) -> cuex.IrrepsArray:
    """Concatenate a list of IrrepsArrays."""
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
    assert_same_group(*[a.irreps(axis) for a in arrays])

    if axis < 0:
        axis += arrays[0].ndim

    irreps = sum(
        (a.irreps(axis) for a in arrays), cue.Irreps(arrays[0].irreps(axis), [])
    )
    list_dirreps = [a.dirreps | {axis: irreps} for a in arrays]
    if not all(d == list_dirreps[0] for d in list_dirreps):
        raise ValueError("All arrays must have the same dirreps")  # pragma: no cover

    return cuex.IrrepsArray(
        list_dirreps[0],
        jnp.concatenate([a.array for a in arrays], axis=axis),
        arrays[0].layout,
    )


def randn(
    key: jax.Array,
    irreps: cue.Irreps | Operand,
    leading_shape: tuple[int, ...] = (),
    layout: cue.IrrepsLayout | None = None,
    dtype: jnp.dtype | None = None,
) -> cuex.IrrepsArray:
    r"""Generate a random IrrepsArray.

    Args:
        key (jax.Array): Random key.
        irreps (Irreps): Irreps of the array.
        leading_shape (tuple[int, ...], optional): Leading shape of the array. Defaults to ().
        layout (IrrepsLayout): Layout of the array.
        dtype (jnp.dtype): Data type of the array.

    Returns:
        IrrepsArray: Random IrrepsArray.

    Example:

    >>> key = jax.random.key(0)
    >>> irreps = cue.Irreps("O3", "2x1o")
    >>> cuex.randn(key, irreps, (), cue.ir_mul)
    {0: 2x1o} [...]
    """
    if isinstance(irreps, Operand):
        assert layout is None
        irreps, layout = irreps.irreps, irreps.layout

    irreps = cue.Irreps(irreps)
    leading_shape = tuple(leading_shape)

    return cuex.IrrepsArray(
        irreps,
        jax.random.normal(key, leading_shape + (irreps.dim,), dtype=dtype),
        layout,
    )


def as_irreps_array(
    input: Any,
    layout: cue.IrrepsLayout | None = None,
    axis: int | Sequence[int] = -1,
    like: cuex.IrrepsArray | None = None,
) -> cuex.IrrepsArray:
    """Converts input to an IrrepsArray. Arrays are assumed to be scalars."""
    # We need first to define axes and layout
    if like is not None:
        assert layout is None
        assert axis == -1
        layout = like.layout
        axes = {
            axis - like.ndim: irreps.irrep_class.trivial()
            for axis, irreps in like.dirreps.items()
        }
    else:
        if isinstance(input, cuex.IrrepsArray):
            axes = {
                axis: input.irreps(axis).irrep_class.trivial()
                for axis in (axis if isinstance(axis, Sequence) else [axis])
            }
        else:
            ir = cue.get_irrep_scope().trivial()
            axes = {
                axis: ir for axis in (axis if isinstance(axis, Sequence) else [axis])
            }
        if layout is None:
            if isinstance(input, cuex.IrrepsArray):
                layout = input.layout
            else:
                layout = cue.get_layout_scope()
    del like, axis

    if isinstance(input, cuex.IrrepsArray):
        if input.layout != layout:
            raise ValueError(
                f"as_irreps_array: layout mismatch {input.layout} != {layout}"
            )
        for axis, ir in axes.items():
            if input.irreps(axis).irrep_class is not type(ir):
                raise ValueError(
                    f"as_irreps_array: irrep mismatch {input.irreps(axis).irrep_class} != {type(ir)}"
                )
        return input

    input: jax.Array = jnp.asarray(input)
    # if max(axes.keys()) >= input.ndim:
    #     raise ValueError(
    #         f"as_irreps_array: input has {input.ndim} dimensions, but axes are {axes.keys()}"
    #     )

    dirreps = {
        axis: cue.Irreps(type(ir), [(input.shape[axis], ir)])
        for axis, ir in axes.items()
    }
    return cuex.IrrepsArray(dirreps, input, layout)
