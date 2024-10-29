# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance.irreps_array.misc_ui import assert_same_group


def concatenate(arrays: list[cuex.IrrepsArray], axis: int = -1) -> cuex.IrrepsArray:
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
    irreps: cue.Irreps | cue.equivariant_tensor_product.Operand,
    leading_shape: tuple[int, ...] = (),
    layout: cue.IrrepsLayout | None = None,
    dtype: jnp.dtype | None = None,
) -> cuex.IrrepsArray:
    if isinstance(irreps, cue.equivariant_tensor_product.Operand):
        assert layout is None
        irreps, layout = irreps.irreps, irreps.layout

    irreps = cue.Irreps(irreps)
    leading_shape = tuple(leading_shape)

    return cuex.IrrepsArray(
        irreps,
        jax.random.normal(key, leading_shape + (irreps.dim,), dtype=dtype),
        layout,
    )
