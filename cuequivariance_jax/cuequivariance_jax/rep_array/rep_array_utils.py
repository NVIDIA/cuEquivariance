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
import numpy as np  # noqa: F401
from cuequivariance.group_theory.irreps_array.misc_ui import assert_same_group

import cuequivariance as cue
import cuequivariance_jax as cuex


def concatenate(arrays: list[cuex.RepArray]) -> cuex.RepArray:
    """Concatenate a list of :class:`cuex.RepArray <cuequivariance_jax.RepArray>`

    Args:
        arrays (list of RepArray): List of arrays to concatenate.
        axis (int, optional): Axis along which to concatenate. Defaults to -1.

    Example:

        >>> with cue.assume(cue.SO3, cue.ir_mul):
        ...     x = cuex.RepArray("3x0", jnp.array([1.0, 2.0, 3.0]))
        ...     y = cuex.RepArray("1x1", jnp.array([0.0, 0.0, 0.0]))
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
    return cuex.RepArray(
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
    """Converts input to a `RepArray`. Arrays are assumed to be scalars.

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
    return cuex.RepArray(irreps, input, layout)


def clebsch_gordan(rep1: cue.Irrep, rep2: cue.Irrep, rep3: cue.Irrep) -> cuex.RepArray:
    r"""
    Compute the Clebsch-Gordan coefficients.

    The Clebsch-Gordan coefficients are used to decompose the tensor product of two irreducible representations
    into a direct sum of irreducible representations. This method computes the Clebsch-Gordan coefficients
    for the given input representations and returns an array of shape ``(num_solutions, dim1, dim2, dim3)``,
    where num_solutions is the number of solutions, ``dim1`` is the dimension of ``rep1``, ``dim2`` is the
    dimension of ``rep2``, and ``dim3`` is the dimension of ``rep3``.

    The Clebsch-Gordan coefficients satisfy the following equation:

    .. math::

        C_{ljk} X^1_{li} + C_{ilk} X^2_{lj} = X^3_{kl} C_{ijl}

    Args:
        rep1 (Irrep): The first irreducible representation (input).
        rep2 (Irrep): The second irreducible representation (input).
        rep3 (Irrep): The third irreducible representation (output).

    Returns:
        RepArray: An array of shape ``(num_solutions, dim1, dim2, dim3)``.

    Examples:
        >>> rep1 = cue.SO3(1)
        >>> rep2 = cue.SO3(1)
        >>> rep3 = cue.SO3(2)
        >>> C1 = cuex.clebsch_gordan(rep1, rep2, rep3)
        >>> C1.shape
        (1, 3, 3, 5)

        According to the definition of the Clebsch-Gordan coefficients, the following transformation should be identity:
        >>> C2 = C1.transform(jnp.array([0.1, -0.3, 0.4]))
        >>> np.testing.assert_allclose(C1.array, C2.array, atol=1e-3)
    """
    return cuex.RepArray(
        {1: rep1, 2: rep2, 3: rep3}, cue.clebsch_gordan(rep1, rep2, rep3)
    )
