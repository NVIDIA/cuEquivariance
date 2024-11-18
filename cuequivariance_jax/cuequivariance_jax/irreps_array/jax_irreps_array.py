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
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import cuequivariance as cue
import cuequivariance_jax as cuex  # noqa: F401


def _check_args(
    dirreps: Any, layout: Any, ndim: int | None
) -> tuple[dict[int, cue.Irreps], cue.IrrepsLayout]:
    if isinstance(dirreps, (cue.Irreps, cue.Irrep, str)):
        dirreps = {-1: cue.Irreps(dirreps)}

    if not isinstance(dirreps, dict):
        raise ValueError(
            f"IrrepsArray: dirreps must be a dict of int -> Irreps, not {dirreps}"
        )

    dirreps = {k: cue.Irreps(v) for k, v in dirreps.items()}

    if not all(
        isinstance(k, int) and isinstance(v, cue.Irreps) for k, v in dirreps.items()
    ):
        raise ValueError(
            f"IrrepsArray: dirreps must be a dict of int -> Irreps, not {dirreps}"
        )

    layout = cue.IrrepsLayout.as_layout(layout)

    if ndim is not None:
        dirreps = {k + ndim if k < 0 else k: v for k, v in dirreps.items()}

    if any(k < 0 for k in dirreps.keys()):
        raise ValueError(
            f"IrrepsArray: dirreps keys must be non-negative, not {dirreps}"
        )

    return dirreps, layout


@dataclass(frozen=True, init=False, repr=False)
class IrrepsArray:
    """
    Wrapper around a jax array with a dict of Irreps for the non-trivial axes.

    .. rubric:: Creation

    >>> cuex.IrrepsArray(
    ...     {-1: cue.Irreps("SO3", "2x0")}, jnp.array([1.0, 2.0]), cue.ir_mul
    ... )
    {0: 2x0} [1. 2.]

    If you don't specify the axis it will default to the last axis:

    >>> cuex.IrrepsArray(
    ...     cue.Irreps("SO3", "2x0"), jnp.array([1.0, 2.0]), cue.ir_mul
    ... )
    {0: 2x0} [1. 2.]

    You can use a default group and layout:

    >>> with cue.assume(cue.SO3, cue.ir_mul):
    ...     cuex.IrrepsArray("2x0", jnp.array([1.0, 2.0]))
    {0: 2x0} [1. 2.]

    .. rubric:: Arithmetic

    Basic arithmetic operations are supported, as long as they are equivariant:

    >>> with cue.assume(cue.SO3, cue.ir_mul):
    ...     x = cuex.IrrepsArray("2x0", jnp.array([1.0, 2.0]))
    ...     y = cuex.IrrepsArray("2x0", jnp.array([3.0, 4.0]))
    ...     x + y
    {0: 2x0} [4. 6.]

    >>> 3.0 * x
    {0: 2x0} [3. 6.]

    .. rubric:: Attributes

    Attributes:
        dirreps: Irreps for the non-trivial axes, see also :func:`irreps() <cuequivariance_jax.IrrepsArray.irreps>` below.
        array: JAX array
        layout: Data layout
        shape: Shape of the array
        ndim: Number of dimensions of the array
        dtype: Data type of the array

    .. rubric:: Methods
    """

    layout: cue.IrrepsLayout = field()
    dirreps: dict[int, cue.Irreps] = field()
    array: jax.Array = field()

    def __init__(
        self,
        irreps: cue.Irreps | str | dict[int, cue.Irreps | str],
        array: jax.Array,
        layout: cue.IrrepsLayout | None = None,
    ):
        dirreps, layout = _check_args(irreps, layout, getattr(array, "ndim", None))

        if (
            hasattr(array, "shape")
            and isinstance(array.shape, tuple)
            and len(array.shape) > 0
        ):
            for axis, irreps_ in dirreps.items():
                if len(array.shape) <= axis or array.shape[axis] != irreps_.dim:
                    raise ValueError(
                        f"IrrepsArray: Array shape {array.shape} incompatible with irreps {irreps_}.\n"
                        "If you are trying to use jax.vmap, use cuex.vmap instead."
                    )

        object.__setattr__(self, "dirreps", dirreps)
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "layout", layout)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> jax.numpy.dtype:
        return self.array.dtype

    def is_simple(self) -> bool:
        """Return True if the IrrepsArray has only the last axis non-trivial.

        Examples:

            >>> cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "2x0"), jnp.array([1.0, 2.0]), cue.ir_mul
            ... ).is_simple()
            True
        """
        if len(self.dirreps) != 1:
            return False
        axis = next(iter(self.dirreps.keys()))
        return axis == self.ndim - 1

    def irreps(self, axis: int = -1) -> cue.Irreps:
        """Return the Irreps for a given axis.

        Examples:

            >>> cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "2x0"), jnp.array([1.0, 2.0]), cue.ir_mul
            ... ).irreps()
            2x0
        """
        axis = axis if axis >= 0 else axis + self.ndim
        if axis not in self.dirreps:
            raise ValueError(f"No Irreps for axis {axis}")
        return self.dirreps[axis]

    def __repr__(self):
        r = str(self.array)
        if "\n" in r:
            return f"{self.dirreps}\n{r}"
        return f"{self.dirreps} {r}"

    def __getitem__(self, key: Any) -> IrrepsArray:
        # self[None]
        if key is None:
            return IrrepsArray(
                {k + 1: irreps for k, irreps in self.dirreps.items()},
                self.array[None],
                self.layout,
            )

        # self[jnp.array([0, 1, 2])]
        assert isinstance(key, jax.Array)
        assert 0 not in self.dirreps
        return IrrepsArray(
            {k + key.ndim - 1: irreps for k, irreps in self.dirreps.items()},
            self.array[key],
            self.layout,
        )

    def slice_by_mul(self, axis: int = -1) -> _MulIndexSliceHelper:
        r"""Return the slice with respect to the multiplicities.

        Examples:

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "2x0 + 1"),
            ...     jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]), cue.ir_mul
            ... )
            >>> x.slice_by_mul()[1:4]
            {0: 0+1} [2. 0. 0. 0.]
        """
        return _MulIndexSliceHelper(self, axis)

    def __neg__(self) -> IrrepsArray:
        return IrrepsArray(self.dirreps, -self.array, self.layout)

    def __add__(self, other: IrrepsArray | int | float) -> IrrepsArray:
        if isinstance(other, (int, float)):
            assert other == 0
            return self

        if self.dirreps != other.dirreps:
            raise ValueError(
                f"Cannot add IrrepsArrays with different dirreps: {self.dirreps} != {other.dirreps}"
            )
        if self.layout != other.layout:
            raise ValueError(
                f"Cannot add IrrepsArrays with different layouts: {self.layout} != {other.layout}"
            )
        return IrrepsArray(self.dirreps, self.array + other.array, self.layout)

    def __radd__(self, other: IrrepsArray) -> IrrepsArray:
        return self + other

    def __sub__(self, other: IrrepsArray | int | float) -> IrrepsArray:
        return self + (-other)

    def __rsub__(self, other: IrrepsArray | int | float) -> IrrepsArray:
        return -self + other

    def __mul__(self, other: jax.Array) -> IrrepsArray:
        other = jnp.asarray(other)
        other = jnp.expand_dims(other, tuple(range(self.ndim - other.ndim)))
        for axis, irreps in self.dirreps.items():
            assert other.shape[axis] == 1
        return IrrepsArray(self.dirreps, self.array * other, self.layout)

    def __truediv__(self, other: jax.Array) -> IrrepsArray:
        other = jnp.asarray(other)
        other = jnp.expand_dims(other, tuple(range(self.ndim - other.ndim)))
        for axis, irreps in self.dirreps.items():
            assert other.shape[axis] == 1
        return IrrepsArray(self.dirreps, self.array / other, self.layout)

    def __rmul__(self, other: jax.Array) -> IrrepsArray:
        return self * other

    def filter(
        self,
        *,
        keep: str | Sequence[cue.Irrep] | Callable[[cue.MulIrrep], bool] | None = None,
        drop: str | Sequence[cue.Irrep] | Callable[[cue.MulIrrep], bool] | None = None,
        mask: Sequence[bool] | None = None,
        axis: int = -1,
    ) -> IrrepsArray:
        """Filter the irreps.

        Args:
            keep: Irreps to keep.
            drop: Irreps to drop.
            mask: Boolean mask for segments to keep.
            axis: Axis to filter.

        Examples:

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "2x0 + 1"),
            ...     jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]), cue.ir_mul
            ... )
            >>> x.filter(keep="0")
            {0: 2x0} [1. 2.]
            >>> x.filter(drop="0")
            {0: 1} [0. 0. 0.]
            >>> x.filter(mask=[True, False])
            {0: 2x0} [1. 2.]
        """
        if mask is None:
            mask = self.irreps(axis).filter_mask(keep=keep, drop=drop)

        if all(mask):
            return self

        if not any(mask):
            shape = list(self.shape)
            shape[axis] = 0
            return IrrepsArray(
                self.dirreps | {axis: cue.Irreps(self.irreps(axis).irrep_class, "")},
                jnp.zeros(shape, dtype=self.dtype),
                self.layout,
            )

        return IrrepsArray(
            self.dirreps | {axis: self.irreps(axis).filter(mask=mask)},
            jnp.concatenate(
                [
                    take_slice(self.array, s, axis)
                    for s, m in zip(self.irreps(axis).slices(), mask)
                    if m
                ],
                axis=axis,
            ),
            self.layout,
        )

    def sort(self, axis: int = -1) -> IrrepsArray:
        """Sort the irreps.

        Examples:

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "1 + 2x0"),
            ...     jnp.array([1.0, 1.0, 1.0, 2.0, 3.0]), cue.ir_mul
            ... )
            >>> x.sort()
            {0: 2x0+1} [2. 3. 1. 1. 1.]
        """
        if axis < 0:
            axis += self.ndim

        irreps = self.irreps(axis)
        r = irreps.sort()

        segments = self.segments(axis)
        return from_segments(
            self.dirreps | {axis: r.irreps},
            [segments[i] for i in r.inv],
            self.shape,
            self.layout,
            self.dtype,
            axis,
        )

    def simplify(self, axis: int = -1) -> IrrepsArray:
        if axis < 0:
            axis += self.ndim

        dirreps = self.dirreps | {axis: self.irreps(axis).simplify()}

        if self.layout == cue.mul_ir:
            return IrrepsArray(dirreps, self.array, self.layout)

        assert self.is_simple()
        segments = []
        last_ir = None
        for x, (mul, ir) in zip(self.segments(), self.irreps()):
            if last_ir is None or last_ir != ir:
                segments.append(x)
                last_ir = ir
            else:
                segments[-1] = jnp.concatenate([segments[-1], x], axis=-1)

        return from_segments(
            self.irreps().simplify(),
            segments,
            self.shape,
            cue.ir_mul,
            self.dtype,
        )

    def regroup(self, axis: int = -1) -> IrrepsArray:
        """Clean up the irreps.

        Examples:

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "0 + 1 + 0"), jnp.array([0., 1., 2., 3., -1.]),
            ...     cue.ir_mul
            ... )
            >>> x.regroup()
            {0: 2x0+1} [ 0. -1.  1.  2.  3.]
        """
        return self.sort(axis).simplify(axis)

    def segments(self, axis: int = -1) -> list[jax.Array]:
        """Split the array into segments.

        Examples:

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "2x0 + 1"), jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]),
            ...     cue.ir_mul
            ... )
            >>> x.segments()
            [Array(...), Array(...)]

        Note:

            See also :func:`cuex.from_segments <cuequivariance_jax.from_segments>`.
        """
        irreps = self.irreps(axis)
        return [
            take_slice(self.array, s, axis).reshape(
                expanded_shape(self.shape, mul_ir, axis, self.layout)
            )
            for s, mul_ir in zip(irreps.slices(), irreps)
        ]

    def change_layout(self, layout: cue.IrrepsLayout | None = None) -> IrrepsArray:
        assert self.is_simple()

        if layout is None:
            layout = cue.get_layout_scope()

        if self.layout == layout:
            return self

        return from_segments(
            self.dirreps,
            [jnp.moveaxis(x, -2, -1) for x in self.segments()],
            self.shape,
            layout,
            self.dtype,
        )

    def move_axis_to_mul(self, axis: int) -> IrrepsArray:
        assert self.is_simple()
        assert self.layout == cue.ir_mul
        if axis < 0:
            axis += self.ndim

        mul = self.shape[axis]
        array = jnp.moveaxis(self.array, axis, -1)
        array = jnp.reshape(array, array.shape[:-2] + (mul * self.irreps().dim,))
        return IrrepsArray(mul * self.irreps(), array, cue.ir_mul)

    def transform(self, v: jax.Array) -> IrrepsArray:
        assert self.is_simple()

        def f(segment: jax.Array, mul: int, ir: cue.Irrep) -> jax.Array:
            X = ir.X
            assert np.allclose(X, -X.conj().T)  # TODO: support other types of X

            X = jnp.asarray(X, dtype=v.dtype)
            iX = 1j * jnp.einsum("a,aij->ij", v, X)
            m, V = jnp.linalg.eigh(iX)
            # np.testing.assert_allclose(V @ np.diag(m) @ V.T.conj(), iX, atol=1e-10)

            phase = jnp.exp(-1j * m)
            R = V @ jnp.diag(phase) @ V.T.conj()
            R = R.real

            match self.layout:
                case cue.mul_ir:
                    return jnp.einsum("ij,...uj->...ui", R, segment)
                case cue.ir_mul:
                    return jnp.einsum("ij,...ju->...iu", R, segment)

        return from_segments(
            self.dirreps,
            [f(x, mul, ir) for x, (mul, ir) in zip(self.segments(), self.irreps())],
            self.shape,
            self.layout,
            self.dtype,
        )


def expanded_shape(
    shape: tuple[int, ...], mul_ir: cue.MulIrrep, axis: int, layout: cue.IrrepsLayout
) -> tuple[int, ...]:
    if axis < 0:
        axis += len(shape)
    return shape[:axis] + layout.shape(mul_ir) + shape[axis + 1 :]


def from_segments(
    dirreps: cue.Irreps | str | dict[int, cue.Irreps | str],
    segments: Sequence[jax.Array],
    shape: tuple[int, ...],
    layout: cue.IrrepsLayout | None = None,
    dtype: jnp.dtype | None = None,
    axis: int = -1,
) -> IrrepsArray:
    """Construct an :class:`cuex.IrrepsArrays <cuequivariance_jax.IrrepsArrays>` from a list of segments.

    Args:
        dirreps: final Irreps.
        segments: list of segments.
        shape: shape of the final array.
        layout: layout of the final array.
        dtype: data type
        axis: axis to concatenate the segments.

    Returns:
        IrrepsArray: IrrepsArray.

    Examples:

        >>> cuex.from_segments(
        ...     cue.Irreps("SO3", "2x0 + 1"),
        ...     [jnp.array([[1.0], [2.0]]), jnp.array([[0.0], [0.0], [0.0]])],
        ...     (-1,), cue.ir_mul)
        {0: 2x0+1} [1. 2. 0. 0. 0.]

    Note:

        See also :func:`cuex.IrrepsArray.segments <cuequivariance_jax.IrrepsArray.segments>`.
    """
    ndim = len(shape)
    dirreps, layout = _check_args(dirreps, layout, ndim)
    if axis < 0:
        axis += ndim

    shape = list(shape)
    for iaxis, irreps in dirreps.items():
        shape[iaxis] = irreps.dim

    if not all(x.ndim == len(shape) + 1 for x in segments):
        raise ValueError(
            "from_segments: segments must have ndim equal to len(shape) + 1"
        )

    if len(segments) != len(dirreps[axis]):
        raise ValueError(
            f"from_segments: the number of segments {len(segments)} must match the number of irreps {len(dirreps[axis])}"
        )

    if dtype is not None:
        segments = [segment.astype(dtype) for segment in segments]
    segments = [
        segment.reshape(
            segment.shape[:axis] + (mul * ir.dim,) + segment.shape[axis + 2 :]
        )
        for (mul, ir), segment in zip(dirreps[axis], segments)
    ]

    if len(segments) > 0:
        array = jnp.concatenate(segments, axis=axis)
    else:
        array = jnp.zeros(shape, dtype=dtype)

    return IrrepsArray(dirreps, array, layout)


def take_slice(x: jax.Array, s: slice, axis: int) -> jax.Array:
    slices = [slice(None)] * x.ndim
    slices[axis] = s
    return x[tuple(slices)]


def encode_irreps_array(x: IrrepsArray) -> tuple:
    data = (x.array,)
    static = (x.layout, x.dirreps)
    return data, static


def decode_irreps_array(static, data) -> IrrepsArray:
    layout, dirreps = static
    (array,) = data
    return IrrepsArray(dirreps, array, layout)


jax.tree_util.register_pytree_node(
    IrrepsArray, encode_irreps_array, decode_irreps_array
)


def remove_axis(dirreps: dict[int, cue.Irreps], axis: int):
    assert axis >= 0
    if axis in dirreps:
        raise ValueError(
            f"Cannot vmap over an Irreps axis. {axis} has Irreps {dirreps[axis]}."
        )
    return {
        a - 1 if a > axis else a: irreps for a, irreps in dirreps.items() if a != axis
    }


def add_axis(dirreps: dict[int, cue.Irreps], axis: int):
    return {a + 1 if a >= axis else a: irreps for a, irreps in dirreps.items()}


def vmap(
    fun: Callable[..., Any],
    in_axes: int | tuple[int, ...] = 0,
    out_axes: int = 0,
) -> Callable[..., Any]:
    """
    Like jax.vmap, but for IrrepsArray.

    Args:
        fun: Callable[..., Any]: Function to vectorize. Can take `IrrepsArray` as input and output.
        in_axes: int | tuple[int, ...]: Axes to vectorize over.
        out_axes: int: Axes to vectorize over.

    Returns:
        Callable[..., Any]: Vectorized function.
    """

    def inside_fun(*args, **kwargs):
        args, kwargs = jax.tree.map(
            lambda x: (
                IrrepsArray(x.dirreps, x.array, x.layout)
                if isinstance(x, _wrapper)
                else x
            ),
            (args, kwargs),
            is_leaf=lambda x: isinstance(x, _wrapper),
        )
        out = fun(*args, **kwargs)
        return jax.tree.map(
            lambda x: (
                _wrapper(x.layout, add_axis(x.dirreps, out_axes), x.array)
                if isinstance(x, IrrepsArray)
                else x
            ),
            out,
            is_leaf=lambda x: isinstance(x, IrrepsArray),
        )

    def outside_fun(*args, **kwargs):
        if isinstance(in_axes, int):
            in_axes_ = (in_axes,) * len(args)
        else:
            in_axes_ = in_axes

        args = [
            jax.tree.map(
                lambda x: (
                    _wrapper(
                        x.layout,
                        remove_axis(x.dirreps, axis if axis >= 0 else axis + x.ndim),
                        x.array,
                    )
                    if isinstance(x, IrrepsArray)
                    else x
                ),
                arg,
                is_leaf=lambda x: isinstance(x, IrrepsArray),
            )
            for axis, arg in zip(in_axes_, args)
        ]
        kwargs = jax.tree.map(
            lambda x: (
                _wrapper(x.layout, remove_axis(x.dirreps, 0), x.array)
                if isinstance(x, IrrepsArray)
                else x
            ),
            kwargs,
            is_leaf=lambda x: isinstance(x, IrrepsArray),
        )
        out = jax.vmap(inside_fun, in_axes, out_axes)(*args, **kwargs)
        return jax.tree.map(
            lambda x: (
                IrrepsArray(x.dirreps, x.array, x.layout)
                if isinstance(x, _wrapper)
                else x
            ),
            out,
            is_leaf=lambda x: isinstance(x, _wrapper),
        )

    return outside_fun


@dataclass(frozen=True)
class _wrapper:
    layout: cue.IrrepsLayout = field()
    dirreps: dict[int, cue.Irreps] = field()
    array: jax.Array = field()


jax.tree_util.register_pytree_node(
    _wrapper,
    lambda x: ((x.array,), (x.layout, x.dirreps)),
    lambda static, data: _wrapper(static[0], static[1], data[0]),
)


class _MulIndexSliceHelper:
    irreps_array: IrrepsArray
    axis: int

    def __init__(self, irreps_array: IrrepsArray, axis: int):
        self.irreps_array = irreps_array
        self.axis = axis if axis >= 0 else axis + irreps_array.ndim

    def __getitem__(self, index: slice) -> IrrepsArray:
        if not isinstance(index, slice):
            raise IndexError(
                "IrrepsArray.slice_by_mul only supports one slices (like IrrepsArray.slice_by_mul[2:4])."
            )

        input_irreps = self.irreps_array.irreps(self.axis)
        start, stop, stride = index.indices(input_irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError(
                "IrrepsArray.slice_by_mul does not support strides."
            )

        mul_axis = {
            cue.mul_ir: self.axis,
            cue.ir_mul: self.axis + 1,
        }[self.irreps_array.layout]

        output_irreps = []
        segments = []
        i = 0
        for (mul, ir), x in zip(input_irreps, self.irreps_array.segments(self.axis)):
            if start <= i and i + mul <= stop:
                output_irreps.append((mul, ir))
                segments.append(x)
            elif start < i + mul and i < stop:
                output_irreps.append((min(stop, i + mul) - max(start, i), ir))
                segments.append(
                    take_slice(
                        x, slice(max(start, i) - i, min(stop, i + mul) - i), mul_axis
                    )
                )

            i += mul

        return from_segments(
            self.irreps_array.dirreps
            | {self.axis: cue.Irreps(input_irreps.irrep_class, output_irreps)},
            segments,
            self.irreps_array.shape,
            self.irreps_array.layout,
            self.irreps_array.dtype,
            self.axis,
        )
