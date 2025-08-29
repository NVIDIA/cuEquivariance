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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import cuequivariance as cue
import cuequivariance_jax as cuex  # noqa: F401


@dataclass(frozen=True, init=False, repr=False)
class RepArray:
    """
    A :class:`jax.Array <jax.Array>` decorated with a dict of :class:`cue.Rep <cuequivariance.Rep>` for the axes transforming under a group representation.

    Example:

    You can create a :class:`cuex.RepArray <cuequivariance_jax.RepArray>` by specifying the :class:`cue.Rep <cuequivariance.Rep>` for each axis:

    >>> cuex.RepArray({0: cue.SO3(1), 1: cue.SO3(1)}, jnp.eye(3))
    {0: 1, 1: 1}
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

    By default, arguments that are not :class:`cue.Rep <cuequivariance.Rep>` will be automatically converted into :class:`cue.IrrepsAndLayout <cuequivariance.IrrepsAndLayout>`:

    >>> with cue.assume(cue.SO3, cue.ir_mul):
    ...     x = cuex.RepArray({0: "1", 1: "2"}, jnp.ones((3, 5)))
    >>> x
    {0: 1, 1: 2}
    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    >>> x.rep(0).irreps, x.rep(0).layout
    (1, (irrep,mul))

    .. rubric:: IrrepsArray

    An ``IrrepsArray`` is just a special case of a ``RepArray`` where the last axis is a :class:`cue.IrrepsAndLayout <cuequivariance.IrrepsAndLayout>`:

    >>> x = cuex.RepArray(
    ...     cue.Irreps("SO3", "2x0"), jnp.zeros((3, 2)), cue.ir_mul
    ... )
    >>> x
    {1: 2x0}
    [[0. 0.]
     [0. 0.]
     [0. 0.]]

    >>> x.is_irreps_array()
    True

    You can use a default group and layout:

    >>> with cue.assume(cue.SO3, cue.ir_mul):
    ...     cuex.RepArray("2x0", jnp.array([1.0, 2.0]))
    {0: 2x0} [1. 2.]

    .. rubric:: Arithmetic

    Basic arithmetic operations are supported, as long as they are equivariant:

    >>> with cue.assume(cue.SO3, cue.ir_mul):
    ...     x = cuex.RepArray("2x0", jnp.array([1.0, 2.0]))
    ...     y = cuex.RepArray("2x0", jnp.array([3.0, 4.0]))
    ...     x + y
    {0: 2x0} [4. 6.]

    >>> 3.0 * x
    {0: 2x0} [3. 6.]
    """

    reps: dict[int, cue.Rep] = field()
    array: jax.Array = field()

    def __init__(
        self,
        reps: dict[int, cue.Rep]
        | cue.Rep
        | cue.Irreps
        | str
        | dict[int, cue.Irreps]
        | dict[int, str],
        array: jax.Array,
        layout: cue.IrrepsLayout | None = None,
    ):
        if not isinstance(reps, dict):
            reps = {-1: reps}

        # Remaining cases: dict[int, cue.Rep] | dict[int, cue.Irreps] | dict[int, str]

        reps = {
            axis: cue.Irreps(rep) if isinstance(rep, str) else rep
            for axis, rep in reps.items()
        }

        # Remaining cases: dict[int, cue.Rep] | dict[int, cue.Irreps]

        reps = {
            axis: cue.IrrepsAndLayout(rep, layout)
            if isinstance(rep, cue.Irreps)
            else rep
            for axis, rep in reps.items()
        }

        del layout
        assert isinstance(reps, dict)
        assert all(isinstance(k, int) for k in reps)
        assert all(isinstance(v, cue.Rep) for v in reps.values())

        ndim = getattr(array, "ndim", None)
        if ndim is not None:
            reps = {k + ndim if k < 0 else k: v for k, v in reps.items()}

        assert all(
            isinstance(k, int) and isinstance(v, cue.Rep) for k, v in reps.items()
        )
        assert all(k >= 0 for k in reps)

        if (
            hasattr(array, "shape")
            and isinstance(array.shape, tuple)
            and len(array.shape) > 0
        ):
            for axis, rep_ in reps.items():
                if len(array.shape) <= axis or array.shape[axis] != rep_.dim:
                    raise ValueError(
                        f"RepArray: Array shape {array.shape} incompatible with irreps {rep_}.\n"
                        "If you are trying to use jax.vmap, use cuex.vmap instead."
                    )

        if isinstance(array, RepArray):
            raise ValueError(
                "RepArray: Cannot create a RepArray from another RepArray."
            )

        object.__setattr__(self, "reps", reps)
        object.__setattr__(self, "array", array)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        return self.array.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        return self.array.ndim

    @property
    def dtype(self) -> jax.numpy.dtype:
        """Data type of the array."""
        return self.array.dtype

    def is_irreps_array(self) -> bool:
        """Check if the RepArray is an ``IrrepsArray``.

        An ``IrrepsArray`` is a `RepArray` where the last axis is an `IrrepsAndLayout`.
        """
        if len(self.reps) != 1:
            return False
        axis = next(iter(self.reps.keys()))
        if axis != self.ndim - 1:
            return False
        rep = self.rep(-1)
        return isinstance(rep, cue.IrrepsAndLayout)

    def rep(self, axis: int) -> cue.Rep:
        """Return the Rep for a given axis."""
        axis = axis if axis >= 0 else axis + self.ndim
        if axis not in self.reps:
            raise ValueError(f"No Rep for axis {axis}")
        return self.reps[axis]

    @property
    def irreps(self) -> cue.Irreps:
        """Return the `Irreps` of the ``IrrepsArray``.

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        assert self.is_irreps_array()
        return self.rep(-1).irreps

    @property
    def layout(self) -> cue.IrrepsLayout:
        """Return the layout of the ``IrrepsArray``.

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        assert self.is_irreps_array()
        return self.rep(-1).layout

    def __repr__(self):
        r = str(self.array)
        if "\n" in r:
            return f"{self.reps}\n{r}"
        return f"{self.reps} {r}"

    def __getitem__(self, key: Any) -> RepArray:
        # self[None]
        if key is None:
            return RepArray(
                {k + 1: rep for k, rep in self.reps.items()},
                self.array[None],
            )

        assert 0 not in self.reps

        # self[1:4]
        if isinstance(key, slice):
            return RepArray(self.reps, self.array[key])

        # self[jnp.array([0, 1, 2])]
        if isinstance(key, jax.Array):
            return RepArray(
                {k + key.ndim - 1: irreps for k, irreps in self.reps.items()},
                self.array[key],
            )

    @property
    def slice_by_mul(self) -> _MulIndexSliceHelper:
        r"""Return the slice with respect to the multiplicities.

        Examples:

            >>> x = cuex.RepArray(
            ...     cue.Irreps("SO3", "2x0 + 1"),
            ...     jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]),
            ...     cue.ir_mul
            ... )
            >>> x.slice_by_mul[1:4]
            {0: 0+1} [2. 0. 0. 0.]

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        assert self.is_irreps_array()
        return _MulIndexSliceHelper(self)

    def __neg__(self) -> RepArray:
        return RepArray(self.reps, -self.array)

    def __add__(self, other: RepArray | int | float) -> RepArray:
        if isinstance(other, (int, float)):
            assert other == 0
            return self

        if not isinstance(other, RepArray):
            raise ValueError(
                f"Try to add a RepArray with something that is not a RepArray: {other}"
            )

        if self.reps != other.reps:
            raise ValueError(
                f"Cannot add RepArray with different reps: {self.reps} != {other.reps}"
            )

        return RepArray(self.reps, self.array + other.array)

    def __radd__(self, other: RepArray) -> RepArray:
        return self + other

    def __sub__(self, other: RepArray | int | float) -> RepArray:
        return self + (-other)

    def __rsub__(self, other: RepArray | int | float) -> RepArray:
        return -self + other

    def __mul__(self, other: jax.Array) -> RepArray:
        other = jnp.asarray(other)
        other = jnp.expand_dims(other, tuple(range(self.ndim - other.ndim)))
        for axis, _ in self.reps.items():
            assert other.shape[axis] == 1
        return RepArray(self.reps, self.array * other)

    def __truediv__(self, other: jax.Array) -> RepArray:
        other = jnp.asarray(other)
        other = jnp.expand_dims(other, tuple(range(self.ndim - other.ndim)))
        for axis, _ in self.reps.items():
            assert other.shape[axis] == 1
        return RepArray(self.reps, self.array / other)

    def __rmul__(self, other: jax.Array) -> RepArray:
        return self * other

    def transform(self, v: jax.Array) -> RepArray:
        """Transform the array according to the representation.

        Args:
            v: Vector of angles.

        Examples:

            >>> x = cuex.RepArray(
            ...     {0: cue.SO3(1), 1: cue.SO3(1)}, jnp.ones((3, 3))
            ... )
            >>> x
            {0: 1, 1: 1}
            [[1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]]
            >>> x.transform(jnp.array([np.pi, 0.0, 0.0])).array.round(1)
            Array([[ 1., -1., -1.],
                   [-1.,  1.,  1.],
                   [-1.,  1.,  1.]]...)
        """

        def matrix(rep: cue.Rep) -> jax.Array:
            X = rep.X
            assert np.allclose(
                X, -X.conj().transpose((0, 2, 1))
            )  # TODO: support other types of X

            X = jnp.asarray(X, dtype=v.dtype)
            iX = 1j * jnp.einsum("a,aij->ij", v, X)
            m, V = jnp.linalg.eigh(iX)
            # np.testing.assert_allclose(V @ np.diag(m) @ V.T.conj(), iX, atol=1e-10)

            phase = jnp.exp(-1j * m)
            R = V @ jnp.diag(phase) @ V.T.conj()
            R = jnp.real(R)
            return R

        if self.is_irreps_array():

            def f(segment: jax.Array, ir: cue.Irrep) -> jax.Array:
                R = matrix(ir)
                match self.layout:
                    case cue.mul_ir:
                        return jnp.einsum("ij,...uj->...ui", R, segment)
                    case cue.ir_mul:
                        return jnp.einsum("ij,...ju->...iu", R, segment)

            return from_segments(
                self.irreps,
                [f(x, ir) for x, (_, ir) in zip(self.segments, self.irreps)],
                self.shape,
                self.layout,
                self.dtype,
            )

        a = self.array
        for axis, rep in self.reps.items():
            a = jnp.moveaxis(a, axis, 0)
            R = matrix(rep)
            a = jnp.einsum("ij,j...->i...", R, a)
            a = jnp.moveaxis(a, 0, axis)

        return RepArray(self.reps, a)

    @property
    def segments(self) -> list[jax.Array]:
        """Split the array into segments.

        Examples:

            >>> x = cuex.RepArray(
            ...     cue.Irreps("SO3", "2x0 + 1"), jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]),
            ...     cue.ir_mul
            ... )
            >>> x.segments
            [Array(...), Array(...)]

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        assert self.is_irreps_array()
        return [
            jnp.reshape(self.array[..., s], self.shape[:-1] + self.layout.shape(mulir))
            for s, mulir in zip(self.irreps.slices(), self.irreps)
        ]

    def filter(
        self,
        *,
        keep: str | Sequence[cue.Irrep] | Callable[[cue.MulIrrep], bool] | None = None,
        drop: str | Sequence[cue.Irrep] | Callable[[cue.MulIrrep], bool] | None = None,
        mask: Sequence[bool] | None = None,
    ) -> RepArray:
        """Filter the irreps.

        Args:
            keep: Irreps to keep.
            drop: Irreps to drop.
            mask: Boolean mask for segments to keep.
            axis: Axis to filter.

        Examples:

            >>> x = cuex.RepArray(
            ...     cue.Irreps("SO3", "2x0 + 1"),
            ...     jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]), cue.ir_mul
            ... )
            >>> x.filter(keep="0")
            {0: 2x0} [1. 2.]
            >>> x.filter(drop="0")
            {0: 1} [0. 0. 0.]
            >>> x.filter(mask=[True, False])
            {0: 2x0} [1. 2.]

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        assert self.is_irreps_array()

        if mask is None:
            mask = self.irreps.filter_mask(keep=keep, drop=drop)

        if all(mask):
            return self

        if not any(mask):
            shape = list(self.shape)
            shape[-1] = 0
            return RepArray(
                cue.Irreps(self.irreps.irrep_class, ""),
                jnp.zeros(shape, dtype=self.dtype),
                self.layout,
            )

        return RepArray(
            self.irreps.filter(mask=mask),
            jnp.concatenate(
                [self.array[..., s] for s, m in zip(self.irreps.slices(), mask) if m],
                axis=-1,
            ),
            self.layout,
        )

    def sort(self) -> RepArray:
        """Sort the irreps.

        Examples:

            >>> x = cuex.RepArray(
            ...     cue.Irreps("SO3", "1 + 2x0"),
            ...     jnp.array([1.0, 1.0, 1.0, 2.0, 3.0]), cue.ir_mul
            ... )
            >>> x.sort()
            {0: 2x0+1} [2. 3. 1. 1. 1.]

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        assert self.is_irreps_array()

        irreps = self.irreps
        r = irreps.sort()

        segments = self.segments
        return from_segments(
            r.irreps,
            [segments[i] for i in r.inv],
            self.shape,
            self.layout,
            self.dtype,
        )

    def simplify(self) -> RepArray:
        assert self.is_irreps_array()

        simplified_irreps = self.irreps.simplify()

        if self.layout == cue.mul_ir:
            return RepArray(simplified_irreps, self.array, self.layout)

        segments = []
        last_ir = None
        for x, (_mul, ir) in zip(self.segments, self.irreps):
            if last_ir is None or last_ir != ir:
                segments.append(x)
                last_ir = ir
            else:
                segments[-1] = jnp.concatenate([segments[-1], x], axis=-1)

        return from_segments(
            simplified_irreps,
            segments,
            self.shape,
            cue.ir_mul,
            self.dtype,
        )

    def regroup(self) -> RepArray:
        """Clean up the irreps.

        Examples:

            >>> x = cuex.RepArray(
            ...     cue.Irreps("SO3", "0 + 1 + 0"), jnp.array([0., 1., 2., 3., -1.]),
            ...     cue.ir_mul
            ... )
            >>> x.regroup()
            {0: 2x0+1} [ 0. -1.  1.  2.  3.]

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        return self.sort().simplify()

    def change_layout(self, layout: cue.IrrepsLayout) -> RepArray:
        """Change the layout of the ``IrrepsArray``.

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        assert self.is_irreps_array()
        if self.layout == layout:
            return self

        return from_segments(
            self.irreps,
            [jnp.moveaxis(x, -2, -1) for x in self.segments],
            self.shape,
            layout,
            self.dtype,
        )

    def move_axis_to_mul(self, axis: int) -> RepArray:
        """Move an axis to the multiplicities.

        Note:

            This method is only available for ``IrrepsArray``.
            See :func:`is_irreps_array <cuequivariance_jax.RepArray.is_irreps_array>`.
        """
        assert self.is_irreps_array()

        if axis < 0:
            axis += self.ndim
        assert axis < self.ndim - 1

        mul = self.shape[axis]

        match self.layout:
            case cue.ir_mul:
                array = jnp.moveaxis(self.array, axis, -1)
                array = jnp.reshape(array, array.shape[:-2] + (self.irreps.dim * mul,))
                return RepArray(mul * self.irreps, array, cue.ir_mul)
            case cue.mul_ir:

                def f(x):
                    x = jnp.moveaxis(x, axis, -3)
                    return jnp.reshape(
                        x, x.shape[:-3] + (mul * x.shape[-2], x.shape[-1])
                    )

                shape = list(self.shape)
                del shape[axis]
                shape[-1] = mul * shape[-1]

                return from_segments(
                    mul * self.irreps,
                    [f(x) for x in self.segments],
                    shape,
                    self.layout,
                    self.dtype,
                )


def encode_rep_array(x: RepArray) -> tuple:
    data = (x.array,)
    static = (x.reps,)
    return data, static


def decode_rep_array(static, data) -> RepArray:
    (reps,) = static
    (array,) = data
    return RepArray(reps, array)


jax.tree_util.register_pytree_node(RepArray, encode_rep_array, decode_rep_array)


def from_segments(
    irreps: cue.Irreps | str,
    segments: Sequence[jax.Array],
    shape: tuple[int, ...],
    layout: cue.IrrepsLayout | None = None,
    dtype: jnp.dtype | None = None,
) -> RepArray:
    """Construct a `RepArray` from segments.

    Args:
        irreps (Irreps): irreps.
        segments (list of jax.Array): segments.
        shape (tuple of int): shape of the final array.
        layout (IrrepsLayout): data layout.
        dtype: data type

    Returns:
        RepArray: the RepArray.

    Examples:

        >>> cuex.from_segments(
        ...     cue.Irreps("SO3", "2x0 + 1"),
        ...     [jnp.array([[1.0], [2.0]]), jnp.array([[0.0], [0.0], [0.0]])],
        ...     (-1,), cue.ir_mul)
        {0: 2x0+1} [1. 2. 0. 0. 0.]
    """
    irreps = cue.Irreps(irreps)
    shape = list(shape)
    shape[-1] = irreps.dim

    if not all(x.ndim == len(shape) + 1 for x in segments):
        raise ValueError(
            "from_segments: segments must have ndim equal to len(shape) + 1"
        )

    if len(segments) != len(irreps):
        raise ValueError(
            f"from_segments: the number of segments {len(segments)} must match the number of irreps {len(irreps)}"
        )

    if dtype is not None:
        segments = [segment.astype(dtype) for segment in segments]

    segments = [
        segment.reshape(segment.shape[:-2] + (mul * ir.dim,))
        for (mul, ir), segment in zip(irreps, segments)
    ]

    if len(segments) > 0:
        array = jnp.concatenate(segments, axis=-1)
    else:
        array = jnp.zeros(shape, dtype=dtype)

    return RepArray(irreps, array, layout)


class _MulIndexSliceHelper:
    irreps_array: RepArray

    def __init__(self, irreps_array: RepArray):
        assert irreps_array.is_irreps_array()
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> RepArray:
        if not isinstance(index, slice):
            raise IndexError(
                "RepArray.slice_by_mul only supports one slices (like RepArray.slice_by_mul[2:4])."
            )

        input_irreps = self.irreps_array.irreps
        start, stop, stride = index.indices(input_irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError("RepArray.slice_by_mul does not support strides.")

        output_irreps = []
        segments = []
        i = 0
        for (mul, ir), x in zip(input_irreps, self.irreps_array.segments):
            if start <= i and i + mul <= stop:
                output_irreps.append((mul, ir))
                segments.append(x)
            elif start < i + mul and i < stop:
                output_irreps.append((min(stop, i + mul) - max(start, i), ir))
                match self.irreps_array.layout:
                    case cue.mul_ir:
                        segments.append(
                            x[..., slice(max(start, i) - i, min(stop, i + mul) - i), :]
                        )
                    case cue.ir_mul:
                        segments.append(
                            x[..., slice(max(start, i) - i, min(stop, i + mul) - i)]
                        )

            i += mul

        return from_segments(
            cue.Irreps(input_irreps.irrep_class, output_irreps),
            segments,
            self.irreps_array.shape,
            self.irreps_array.layout,
            self.irreps_array.dtype,
        )
