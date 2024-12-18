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
    Wrapper around a jax array with a dict of Rep for the non-trivial axes.

    .. rubric:: Creation

    You can create a RepArray by specifying the Reps for each axis:

    >>> cuex.RepArray({0: cue.SO3(1), 1: cue.SO3(1)}, jnp.eye(3))
    {0: 1, 1: 1}
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

    .. rubric:: Special case for Irreps

    If you are using Irreps, you can use the following syntax:

    >>> cuex.IrrepsArray(
    ...     cue.Irreps("SO3", "2x0"), jnp.array([1.0, 2.0]), cue.ir_mul
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
    """

    reps: dict[int, cue.Rep] = field()
    array: jax.Array = field()

    def __init__(
        self,
        reps: cue.Rep | dict[int, cue.Rep],
        array: jax.Array,
        layout: cue.IrrepsLayout | None = None,
    ):
        # Support for RepArray(irreps, array, layout)
        if isinstance(reps, str):
            reps = cue.Irreps(reps)
        if isinstance(reps, cue.Irreps):
            reps = cue.IrrepsAndLayout(reps, layout)
        else:
            assert layout is None
        # End of support for RepArray(irreps, array, layout)

        if isinstance(reps, cue.Rep):
            reps = {-1: reps}

        if not isinstance(reps, dict):
            raise ValueError(f"Invalid input for reps: {reps}, {type(reps)}")

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

        object.__setattr__(self, "reps", reps)
        object.__setattr__(self, "array", array)

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
        if len(self.reps) != 1:
            return False
        axis = next(iter(self.reps.keys()))
        return axis == self.ndim - 1

    def is_irreps_array(self) -> bool:
        if not self.is_simple():
            return False
        rep = self.rep()
        return isinstance(rep, cue.IrrepsAndLayout)

    def rep(self, axis: int = -1) -> cue.Rep:
        axis = axis if axis >= 0 else axis + self.ndim
        if axis not in self.reps:
            raise ValueError(f"No Rep for axis {axis}")
        return self.reps[axis]

    @property
    def irreps(self) -> cue.Irreps:
        """Return the Irreps of the RepArray if it is an IrrepsArray.

        Examples:

            >>> cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "2x0"), jnp.array([1.0, 2.0]), cue.ir_mul
            ... ).irreps
            2x0
        """
        assert self.is_irreps_array()
        return self.rep().irreps

    @property
    def layout(self) -> cue.IrrepsLayout:
        assert self.is_irreps_array()
        return self.rep().layout

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

        # self[jnp.array([0, 1, 2])]
        assert isinstance(key, jax.Array)
        assert 0 not in self.reps
        return RepArray(
            {k + key.ndim - 1: irreps for k, irreps in self.reps.items()},
            self.array[key],
        )

    @property
    def slice_by_mul(self) -> _MulIndexSliceHelper:
        r"""Return the slice with respect to the multiplicities.

        Examples:

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "2x0 + 1"),
            ...     jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]), cue.ir_mul
            ... )
            >>> x.slice_by_mul[1:4]
            {0: 0+1} [2. 0. 0. 0.]
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

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "2x0 + 1"), jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]),
            ...     cue.ir_mul
            ... )
            >>> x.segments
            [Array(...), Array(...)]

        Note:

            See also :func:`cuex.from_segments <cuequivariance_jax.from_segments>`.
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
        assert self.is_irreps_array()

        if mask is None:
            mask = self.irreps.filter_mask(keep=keep, drop=drop)

        if all(mask):
            return self

        if not any(mask):
            shape = list(self.shape)
            shape[-1] = 0
            return IrrepsArray(
                cue.Irreps(self.irreps.irrep_class, ""),
                jnp.zeros(shape, dtype=self.dtype),
                self.layout,
            )

        return IrrepsArray(
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

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "1 + 2x0"),
            ...     jnp.array([1.0, 1.0, 1.0, 2.0, 3.0]), cue.ir_mul
            ... )
            >>> x.sort()
            {0: 2x0+1} [2. 3. 1. 1. 1.]
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
            return IrrepsArray(simplified_irreps, self.array, self.layout)

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

            >>> x = cuex.IrrepsArray(
            ...     cue.Irreps("SO3", "0 + 1 + 0"), jnp.array([0., 1., 2., 3., -1.]),
            ...     cue.ir_mul
            ... )
            >>> x.regroup()
            {0: 2x0+1} [ 0. -1.  1.  2.  3.]
        """
        return self.sort().simplify()

    def change_layout(self, layout: cue.IrrepsLayout) -> RepArray:
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

IrrepsArray = RepArray


def from_segments(
    irreps: cue.Irreps | str,
    segments: Sequence[jax.Array],
    shape: tuple[int, ...],
    layout: cue.IrrepsLayout | None = None,
    dtype: jnp.dtype | None = None,
) -> RepArray:
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

    return IrrepsArray(irreps, array, layout)


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
