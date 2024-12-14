from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import cuequivariance as cue
import cuequivariance_jax as cuex  # noqa: F401


@dataclass(frozen=True, init=False, repr=False)
class RepArray:
    """
    Wrapper around a jax array with a dict of Rep for the non-trivial axes.
    """

    reps: dict[int, cue.Rep] = field()
    array: jax.Array = field()

    def __init__(
        self,
        reps: cue.Rep | dict[int, cue.Rep],
        array: jax.Array,
    ):
        if isinstance(reps, cue.Rep):
            reps = {-1: reps}

        if not isinstance(reps, dict):
            raise ValueError("Invalid input")

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

    def rep(self, axis: int = -1) -> cue.Rep:
        axis = axis if axis >= 0 else axis + self.ndim
        if axis not in self.reps:
            raise ValueError(f"No Rep for axis {axis}")
        return self.reps[axis]

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
        assert self.is_simple()

        X = self.rep().X
        assert np.allclose(X, -X.conj().T)

        X = jnp.asarray(X, dtype=v.dtype)
        iX = 1j * jnp.einsum("a,aij->ij", v, X)
        m, V = jnp.linalg.eigh(iX)
        # np.testing.assert_allclose(V @ np.diag(m) @ V.T.conj(), iX, atol=1e-10)

        phase = jnp.exp(-1j * m)
        R = V @ jnp.diag(phase) @ V.T.conj()
        R = jnp.real(R)

        return RepArray(self.reps, jnp.einsum("ij,...j->...i", R, self.array))


def encode_rep_array(x: RepArray) -> tuple:
    data = (x.array,)
    static = (x.reps,)
    return data, static


def decode_rep_array(static, data) -> RepArray:
    (reps,) = static
    (array,) = data
    return RepArray(reps, array)


jax.tree_util.register_pytree_node(RepArray, encode_rep_array, decode_rep_array)
