# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import *

import numpy as np

from cuequivariance.misc.linalg import round_to_sqrt_rational
from cuequivariance.representation import Irrep, SU2


# This function is copied from https://github.com/lie-nn/lie-nn/blob/70adebce44e3197ee17f780585c6570d836fc2fe/lie_nn/_src/irreps/so3_real.py
def change_basis_real_to_complex(l: int) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / np.sqrt(2)
        q[l + m, l - abs(m)] = -1j / np.sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)

    # Added factor of 1j**l to make the Clebsch-Gordan coefficients real
    return q * (-1j) ** l


# This class is adapted from https://github.com/lie-nn/lie-nn/blob/70adebce44e3197ee17f780585c6570d836fc2fe/lie_nn/_src/irreps/so3_real.py
@dataclass(frozen=True)
class SO3(Irrep):
    r"""Subclass of :class:`Irrep`, real irreducible representations of the 3D rotation group :math:`SO(3)`.

    Each representation is labeled by a non-negative integer :math:`l`.
    """

    l: int

    @classmethod
    def regexp_pattern(cls) -> re.Pattern:
        return re.compile(r"(\d+)")

    @classmethod
    def from_string(cls, s: str) -> SO3:
        return cls(l=int(s))

    def __repr__(rep: SO3) -> str:
        return f"{rep.l}"

    def __mul__(rep1: SO3, rep2: SO3) -> Iterator[SO3]:
        rep2 = rep1._from(rep2)
        return [SO3(l=l) for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, rep1: SO3, rep2: SO3, rep3: SO3) -> np.ndarray:
        rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        C = SU2.clebsch_gordan(SU2(j=rep1.l), SU2(j=rep2.l), SU2(j=rep3.l))
        Q1 = change_basis_real_to_complex(rep1.l)
        Q2 = change_basis_real_to_complex(rep2.l)
        Q3 = change_basis_real_to_complex(rep3.l)
        C = np.einsum("ij,kl,mn,zikn->zjlm", Q1, Q2, np.conj(Q3.T), C)

        # ensure it's real
        assert np.all(np.abs(np.imag(C)) < 1e-5)
        C = np.real(C)

        return C

    @property
    def dim(rep: SO3) -> int:
        return 2 * rep.l + 1

    def __lt__(rep1: SO3, rep2: SO3) -> bool:
        rep2 = rep1._from(rep2)
        return rep1.l < rep2.l

    @classmethod
    def iterator(cls) -> Iterator[SO3]:
        for l in itertools.count(0):
            yield cls(l=l)

    def continuous_generators(rep: SO3) -> np.ndarray:
        X = SU2(j=rep.l).X
        Q = change_basis_real_to_complex(rep.l)
        X = np.conj(Q.T) @ X @ Q
        assert np.all(np.abs(np.imag(X)) < 1e-5)
        return np.real(X)

    def discrete_generators(rep: SO3) -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def algebra(rep=None) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SU2.algebra(rep)

    def rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotation matrix for the representation."""
        l = self.l
        m = np.arange(-l, l + 1)

        axis = axis / np.linalg.norm(axis)
        iX = 1j * np.sum(self.X * axis[:, None, None], axis=0)
        _val, V = np.linalg.eigh(iX)

        # np.testing.assert_allclose(_val, m, atol=1e-10)
        # np.testing.assert_allclose(V @ np.diag(m) @ V.T.conj(), iX, atol=1e-10)

        phase = np.exp(-1j * angle * m)

        R = V @ np.diag(phase) @ V.T.conj()
        # np.testing.assert_allclose(R.imag, 0, atol=1e-10)
        R = R.real

        if abs(angle) in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            R = round_to_sqrt_rational(R, 2**40 + 1)

        return R

    # def exp_map(
    #     self, continuous_params: np.ndarray, discrete_params: np.ndarray
    # ) -> np.ndarray:
    #     axis = continuous_params
    #     angle = np.linalg.norm(axis)
    #     if angle == 0:
    #         return np.eye(self.dim)

    #     return self.rotation(axis, angle)
