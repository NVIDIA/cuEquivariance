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

import numpy as np
import scipy.linalg


# This class is adapted from https://github.com/lie-nn/lie-nn/blob/70adebce44e3197ee17f780585c6570d836fc2fe/lie_nn/_src/rep.py
class Rep:
    r"""Abstract Class, Representation of a Lie group.

    ``Rep`` abstractly defines what a group representation is and how it can be used.
    """

    @property
    def lie_dim(self) -> int:
        """Dimension of the Lie algebra

        Returns:
            int: The dimension of the Lie algebra.
        """
        A = self.algebra()
        d = A.shape[0]
        return d

    @property
    def dim(self) -> int:
        """Dimension of the representation

        Returns:
            int: The dimension of the representation.
        """
        return self._dim()

    def _dim(self) -> int:
        X = self.continuous_generators()
        d = X.shape[1]
        return d

    @property
    def A(self) -> np.ndarray:
        """
        Algebra of the Lie group, ``(lie_dim, lie_dim, lie_dim)``

        See Also:
            :meth:`algebra`
        """
        return self.algebra()

    @property
    def X(self) -> np.ndarray:
        """
        Generators of the representation, ``(lie_dim, dim, dim)``

        See Also:
            :meth:`continuous_generators`
        """
        return self.continuous_generators()

    @property
    def H(self) -> np.ndarray:
        """
        Discrete generators of the representation, ``(len(H), dim, dim)``

        See Also:
            :meth:`discrete_generators`
        """
        return self.discrete_generators()

    def algebra(self) -> np.ndarray:
        """Algebra of the Lie group

        The algebra of the Lie group is defined by the following equation:

        .. math::

            [X_i, X_j] = A_{ijk} X_k

        where :math:`X_i` are the continuous generators and :math:`A_{ijk}` is the algebra.

        Returns:
            np.ndarray: An array of shape ``(lie_dim, lie_dim, lie_dim)``.
        """
        raise NotImplementedError  # pragma: no cover

    def continuous_generators(self) -> np.ndarray:
        r"""Generators of the representation

        The generators of the representation are defined by the following equation:

        .. math::

            \rho(\alpha) = \exp\left(\alpha_i X_i\right)

        Where :math:`\rho(\alpha)` is the representation of the group element
        corresponding to the parameter :math:`\alpha` and :math:`X_i` are the
        (continuous) generators of the representation, each of shape ``(dim, dim)``.

        Returns:
            np.ndarray: An array of shape ``(lie_dim, dim, dim)``.
        """
        raise NotImplementedError  # pragma: no cover

    def discrete_generators(self) -> np.ndarray:
        r"""Discrete generators of the representation

        .. math::

            \rho(n) = H^n

        Returns:
            np.ndarray: An array of shape ``(len(H), dim, dim)``.
        """
        raise NotImplementedError  # pragma: no cover

    def trivial(self) -> Rep:
        """
        Create a trivial representation from the same group as self
        """
        raise NotImplementedError  # pragma: no cover

    def exp_map(
        self, continuous_params: np.ndarray, discrete_params: np.ndarray
    ) -> np.ndarray:
        """Exponential map of the representation

        Args:
            continuous_params (np.ndarray): An array of shape ``(lie_dim,)``.
            discrete_params (np.ndarray): An array of shape ``(len(H),)``.

        Returns:
            np.ndarray: An matrix of shape ``(dim, dim)``.
        """
        output = scipy.linalg.expm(
            np.einsum("a,aij->ij", continuous_params, self.continuous_generators())
        )
        for k, h in reversed(list(zip(discrete_params, self.discrete_generators()))):
            output = np.linalg.matrix_power(h, k) @ output
        return output

    def is_scalar(self) -> bool:
        """Check if the representation is scalar (acting as the identity)"""
        return np.all(self.X == 0.0) and np.all(self.H == np.eye(self.dim))

    def is_trivial(self) -> bool:
        """Check if the representation is trivial (scalar of dimension 1)"""
        return self.dim == 1 and self.is_scalar()

    def __eq__(self, other: Rep) -> bool:
        return (
            self.dim == other.dim
            and np.allclose(self.A, other.A)
            and np.allclose(self.H, other.H)
            and np.allclose(self.X, other.X)
        )

    def __repr__(self) -> str:
        return f"Rep(dim={self.dim}, lie_dim={self.lie_dim}, len(H)={len(self.H)})"
