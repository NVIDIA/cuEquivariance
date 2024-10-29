# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# MIT License
# Copyright (c) 2023 lie-nn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import numpy as np
import scipy.linalg


class Rep:
    r"""Abstract Class, Representation of a Lie group.

    This class is the cornerstone of the Irreps class.
    It abstractly defines what a group representation is and how it can be used.
    """

    @property
    def lie_dim(self) -> int:
        """Dimension of the Lie algebra.

        Returns:
            int: The dimension of the Lie algebra.
        """
        A = self.algebra()
        d = A.shape[0]
        return d

    @property
    def dim(self) -> int:
        """Dimension of the representation.

        Returns:
            int: The dimension of the representation.
        """
        X = self.continuous_generators()
        d = X.shape[1]
        return d

    def algebra(self) -> np.ndarray:
        """
        Algebra of the Lie group.

        The algebra of the Lie group is defined by the following equation:

        .. math::

            [X_i, X_j] = A_{ijk} X_k

        Returns:
            np.ndarray: An array of shape (lie_dim, lie_dim, lie_dim).

        Raises:
            NotImplementedError: This method is not implemented and should be overridden by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def A(self) -> np.ndarray:
        """Algebra of the Lie group.

        Returns:
            np.ndarray: The algebra of the Lie group.
        """
        return self.algebra()

    def continuous_generators(self) -> np.ndarray:
        r"""
        Generators of the representation.

        The generators of the representation are defined by the following equation:

        .. math::

            \rho(\alpha) = \exp\left(\alpha_i X_i\right)

        Where :math:`\rho(\alpha)` is the representation of the group element
        corresponding to the parameter :math:`\alpha` and :math:`X_i` are the
        (continuous) generators of the representation.

        Returns:
            np.ndarray: An array of shape (lie_dim, dim, dim).

        Raises:
            NotImplementedError: This method is not implemented and should be overridden by subclasses.
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def X(self) -> np.ndarray:
        """Generators of the representation"""
        return self.continuous_generators()

    def discrete_generators(self) -> np.ndarray:
        r"""Discrete generators of the representation

        .. math::

            \rho(n) = H^n

        Returns:
            np.ndarray: An array of shape (len(H), dim, dim).
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def H(self) -> np.ndarray:
        """Discrete generators of the representation"""
        return self.discrete_generators()

    def trivial(self) -> Rep:
        """Create a trivial representation from the same group as self"""
        raise NotImplementedError  # pragma: no cover

    def exp_map(
        self, continuous_params: np.ndarray, discrete_params: np.ndarray
    ) -> np.ndarray:
        """
        Exponential map of the representation.

        Args:
            continuous_params (np.ndarray): An array of shape (lie_dim,).
            discrete_params (np.ndarray): An array of shape (len(H),).

        Returns:
            np.ndarray: An array of shape (dim, dim).
        """
        output = scipy.linalg.expm(
            np.einsum("a,aij->ij", continuous_params, self.continuous_generators())
        )
        for k, h in reversed(list(zip(discrete_params, self.discrete_generators()))):
            output = np.linalg.matrix_power(h, k) @ output
        return output

    def __repr__(self) -> str:
        return f"Rep(dim={self.dim}, lie_dim={self.lie_dim}, len(H)={len(self.H)})"

    def is_scalar(self) -> bool:
        """Check if the representation is scalar"""
        return np.all(self.X == 0.0) and np.all(self.H == np.eye(self.dim))

    def is_trivial(self) -> bool:
        """Check if the representation is trivial"""
        return self.dim == 1 and self.is_scalar()
