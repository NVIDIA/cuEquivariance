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
import itertools
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Set

import networkx
import numpy as np

# With a mantissa of 53 bits, the maximum denominator is 2**53
# We use 2**40 to give some room for the numerical errors
DEFAULT_MAX_DENOMINATOR = 2**40


def normalize_integer_ratio(n, d):
    assert np.all(d > 0)
    g = np.gcd(n, d)
    g = np.where(d < 0, -g, g)
    return n // g, d // g


def _as_approx_integer_ratio(x):
    # only for 0 < x <= 1
    assert x.dtype == np.float64
    assert np.all(0 < x)
    assert np.all(x <= 1)

    big = 1 << 62
    n = np.floor(x * big).astype(np.int64)
    d = np.round(n / x).astype(np.int64)

    # if x is too small, then n = 0 and d = 0
    return n, d


def as_approx_integer_ratio(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # return (0, 0) if x is too small or too large
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x).astype(np.int64)
    x = np.abs(x)

    # case: x = 0
    n, d = np.zeros_like(x, np.int64), np.ones_like(x, np.int64)

    # case: 0 < x <= 1
    mask = (x > 0.0) & (x <= 1.0)
    n, d = np.where(
        mask,
        _as_approx_integer_ratio(np.where(mask, x, 1.0)),
        (n, d),
    )

    # case: x > 1
    mask = x > 1.0
    n, d = np.where(
        mask,
        _as_approx_integer_ratio(1 / np.where(mask, x, 1.0))[::-1],
        (n, d),
    )

    mask = d > 0
    n, d = np.where(
        mask, normalize_integer_ratio(sign * n, np.where(mask, d, 1)), (n, d)
    )

    return n, d


def limit_denominator(n, d, max_denominator: int):
    # (n, d) = must be normalized
    n = np.asarray(n, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    assert np.all(d > 0)
    assert np.all(np.gcd(n, d) == 1)

    n0, d0 = n, d
    p0, q0, p1, q1 = (
        np.zeros_like(n),
        np.ones_like(n),
        np.ones_like(n),
        np.zeros_like(n),
    )
    while True:
        a = n // d
        q2 = q0 + a * q1
        stop = (q2 > max_denominator) | (d0 <= max_denominator)
        if np.all(stop):
            break
        p0, q0, p1, q1 = np.where(stop, (p0, q0, p1, q1), (p1, q1, p0 + a * p1, q2))
        n, d = np.where(stop, (n, d), (d, n - a * d))

    with np.errstate(divide="ignore"):
        k = (max_denominator - q0) // q1
    n1, d1 = p0 + k * p1, q0 + k * q1
    n2, d2 = p1, q1
    with np.errstate(over="ignore"):
        mask = np.abs(d1 * (n2 * d0 - n0 * d2)) <= np.abs(d2 * (n1 * d0 - n0 * d1))
    return np.where(
        d0 <= max_denominator,
        (n0, d0),
        np.where(mask, (n2, d2), (n1, d1)),
    )


def support_complex(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(x, *args, **kwargs):
        if np.iscomplex(x).any():
            r = f(np.real(x), *args, **kwargs)
            i = f(np.imag(x), *args, **kwargs)
            return r + 1j * i
        return f(x, *args, **kwargs)

    return wrapper


@support_complex
def round_to_rational(
    x: np.ndarray, max_denominator: int = DEFAULT_MAX_DENOMINATOR
) -> np.ndarray:
    """Round a number to the closest number of the form ``n/d`` for ``d <= max_denominator``"""
    x = np.asarray(x, dtype=np.float64)
    n, d = as_approx_integer_ratio(x)

    mask = d > 0
    n, d = limit_denominator(n, np.where(mask, d, 1), max_denominator)
    return np.where(mask, n / d, 0.0)


@support_complex
def round_to_sqrt_rational(
    x: np.ndarray, max_denominator: int = DEFAULT_MAX_DENOMINATOR
) -> np.ndarray:
    """Round a number to the closest number of the form ``sqrt(n/d)`` for ``d <= max_denominator``"""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    n, d = as_approx_integer_ratio(x**2)

    mask = d > 0
    n, d = limit_denominator(n, np.where(mask, d, 1), max_denominator)
    return np.where(mask, sign * np.sqrt(n / d), 0.0)


def gram_schmidt(A: np.ndarray, *, epsilon=1e-5, round_fn=lambda x: x) -> np.ndarray:
    """
    Orthogonalize a matrix using the Gram-Schmidt process.
    """
    assert A.ndim == 2, "Gram-Schmidt process only works for matrices."
    assert A.dtype in [
        np.float64,
        np.complex128,
    ], "Gram-Schmidt process only works for float64 matrices."
    Q = []
    for i in range(A.shape[0]):
        v = A[i]
        for w in Q:
            v -= np.dot(np.conj(w), v) * w
        norm = np.linalg.norm(v)
        if norm > epsilon:
            v = round_fn(v / norm)
            Q += [v]
    return np.stack(Q) if len(Q) > 0 else np.empty((0, A.shape[1]))


def basis_intersection(
    basis1: np.ndarray, basis2: np.ndarray, *, epsilon=1e-5, round_fn=lambda x: x
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the intersection of two bases

    Args:
        basis1 (np.ndarray): A basis, shape ``(n1, d)``
        basis2 (np.ndarray): Another basis, shape ``(n2, d)``
        epsilon (float, optional): Tolerance for the norm of the vectors. Defaults to 1e-4.
        round_fn (function, optional): Function to round the vectors. Defaults to lambda x: x.

    Returns:
        (tuple): tuple containing:

            np.ndarray: A projection matrix that projects vectors of the first basis in the intersection of the two bases.
                Shape ``(dim_intersection, n1)``
            np.ndarray: A projection matrix that projects vectors of the second basis in the intersection of the two bases.
                Shape ``(dim_intersection, n2)``

    Examples:
        >>> basis1 = np.array([[1, 0, 0], [0, 0, 1.0]])
        >>> basis2 = np.array([[1, 1, 0], [0, 1, 0.0]])
        >>> P1, P2 = basis_intersection(basis1, basis2)
        >>> P1 @ basis1
        array([[1., 0., 0.]])
    """
    assert basis1.ndim == 2  # (n1, d)
    assert basis2.ndim == 2  # (n2, d)
    assert basis1.shape[1] == basis2.shape[1]

    p = np.concatenate(
        [
            np.concatenate([basis1 @ basis1.T, -basis1 @ basis2.T], axis=1),
            np.concatenate([-basis2 @ basis1.T, basis2 @ basis2.T], axis=1),
        ],
        axis=0,
    )
    p = round_fn(p)
    # p.shape = (n1 + n2, n1 + n2)

    w, v = np.linalg.eigh(p)
    v = v[:, w < epsilon]
    # ni = dim_intersection <= min(n1, n2)
    # v.shape = (n1 + n2, ni)

    x1 = v[: basis1.shape[0], :]  # (n1, ni)
    x1 = gram_schmidt(x1 @ x1.T, epsilon=epsilon, round_fn=round_fn)
    # x1.shape = (ni, n1)

    x2 = v[basis1.shape[0] :, :]  # (n2, ni)
    x2 = gram_schmidt(x2 @ x2.T, epsilon=epsilon, round_fn=round_fn)
    # x2.shape = (ni, n2)

    return x1, x2


TY_PERM = tuple[int, ...]


def perm_compose(p1: TY_PERM, p2: TY_PERM) -> TY_PERM:
    # p: i |-> p[i]

    # [p1.p2](i) = p1(p2(i)) = p1[p2[i]]
    return tuple(p1[p2[i]] for i in range(len(p1)))


def perm_inverse(p: TY_PERM) -> TY_PERM:
    return tuple(p.index(i) for i in range(len(p)))


def perm_to_cycles(p: TY_PERM) -> Set[tuple[int]]:
    n = len(p)

    cycles = set()

    for i in range(n):
        c = [i]
        while p[i] != c[0]:
            i = p[i]
            c += [i]
        if len(c) >= 2:
            i = c.index(min(c))
            c = c[i:] + c[:i]
            cycles.add(tuple(c))

    return cycles


def perm_sign(p: TY_PERM) -> int:
    s = 1
    for c in perm_to_cycles(p):
        if len(c) % 2 == 0:
            s = -s
    return s


def triu_coo(input: np.ndarray, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.nonzero(input)
    values = input[indices]
    indices = np.stack(indices, axis=1)
    indices = np.concatenate(
        [np.sort(indices[:, :ndim], axis=1), indices[:, ndim:]], axis=1
    )
    indices, inverse = np.unique(indices, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    values = np.bincount(inverse, values)
    return indices, values


def triu_array(input: np.ndarray, ndim: int) -> np.ndarray:
    indices, values = triu_coo(input, ndim)
    input = np.zeros_like(input)
    input[tuple(indices.T)] = values
    return input


@dataclass
class SparsifyResult:
    pass


@dataclass
class DisjointRows(SparsifyResult):
    pass


@dataclass
class ReplaceRow(SparsifyResult):
    a0: float
    a1: float
    row: np.ndarray
    which: int


@dataclass
class AlreadySparse(SparsifyResult):
    nc: int
    nx: int
    n0: int
    n1: int


def sparsify_rows(
    row0: np.ndarray, row1: np.ndarray, round_fn=lambda x: x
) -> SparsifyResult:
    """Try to find a sparse linear combination of row0 and row1.

    Args:
        row0: The first row.
        row1: The second row.

    Returns:
        The coefficients of the sparse linear combination.
        The linear combination.
        The index of the row that worth sparsifying.

    One could always reorder the rows so that:

        row0:
        [maximally comensurate][nonzero][nonzero][       ][     ]
        row1:
        [maximally comensurate][nonzero][       ][nonzero][     ]
             nc         nx       n0       n1
    """
    res = np.zeros_like(row0)

    i = np.where((row0 != 0) & (row1 != 0))[0]
    if len(i) == 0:
        return DisjointRows()

    r = row0[i] / row1[i]
    r = round_fn(r)
    r, c = np.unique(r, return_counts=True)
    r, nc = r[np.argmax(c)], np.max(c)

    n0 = np.sum((row0 != 0) & (row1 == 0))
    n1 = np.sum((row0 == 0) & (row1 != 0))

    if nc > min(n0, n1):
        res = row0 - r * row1
        j = 1 if n0 < n1 else 0
        v = np.linalg.norm(row0 if j == 0 else row1) / np.linalg.norm(res)
        res = res * v
        res = round_fn(res)
        return ReplaceRow(a0=v, a1=-r * v, row=res, which=j)

    return AlreadySparse(nc=nc, nx=len(i) - nc, n0=n0, n1=n1)


def sparsify_matrix(
    input: np.ndarray, max_iterations: int = 20, round_fn=lambda x: x
) -> tuple[np.ndarray, np.ndarray, networkx.Graph]:
    """Reduce the number of non-zero elements in a matrix.

    Args:
        input: The matrix to reduce.

    Returns:
        An invertible matrix q such that q @ input is sparser.
    """
    row_shape = input.shape[1:]
    input = np.reshape(input, (len(input), np.prod(row_shape)))
    assert input.ndim == 2

    non_zero_columns = np.where(np.any(input != 0, axis=0))[0]
    x = input[:, non_zero_columns]
    q = np.eye(x.shape[0])

    graph = networkx.Graph()
    graph.add_nodes_from(range(x.shape[0]))
    for i0, i1 in itertools.combinations(range(x.shape[0]), 2):
        graph.add_edge(i0, i1)

    iterations = 0
    hope = True
    while hope and iterations < max_iterations:
        hope = False

        next_graph = networkx.Graph()
        next_graph.add_nodes_from(range(x.shape[0]))

        for i0, i1 in graph.edges:
            result = sparsify_rows(x[i0], x[i1], round_fn)

            match result:
                case DisjointRows():
                    pass
                case ReplaceRow(a0=a0, a1=a1, row=row, which=which):
                    hope = True
                    which = i0 if which == 0 else i1
                    x[which] = row
                    q[which] = a0 * q[i0] + a1 * q[i1]

                    next_graph.add_edge(i0, i1)
                    for i in graph.neighbors(i0):
                        if i != i1:
                            next_graph.add_edge(i, i1)
                    for i in graph.neighbors(i1):
                        if i != i0:
                            next_graph.add_edge(i, i0)
                case AlreadySparse():
                    next_graph.add_edge(i0, i1)
                case _:
                    raise ValueError(f"Unknown result type: {result}")

        iterations += 1
        graph = next_graph

    output = np.zeros_like(input)
    output[:, non_zero_columns] = x
    output = np.reshape(output, (len(output),) + row_shape)
    return output, q, graph
