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
import functools
import itertools
import logging
from math import prod
from typing import FrozenSet, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

import cuequivariance as cue
from cuequivariance.etc.linalg import (
    basis_intersection,
    perm_compose,
    perm_inverse,
    perm_sign,
    round_to_sqrt_rational,
    sparsify_matrix,
)
from cuequivariance.group_theory import Irrep, irreps_array
from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep

logger = logging.getLogger(__name__)


# The code below is adapted from https://github.com/e3nn/e3nn-jax/blob/245e17eb23deaccad9f2c9cfd40fe40515e3c074/e3nn_jax/_src/reduced_tensor_product.py

"""
History of the different versions of the code:
- Initially developed by Mario Geiger in `e3nn`
- Ported in julia by Song Kim https://github.com/songk42/ReducedTensorProduct.jl
- Ported in `e3nn-jax` by Mario Geiger
- Optimized the symmetric case by Ameya Daigavane and Mario Geiger
- imported and modified in cuequivariance by NVIDIA Corporation & Affiliates
"""


###### Frontend functions ######


def reduced_tensor_product_basis(
    formula_or_irreps_list: Union[str, List[irreps_array.Irreps]],
    *,
    layout: irreps_array.IrrepsLayout = irreps_array.mul_ir,
    epsilon: float = 1e-5,
    keep_ir: Optional[Union[irreps_array.Irreps, List[Irrep]]] = None,
    _use_optimized_implementation: bool = True,
    **irreps_dict,
) -> irreps_array.NumpyIrrepsArray:
    r"""Reduce a tensor product of multiple irreps subject to some permutation symmetry given by a formula.

    Args:
        formula_or_irreps_list (str or list of Irreps): a formula of the form ``ijk=jik=ikj`` or ``ijk=-jki``.
            The left hand side is the original formula and the right hand side are the signed permutations.
            If no index symmetry is present, a list of irreps can be given instead.

        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        keep_ir (list of Irrep): irrep to keep in the output. Default: keep all irrep
        irreps_dict (dict): the irreps of each index of the formula. For instance ``i="1x1o"``.

    Returns:
        IrrepsArray: The change of basis
            The shape is ``(d1, ..., dn, irreps_out.dim)``
            where ``di`` is the dimension of the index ``i`` and ``n`` is the number of indices in the formula.
    """
    if not isinstance(layout, irreps_array.IrrepsLayout):
        layout = irreps_array.IrrepsLayout[layout]

    if isinstance(formula_or_irreps_list, (tuple, list)):
        irreps_tuple = tuple(formula_or_irreps_list)
        assert all(isinstance(irreps, irreps_array.Irreps) for irreps in irreps_tuple)
        perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]] = frozenset(
            {(1, tuple(range(len(irreps_tuple))))}
        )
        if keep_ir is not None:
            keep_ir = frozenset(
                into_list_of_irrep(irreps_tuple[0].irrep_class, keep_ir)
            )
        return _entrypoint(
            irreps_tuple,
            perm_repr,
            keep_ir,
            layout,
            epsilon,
            _use_optimized_implementation,
        )

    formula = formula_or_irreps_list
    f0, perm_repr = germinate_perm_repr(formula)

    # irreps_dict = {i: irreps_array.Irreps(irs) for i, irs in irreps_dict.items()}
    assert all(isinstance(irs, irreps_array.Irreps) for irs in irreps_dict.values())

    for i in irreps_dict:
        if len(i) != 1:
            raise TypeError(f"got an unexpected keyword argument '{i}'")

    for _sign, p in perm_repr:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if (
                i in irreps_dict
                and j in irreps_dict
                and irreps_dict[i] != irreps_dict[j]
            ):
                raise RuntimeError(f"irreps of {i} and {j} should be the same")
            if i in irreps_dict:
                irreps_dict[j] = irreps_dict[i]
            if j in irreps_dict:
                irreps_dict[i] = irreps_dict[j]

    for i in f0:
        if i not in irreps_dict:
            raise RuntimeError(f"index {i} has no irreps associated to it")

    for i in irreps_dict:
        if i not in f0:
            raise RuntimeError(
                f"index {i} has an irreps but does not appear in the fomula"
            )

    irreps_tuple = tuple(irreps_dict[i] for i in f0)

    if keep_ir is not None:
        keep_ir = frozenset(into_list_of_irrep(irreps_tuple[0].irrep_class, keep_ir))
    return _entrypoint(
        irreps_tuple, perm_repr, keep_ir, layout, epsilon, _use_optimized_implementation
    )


def _symmetric_perm_repr(n: int):
    return frozenset((1, p) for p in itertools.permutations(range(n)))


def reduced_symmetric_tensor_product_basis(
    irreps: irreps_array.Irreps,
    degree: int,
    *,
    layout: irreps_array.IrrepsLayout = irreps_array.mul_ir,
    epsilon: float = 1e-5,
    keep_ir: Optional[Union[irreps_array.Irreps, List[Irrep]]] = None,
    _use_optimized_implementation: bool = True,
) -> irreps_array.NumpyIrrepsArray:
    r"""Reduce a symmetric tensor product, usually called for a single irrep.

    Args:
        irreps (Irreps): the irreps of each index.
        degree (int): the degree of the tensor product. i.e. the number of indices.
        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        keep_ir (list of Irrep): irrep to keep in the output. Default: keep all irrep

    Returns:
        IrrepsArray: The change of basis
            The shape is ``(d, ..., d, irreps_out.dim)``
            where ``d`` is the dimension of ``irreps``.
    """
    assert isinstance(irreps, irreps_array.Irreps)

    if not isinstance(layout, irreps_array.IrrepsLayout):
        layout = irreps_array.IrrepsLayout[layout]

    if keep_ir is not None:
        from .irrep_utils import into_list_of_irrep

        keep_ir = frozenset(into_list_of_irrep(irreps.irrep_class, keep_ir))

    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]] = _symmetric_perm_repr(degree)
    return _entrypoint(
        tuple([irreps] * degree),
        perm_repr,
        keep_ir,
        layout,
        epsilon,
        _use_optimized_implementation,
    )


def _antisymmetric_perm_repr(n: int):
    return frozenset((perm_sign(p), p) for p in itertools.permutations(range(n)))


def reduced_antisymmetric_tensor_product_basis(
    irreps: irreps_array.Irreps,
    degree: int,
    *,
    layout: irreps_array.IrrepsLayout = irreps_array.mul_ir,
    epsilon: float = 1e-5,
    keep_ir: Optional[Union[irreps_array.Irreps, List[Irrep]]] = None,
    _use_optimized_implementation: bool = True,
) -> irreps_array.NumpyIrrepsArray:
    r"""Reduce an antisymmetric tensor product.

    Args:
        irreps (Irreps): the irreps of each index.
        degree (int): the degree of the tensor product. i.e. the number of indices.
        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        keep_ir (list of Irrep): irrep to keep in the output. Default: keep all irrep

    Returns:
        IrrepsArray: The change of basis
            The shape is ``(d, ..., d, irreps_out.dim)``
            where ``d`` is the dimension of ``irreps``.
    """
    assert isinstance(irreps, irreps_array.Irreps)

    if not isinstance(layout, irreps_array.IrrepsLayout):
        layout = irreps_array.IrrepsLayout[layout]

    if keep_ir is not None:
        from .irrep_utils import into_list_of_irrep

        keep_ir = frozenset(into_list_of_irrep(irreps.irrep_class, keep_ir))

    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]] = _antisymmetric_perm_repr(degree)
    return _entrypoint(
        tuple([irreps] * degree),
        perm_repr,
        keep_ir,
        layout,
        epsilon,
        _use_optimized_implementation,
    )


###### Backend functions ######


def _entrypoint(
    irreps_tuple: Tuple[irreps_array.Irreps, ...],
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]],
    keep_ir: Optional[FrozenSet[Irrep]],
    layout: irreps_array.IrrepsLayout,
    epsilon: float,
    _use_optimized_implementation: bool,
) -> irreps_array.NumpyIrrepsArray:
    output = _main_cached_recursive(
        irreps_tuple,
        perm_repr,
        keep_ir,
        epsilon,
        _use_optimized_implementation,
    )

    # Sparsifying only at the end seems to give better sparsification
    output = _sparsify(output)

    if layout == cue.ir_mul:
        x = output.array

        for axis, irreps in enumerate(irreps_tuple + (output.irreps,)):
            x = np.moveaxis(x, axis, -1)
            x = (
                irreps_array.NumpyIrrepsArray(irreps, x, irreps_array.mul_ir)
                .change_layout(cue.ir_mul)
                .array
            )
            x = np.moveaxis(x, -1, axis)

        return irreps_array.NumpyIrrepsArray(output.irreps, x, cue.ir_mul)
    return output


@functools.lru_cache(maxsize=None)
def _main_cached_recursive(
    irreps_tuple: Tuple[irreps_array.Irreps, ...],
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]],
    keep_ir: Optional[FrozenSet[Irrep]],
    epsilon: float,
    _use_optimized_implementation: bool,
) -> irreps_array.NumpyIrrepsArray:
    degree = len(irreps_tuple)

    # Optimized case
    if (
        _use_optimized_implementation
        and perm_repr == _symmetric_perm_repr(degree)
        and degree > 1
    ):
        irreps = irreps_tuple[0]

        if irreps.num_irreps > 1:
            logger.debug(
                f"_reduced_tensor_product_basis {irreps_tuple}, {perm_repr}, keep_ir={keep_ir}, use _optimized_reduced_symmetric_tensor_product_basis"
            )
            return _rounding(
                _optimized_reduced_symmetric_tensor_product_basis(
                    irreps, degree, epsilon=epsilon, keep_ir=keep_ir
                )
            )

    logger.debug(
        f"_reduced_tensor_product_basis {irreps_tuple}, {perm_repr}, keep_ir={keep_ir}"
    )

    # General case
    dims = tuple(irreps.dim for irreps in irreps_tuple)

    bases = [
        (
            frozenset({i}),
            irreps_array.NumpyIrrepsArray(
                irreps,
                np.reshape(
                    np.eye(irreps.dim),
                    (1,) * i
                    + (irreps.dim,)
                    + (1,) * (len(irreps_tuple) - i - 1)
                    + (irreps.dim,),
                ),
                irreps_array.mul_ir,
            ),
        )
        for i, irreps in enumerate(irreps_tuple)
    ]

    while True:
        if len(bases) == 0:
            raise RuntimeError("Tensor Product raised to the 0th power is not defined.")

        if len(bases) == 1:
            f, b = bases[0]
            assert f == frozenset(range(len(irreps_tuple)))
            if keep_ir is not None:
                b = b.filter(keep=keep_ir)
            return b.regroup()

        if len(bases) == 2:
            (fa, a) = bases[0]
            (fb, b) = bases[1]
            f = frozenset(fa | fb)
            ab = reduce_basis_product(a, b, keep_ir)
            if len(subrepr_permutation(f, perm_repr)) == 1:
                return ab.regroup()
            p = reduce_subgroup_permutation(f, perm_repr, dims)
            ab = constrain_rotation_basis_by_permutation_basis(ab, p, epsilon=epsilon)
            return _rounding(ab.regroup())

        # greedy algorithm
        min_p = np.inf
        best = None

        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                (fa, _) = bases[i]
                (fb, _) = bases[j]
                f = frozenset(fa | fb)
                p_dim = reduce_subgroup_permutation(f, perm_repr, dims, return_dim=True)
                if p_dim < min_p:
                    min_p = p_dim
                    best = (i, j, f)

        i, j, f = best
        del bases[j]
        del bases[i]
        del i, j
        sub_irreps = tuple(irreps_tuple[i] for i in f)
        sub_perm_repr = subrepr_permutation(f, perm_repr)
        keep = keep_ir
        for i, irreps in enumerate(irreps_tuple):
            if i not in f:
                keep = cue.selection_rule_product(keep, {ir for _, ir in irreps})
        ab = _main_cached_recursive(
            sub_irreps,
            sub_perm_repr,
            keep,
            epsilon,
            _use_optimized_implementation,
        )
        ab = ab.reshape(
            tuple(dims[i] if i in f else 1 for i in range(len(dims))) + (-1,)
        )
        bases = [(f, ab)] + bases


###### Special case for symmetric tensor product ######


def generate_tuples_with_fixed_sum(length: int, sum: int):
    """Generates all non-negative integer tuples of a certain length with the specified sum.

    Examples:
        generate_tuples_with_fixed_sum(1, 3) = (3,)
        generate_tuples_with_fixed_sum(2, 3) = (3, 0) (2, 1) (1, 2) (0, 3)
    """
    if length == 1:
        yield (sum,)
        return

    if sum == 0:
        yield tuple(0 for _ in range(length))
        return

    for first in range(sum, -1, -1):
        for subtuple in generate_tuples_with_fixed_sum(length - 1, sum - first):
            yield (first,) + subtuple


def repeat_indices(indices: Sequence[int], powers: Sequence[int]) -> List[int]:
    """Given [i1, i2, ...] and [p1, p2, ...], returns [i1, i1, ... (p1 times), i2, i2, ... (p2 times), ...]"""
    repeated_indices = []
    for index, power in zip(indices, powers):
        repeated_indices.extend([index] * power)
    return repeated_indices


def generate_permutations(
    seq: Sequence[float],
) -> Iterator[Tuple[Sequence[float], Sequence[int]]]:
    """Generates permutations of a sequence along with the indices used to create the permutation."""
    indices = range(len(seq))
    for permuted_indices in itertools.permutations(indices):
        permuted_sequence = tuple(seq[index] for index in permuted_indices)
        yield permuted_sequence, permuted_indices


def cumsum_before(seq: Sequence[float]) -> np.ndarray:
    """Returns the cumulative sum before every index.

    For example, cumsum_before([1, 2, 3]) == [0, 1, 3].
    """
    return np.cumsum([0, *seq])[:-1]


def cumsum_after(seq: Sequence[float]) -> np.ndarray:
    """Returns the cumulative sum after every index.

    For example, cumsum_after([1, 2, 3]) == [5, 3, 0].
    """
    return cumsum_before(seq[::-1])[::-1]


def reshape_for_basis_product(
    terms: Sequence[irreps_array.NumpyIrrepsArray], non_zero_powers: Sequence[float]
):
    """Adds extra axes to each term to be compatible for reduce_basis_product()."""
    term_powers_cumsum_before = cumsum_before(non_zero_powers)
    term_powers_cumsum_after = cumsum_after(non_zero_powers)

    def reshape_term_for_basis_product(
        index: int, term: irreps_array.NumpyIrrepsArray
    ) -> irreps_array.NumpyIrrepsArray:
        new_shape = (
            (1,) * term_powers_cumsum_before[index]
            + term.shape[:-1]
            + (1,) * term_powers_cumsum_after[index]
            + term.shape[-1:]
        )
        return term.reshape(new_shape)

    return [
        reshape_term_for_basis_product(index, term) for index, term in enumerate(terms)
    ]


def _optimized_reduced_symmetric_tensor_product_basis(
    irreps: irreps_array.Irreps,
    degree: int,
    *,
    epsilon: float,
    keep_ir: Optional[List[Irrep]] = None,
):
    r"""Reduce a symmetric tensor product.

    Args:
        irreps (Irreps): the irreps of each index.
        degree (int): the degree of the tensor product. i.e. the number of indices.
        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        keep_ir (list of Irrep): irrep to keep in the output. Default: keep all irrep

    Returns:
        IrrepsArray: The change of basis
            The shape is ``(irreps.dim, ..., irreps.dim, irreps_out.dim)``
    """
    assert isinstance(irreps, irreps_array.Irreps)

    def compute_padding_for_term(irrep_indices: Sequence[int]) -> List[Tuple[int, int]]:
        """Computes the padding for the given term at each index.

        This is required because the output change-of-basis must have the
        shape (irreps.dim, ..., irreps.dim, irreps_out.dim),
        which means all input axes must have the same length.

        For example, when computing ir^3 = (ir_1 + ir_2)^3,
        we get a term corresponding to (ir_1)^3 with shape (ir_1.dim, ..., ir_1.dim, ...).
        We need to pad this term to have shape (ir.dim, ..., ir.dim, ...) instead.
        A similar logic applies for all of the terms.
        """
        inp_irreps_dims = [mul * ir.dim for mul, ir in irreps]
        inp_irreps_dims_cumsum_before = cumsum_before(inp_irreps_dims)
        inp_irreps_dims_cumsum_after = cumsum_after(inp_irreps_dims)

        def compute_padding_for_irrep_index(irrep_index: int):
            dims_before = inp_irreps_dims_cumsum_before[irrep_index]
            dims_after = inp_irreps_dims_cumsum_after[irrep_index]
            return (dims_before, dims_after)

        return [
            compute_padding_for_irrep_index(irrep_index)
            for irrep_index in irrep_indices
        ] + [(0, 0)]

    irreps = irreps_array.Irreps(
        irreps.irrep_class, [(1, ir) for mul, ir in irreps for _ in range(mul)]
    )

    # Precompute powers of irreps.
    irreps_powers = {}
    for i, mul_ir in enumerate(irreps):
        irreps_powers[i] = [
            irreps_array.NumpyIrrepsArray(
                irreps_array.Irreps(irreps.irrep_class, [irreps.irrep_class.trivial()]),
                np.asarray([1.0]),
                irreps_array.mul_ir,
            )
        ]
        for n in range(1, degree + 1):
            irreps_powers[i].append(
                reduced_symmetric_tensor_product_basis(
                    irreps_array.Irreps(irreps.irrep_class, [mul_ir]),
                    n,
                    epsilon=epsilon,
                    keep_ir=cue.selection_rule_product(
                        keep_ir,
                        cue.selection_rule_power(
                            irreps.irrep_class, {ir for _, ir in irreps}, degree - n
                        ),
                    ),
                )
            )

    # Take all products of irreps whose powers sum up to degree.
    # For example, if we are computing (ir1 + ir2)^3, we would consider terms of the form:
    # - ir_1 ir_1 ir_1
    # - ir_1 ir_1 ir_2, ir_1 ir_2 ir_1, ir_2 ir_1 ir_1
    # - ir_1 ir_2 ir_2, ir_2 ir_1 ir_2, ir_2 ir_2 ir_1
    # - ir_2 ir_2 ir_2
    # where the terms on the same line will be averaged over.
    # Each line above corresponds to a unique tuple:
    # - (3, 0)
    # - (2, 1)
    # - (1, 2)
    # - (0, 3)
    # indicating the powers of the individual irreps ir_1 and ir_2.
    # Note that possible many terms correspond to the same tuple,
    # since the tuple does not indicate the degree of multiplication.
    symmetric_product = []
    for term_powers in generate_tuples_with_fixed_sum(len(irreps), degree):
        term_powers = list(term_powers)

        non_zero_indices = [i for i, n in enumerate(term_powers) if n != 0]
        non_zero_powers = [n for n in term_powers if n != 0]
        non_zero_indices_repeated = tuple(
            repeat_indices(non_zero_indices, non_zero_powers)
        )

        # Add axes to all terms, so that they have the same number of input axes.
        non_zero_terms = [
            irreps_powers[i][n] for i, n in zip(non_zero_indices, non_zero_powers)
        ]
        non_zero_terms_reshaped = reshape_for_basis_product(
            non_zero_terms, non_zero_powers
        )

        # Compute basis product, two terms at a time.
        if len(non_zero_terms_reshaped) == 1:
            product_basis = non_zero_terms_reshaped[0].filter(keep=keep_ir)
        else:
            current_term = non_zero_terms_reshaped[0]
            for next_term in non_zero_terms_reshaped[1:-1]:
                current_term = reduce_basis_product(current_term, next_term)
            last_term = non_zero_terms_reshaped[-1]
            product_basis = reduce_basis_product(
                current_term, last_term, keep_ir=keep_ir
            )

        if product_basis.irreps.dim == 0:
            continue

        shape = (irreps.dim,) * degree + (product_basis.irreps.dim,)
        sum_of_permuted_bases = np.zeros_like(product_basis.array, shape=shape)
        seen_permutations = set()

        # Now, average over the different permutations.
        for permuted_indices_repeated, permuted_axes in generate_permutations(
            non_zero_indices_repeated
        ):
            # Keep track of which permutations we have seen.
            # Don't repeat permutations!
            if permuted_indices_repeated in seen_permutations:
                continue
            seen_permutations.add(permuted_indices_repeated)

            # Permute axes according to this term.
            permuted_product_basis_array = np.transpose(
                product_basis.array, permuted_axes + (len(permuted_axes),)
            )

            # Add padding.
            padding = compute_padding_for_term(permuted_indices_repeated)
            slices = tuple(
                slice(start, total - stop)
                for (start, stop), total in zip(padding, shape)
            )

            sum_of_permuted_bases[slices] += permuted_product_basis_array

        # Normalize the sum of bases.
        symmetrized_sum_of_permuted_bases = sum_of_permuted_bases / np.sqrt(
            len(seen_permutations)
        )
        product_basis = irreps_array.NumpyIrrepsArray(
            product_basis.irreps, symmetrized_sum_of_permuted_bases, irreps_array.mul_ir
        )
        symmetric_product.append(product_basis)

    # Filter out irreps, if needed.
    basis = cue.concatenate(symmetric_product)
    basis = basis.regroup()
    return basis


###### Utilities ######


@functools.lru_cache(maxsize=None)
def germinate_perm_repr(
    formula: str,
) -> Tuple[str, FrozenSet[Tuple[int, Tuple[int, ...]]]]:
    """Convert the formula (generators) into a group."""
    formulas = [
        (-1 if f.startswith("-") else 1, f.replace("-", "")) for f in formula.split("=")
    ]
    s0, f0 = formulas[0]
    assert s0 == 1

    for _s, f in formulas:
        if len(set(f)) != len(f) or set(f) != set(f0):
            raise RuntimeError(f"{f} is not a permutation of {f0}")
        if len(f0) != len(f):
            raise RuntimeError(f"{f0} and {f} don't have the same number of indices")

    # `perm_repr` is a list of (sign, permutation of indices)
    # each formula can be viewed as a permutation of the original formula
    perm_repr = {
        (s, tuple(f.index(i) for i in f0)) for s, f in formulas
    }  # set of generators (permutations)

    # they can be composed, for instance if you have ijk=jik=ikj
    # you also have ijk=jki
    # applying all possible compositions creates an entire group
    while True:
        n = len(perm_repr)
        perm_repr = perm_repr.union([(s, perm_inverse(p)) for s, p in perm_repr])
        perm_repr = perm_repr.union(
            [
                (s1 * s2, perm_compose(p1, p2))
                for s1, p1 in perm_repr
                for s2, p2 in perm_repr
            ]
        )
        if len(perm_repr) == n:
            break  # we break when the set is stable => it is now a group \o/

    return f0, frozenset(perm_repr)


_cache_reduce_basis_product = {}


def reduce_basis_product(
    basis1: irreps_array.NumpyIrrepsArray,
    basis2: irreps_array.NumpyIrrepsArray,
    keep_ir: Optional[List[Irrep]] = None,
) -> irreps_array.NumpyIrrepsArray:
    """Reduce the product of two basis."""
    basis1 = basis1.regroup()
    basis2 = basis2.regroup()

    assert basis1.layout == basis2.layout
    layout = basis1.layout

    if keep_ir is not None:
        assert all(isinstance(ir, Irrep) for ir in keep_ir)
        keep_ir = frozenset(keep_ir)

    assert basis1.array.dtype == np.float64
    assert basis2.array.dtype == np.float64

    cache_key = (
        keep_ir,
        basis1.irreps,
        basis2.irreps,
        basis1.shape,
        basis2.shape,
        hash(basis1.array.tobytes()),
        hash(basis2.array.tobytes()),
    )
    if cache_key in _cache_reduce_basis_product:
        return _cache_reduce_basis_product[cache_key]

    new_irreps: List[Tuple[int, Irrep]] = []
    new_list = []

    for (mul1, ir1), x1 in zip(basis1.irreps, basis1.segments):
        for (mul2, ir2), x2 in zip(basis2.irreps, basis2.segments):
            for ir in ir1 * ir2:
                if keep_ir is not None and ir not in keep_ir:
                    continue

                w = ir1.clebsch_gordan(ir1, ir2, ir)
                mulcg = len(w)
                if layout == irreps_array.mul_ir:
                    x = np.einsum("...ui,...vj,wijk->...uvwk", x1, x2, w)
                elif layout == cue.ir_mul:
                    x = np.einsum("...iu,...jv,wijk->...kuvw", x1, x2, w)
                x = np.reshape(
                    x, x.shape[:-4] + layout.shape((mul1 * mul2 * mulcg, ir))
                )
                new_list.append(x)
                new_irreps.append((mul1 * mul2 * mulcg, ir))

    new_irreps = irreps_array.Irreps(basis1.irreps.irrep_class, new_irreps)
    leading_shape = np.broadcast_shapes(basis1.shape[:-1], basis2.shape[:-1])
    out = cue.from_segments(new_irreps, new_list, layout, leading_shape).regroup()
    _cache_reduce_basis_product[cache_key] = out
    return out


def constrain_rotation_basis_by_permutation_basis(
    rotation_basis: irreps_array.NumpyIrrepsArray,
    permutation_basis: np.ndarray,
    *,
    epsilon=1e-5,
    round_fn=lambda x: x,
) -> irreps_array.NumpyIrrepsArray:
    """Constrain a rotation basis by a permutation basis.

    Args:
        rotation_basis (e3nn.IrrepsArray): A rotation basis
        permutation_basis (np.ndarray): A permutation basis

    Returns:
        e3nn.IrrepsArray: A rotation basis that is constrained by the permutation basis.
    """
    assert rotation_basis.shape[:-1] == permutation_basis.shape[1:]

    layout = rotation_basis.layout
    assert layout == irreps_array.mul_ir

    perm = np.reshape(
        permutation_basis,
        (permutation_basis.shape[0], prod(permutation_basis.shape[1:])),
    )  # (free, dim)

    new_irreps: List[Tuple[int, Irrep]] = []
    new_list: List[np.ndarray] = []

    for ir in sorted({ir for mul, ir in rotation_basis.irreps}):
        idx = [i for i, (mul, ir_) in enumerate(rotation_basis.irreps) if ir == ir_]
        rot_basis = np.concatenate([rotation_basis.segments[i] for i in idx], axis=-2)
        mul = rot_basis.shape[-2]
        R = rot_basis[..., 0]
        R = np.reshape(R, (-1, mul)).T  # (mul, dim)

        # optimization:
        perm_opt = perm[~np.all(perm[:, ~np.all(R == 0, axis=0)] == 0, axis=1)]
        # NOTE: this optimization work only because perm rows don't share non-zero elements

        P, _ = basis_intersection(R, perm_opt, epsilon=epsilon, round_fn=round_fn)

        new_irreps.append((len(P), ir))
        new_list.append(round_fn(np.einsum("vu,...ui->...vi", P, rot_basis)))

    new_irreps = irreps_array.Irreps(rotation_basis.irreps.irrep_class, new_irreps)
    return cue.from_segments(
        new_irreps, new_list, irreps_array.mul_ir, rotation_basis.shape[:-1]
    )


def subrepr_permutation(
    sub_f0: FrozenSet[int], perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]]
) -> FrozenSet[Tuple[int, Tuple[int, ...]]]:
    sor = sorted(sub_f0)
    return frozenset(
        {
            (s, tuple(sor.index(i) for i in p if i in sub_f0))
            for s, p in perm_repr
            if all(i in sub_f0 or i == j for j, i in enumerate(p))
        }
    )


def reduce_subgroup_permutation(
    sub_f0: FrozenSet[int],
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]],
    dims: Tuple[int, ...],
    return_dim: bool = False,
) -> np.ndarray:
    sub_perm_repr = subrepr_permutation(sub_f0, perm_repr)
    sub_dims = tuple(dims[i] for i in sub_f0)
    if len(sub_perm_repr) == 1:
        if return_dim:
            return prod(sub_dims)
        return np.eye(prod(sub_dims)).reshape((prod(sub_dims),) + sub_dims)
    base = reduce_permutation_base(sub_perm_repr, sub_dims)
    if return_dim:
        return len(base)
    permutation_basis = reduce_permutation_matrix(base, sub_dims)
    return np.reshape(
        permutation_basis,
        (-1,) + tuple(dims[i] if i in sub_f0 else 1 for i in range(len(dims))),
    )


@functools.lru_cache(maxsize=None)
def full_base_fn(dims: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    return list(itertools.product(*(range(d) for d in dims)))


@functools.lru_cache(maxsize=None)
def reduce_permutation_base(
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]], dims: Tuple[int, ...]
) -> FrozenSet[FrozenSet[FrozenSet[Tuple[int, Tuple[int, ...]]]]]:
    full_base = full_base_fn(dims)  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (3, 3, 3)
    # len(full_base) degrees of freedom in an unconstrained tensor

    # but there is constraints given by the group `formulas`
    # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
    base = set()
    for x in full_base:
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = {(s, tuple(x[i] for i in p)) for s, p in perm_repr}
        # s * T[x] are all equal for all (s, x) in xs
        # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
        if (-1, x) not in xs:
            # the sign is arbitrary, put both possibilities
            base.add(frozenset({frozenset(xs), frozenset({(-s, x) for s, x in xs})}))

    # len(base) is the number of degrees of freedom in the tensor.

    return frozenset(base)


@functools.lru_cache(maxsize=None)
def reduce_permutation_matrix(
    base: FrozenSet[FrozenSet[FrozenSet[Tuple[int, Tuple[int, ...]]]]],
    dims: Tuple[int, ...],
) -> np.ndarray:
    base = sorted(
        [sorted([sorted(xs) for xs in x]) for x in base]
    )  # requested for python 3.7 but not for 3.8 (probably a bug in 3.7)

    # First we compute the change of basis (projection) between full_base and base
    d_sym = len(base)
    Q = np.zeros((d_sym, prod(dims)), np.float64)

    for i, x in enumerate(base):
        x = max(x, key=lambda xs: sum(s for s, x in xs))
        for s, e in x:
            j = 0
            for k, d in zip(e, dims):
                j *= d
                j += k
            Q[i, j] = s / len(x) ** 0.5

    np.testing.assert_allclose(Q @ Q.T, np.eye(d_sym))

    return Q.reshape(d_sym, *dims)


def _rounding(x: irreps_array.NumpyIrrepsArray) -> irreps_array.NumpyIrrepsArray:
    return irreps_array.NumpyIrrepsArray(
        x.irreps, round_to_sqrt_rational(x.array), x.layout
    )


def _sparsify(input: irreps_array.NumpyIrrepsArray) -> irreps_array.NumpyIrrepsArray:
    assert input.layout == irreps_array.mul_ir
    input = input.regroup()

    segments = []
    for x in input.segments:
        x = np.moveaxis(x, -2, 0)  # (mul, ...)
        x, _, _ = sparsify_matrix(x, round_fn=round_to_sqrt_rational)
        x = np.moveaxis(x, 0, -2)
        segments.append(x)

    return cue.from_segments(
        input.irreps, segments, irreps_array.mul_ir, input.shape[:-1]
    )
