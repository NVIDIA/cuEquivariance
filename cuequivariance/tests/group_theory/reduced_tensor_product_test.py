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

import itertools

import numpy as np
import pytest
from cuequivariance.group_theory.irreps_array.reduced_tensor_product import (
    reduce_basis_product,
    reduce_subgroup_permutation,
)

import cuequivariance as cue


def test_elasticity_tensor():
    # A general fourth-rank tensor in 3D has 3^4 = 81 independent components, but the elasticity tensor has at most 21 independent components.
    # source: https://en.wikipedia.org/wiki/Elasticity_tensor
    C = cue.reduced_tensor_product_basis("ijkl=klij=jikl", i=cue.Irreps("O3", "1o"))
    assert C.shape == (3, 3, 3, 3, 21)


def test_symmetric_basis():
    C = cue.reduced_symmetric_tensor_product_basis(cue.Irreps("SU2", "1/2 + 1"), 4)
    _irreps, C = C.irreps, C.array
    assert C.shape == (5, 5, 5, 5, 70)

    # The symmetry is respected
    for perm in itertools.permutations(range(4)):
        np.testing.assert_array_equal(C, np.transpose(C, perm + (4,)))

    # All components are independent
    C = np.reshape(C, (5**4, 70))
    assert np.linalg.matrix_rank(C) == 70


def test_reduced_tensor_product_frontends_and_validation():
    vector = cue.Irreps("O3", "1o")
    scalar = cue.Irreps("O3", "0e")

    formula_basis = cue.reduced_tensor_product_basis(
        "ij=ji",
        layout="ir_mul",
        keep_ir=["0e", "2e"],
        i=vector,
        _use_optimized_implementation=False,
    )
    assert formula_basis.layout == cue.ir_mul
    assert formula_basis.irreps == cue.Irreps("O3", "0e + 2e")
    assert formula_basis.shape == (3, 3, 6)
    flat = formula_basis.array.reshape(9, 6)
    np.testing.assert_allclose(flat.T @ flat, np.eye(6), atol=1e-8)
    np.testing.assert_allclose(
        formula_basis.array, formula_basis.array.swapaxes(0, 1), atol=1e-8
    )

    symmetric = cue.reduced_symmetric_tensor_product_basis(
        vector, 2, layout="ir_mul", keep_ir=["0e", "2e"]
    )
    np.testing.assert_allclose(symmetric.array, formula_basis.array, atol=1e-8)

    antisymmetric = cue.reduced_antisymmetric_tensor_product_basis(
        vector, 2, layout="ir_mul", keep_ir="1e"
    )
    assert antisymmetric.irreps == cue.Irreps("O3", "1e")
    assert antisymmetric.shape == (3, 3, 3)
    flat = antisymmetric.array.reshape(9, 3)
    np.testing.assert_allclose(flat.T @ flat, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(
        antisymmetric.array + antisymmetric.array.swapaxes(0, 1), 0, atol=1e-8
    )

    listed = cue.reduced_tensor_product_basis(
        [scalar, vector], layout="ir_mul", keep_ir=["1o"]
    )
    assert listed.layout == cue.ir_mul
    assert listed.irreps == vector
    assert listed.shape == (1, 3, 3)
    matrix = listed.array.reshape(3, 3)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-8)

    with pytest.raises(RuntimeError, match="0th power"):
        cue.reduced_tensor_product_basis([])
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cue.reduced_tensor_product_basis("ij", pair=vector)
    with pytest.raises(RuntimeError, match="irreps of i and j should be the same"):
        cue.reduced_tensor_product_basis("ij=ji", i=scalar, j=vector)
    with pytest.raises(RuntimeError, match="index j has no irreps associated"):
        cue.reduced_tensor_product_basis("ij", i=vector)
    with pytest.raises(RuntimeError, match="does not appear in the fomula"):
        cue.reduced_tensor_product_basis("ij", i=vector, j=vector, k=scalar)
    with pytest.raises(RuntimeError, match="is not a permutation"):
        cue.reduced_tensor_product_basis("ij=ii", i=vector, j=vector)


def test_reduced_tensor_product_greedy_and_layout_helpers():
    scalar = cue.Irreps("O3", "0e")
    vector = cue.Irreps("O3", "1o")

    basis = cue.reduced_tensor_product_basis(
        [scalar, vector, vector], _use_optimized_implementation=False
    )
    assert basis.shape == (1, 3, 3, 9)
    assert basis.irreps.dim == 9
    matrix = basis.array.reshape(9, 9)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(9), atol=1e-8)

    identity_permutation = frozenset({(1, (0, 1))})
    permutation_basis = reduce_subgroup_permutation(
        frozenset({0, 1}), identity_permutation, (2, 3)
    )
    assert (
        reduce_subgroup_permutation(
            frozenset({0, 1}), identity_permutation, (2, 3), return_dim=True
        )
        == 6
    )
    np.testing.assert_array_equal(permutation_basis, np.eye(6).reshape(6, 2, 3))

    repeated_vector = cue.Irreps("O3", "2x1o")
    basis_mul_ir = cue.NumpyIrrepsArray(
        repeated_vector, np.eye(repeated_vector.dim), cue.mul_ir
    )
    basis_ir_mul = basis_mul_ir.change_layout(cue.ir_mul)
    product_ir_mul = reduce_basis_product(basis_ir_mul, basis_ir_mul)
    product_mul_ir = reduce_basis_product(basis_mul_ir, basis_mul_ir)
    assert product_ir_mul.layout == cue.ir_mul
    assert product_ir_mul.irreps == product_mul_ir.irreps
    assert product_ir_mul.irreps.dim == repeated_vector.dim**2
    np.testing.assert_allclose(
        product_ir_mul.change_layout(cue.mul_ir).array,
        product_mul_ir.array,
        atol=1e-8,
    )
