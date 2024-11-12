# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np

import cuequivariance as cue


def test_elasticity_tensor():
    # A general fourth-rank tensor in 3D has 3^4 = 81 independent components, but the elasticity tensor has at most 21 independent components.
    # source: https://en.wikipedia.org/wiki/Elasticity_tensor
    C = cue.reduced_tensor_product_basis("ijkl=klij=jikl", i=cue.Irreps("O3", "1o"))
    assert C.shape == (3, 3, 3, 3, 21)


def test_symmetric_basis():
    C = cue.reduced_symmetric_tensor_product_basis(cue.Irreps("SU2", "1/2 + 1"), 4)
    irreps, C = C.irreps, C.array
    assert C.shape == (5, 5, 5, 5, 70)

    # The symmetry is respected
    for perm in itertools.permutations(range(4)):
        np.testing.assert_array_equal(C, np.transpose(C, perm + (4,)))

    # All components are independent
    C = np.reshape(C, (5**4, 70))
    assert np.linalg.matrix_rank(C) == 70
