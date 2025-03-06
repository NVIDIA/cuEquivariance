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
