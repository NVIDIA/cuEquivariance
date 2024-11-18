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

import cuequivariance as cue


@pytest.mark.parametrize("Group", [cue.SU2, cue.SO3, cue.O3])
def test_clebsch_gordan(Group):
    it = Group.iterator()
    irreps = [next(it) for _ in range(4)]

    for r1, r2, r3 in itertools.combinations_with_replacement(irreps, 3):
        C = Group.clebsch_gordan(r1, r2, r3)

        a1 = np.einsum("zijk,giu->gzujk", C, r1.X)
        a2 = np.einsum("zijk,gju->gziuk", C, r2.X)
        a3 = np.einsum("zijk,guk->gziju", C, r3.X)  # Note the transpose

        np.testing.assert_allclose(a1 + a2, a3, atol=1e-10, rtol=0)
