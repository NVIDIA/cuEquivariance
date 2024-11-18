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
def test_algebra(Group):
    # [X_i, X_j] = A_ijk X_k

    for r in itertools.islice(Group.iterator(), 6):
        xx = np.einsum("iuv,jvw->ijuw", r.X, r.X)
        term1 = xx - np.swapaxes(xx, 0, 1)
        term2 = np.einsum("ijk,kuv->ijuv", r.A, r.X)

        np.testing.assert_allclose(term1, term2, atol=1e-10, rtol=0)
