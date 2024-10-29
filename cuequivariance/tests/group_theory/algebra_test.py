# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
