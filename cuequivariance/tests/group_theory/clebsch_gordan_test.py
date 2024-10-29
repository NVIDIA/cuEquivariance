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
def test_clebsch_gordan(Group):
    it = Group.iterator()
    irreps = [next(it) for _ in range(4)]

    for r1, r2, r3 in itertools.combinations_with_replacement(irreps, 3):
        C = Group.clebsch_gordan(r1, r2, r3)

        a1 = np.einsum("zijk,giu->gzujk", C, r1.X)
        a2 = np.einsum("zijk,gju->gziuk", C, r2.X)
        a3 = np.einsum("zijk,guk->gziju", C, r3.X)  # Note the transpose

        np.testing.assert_allclose(a1 + a2, a3, atol=1e-10, rtol=0)
