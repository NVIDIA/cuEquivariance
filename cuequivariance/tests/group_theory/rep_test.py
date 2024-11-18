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
def test_rep(Group):
    # [X_i, X_j] = A_ijk X_k

    for r in itertools.islice(Group.iterator(), 6):
        A = r.A
        assert A.shape[0] == r.lie_dim
        X = r.X
        H = r.H
        assert r.dim == X.shape[1]
        assert r.dim == H.shape[1]


@pytest.mark.parametrize("Group", [cue.SU2, cue.SO3, cue.O3])
def test_is_scalar(Group):
    for r in itertools.islice(Group.iterator(), 6):
        assert r.is_scalar() == cue.Rep.is_scalar(r)


@pytest.mark.parametrize("Group", [cue.SU2, cue.SO3, cue.O3])
def test_dim(Group):
    for r in itertools.islice(Group.iterator(), 6):
        assert r.dim == cue.Rep.dim.fget(r)


def test_is_scalar_O3():
    assert cue.O3(0, 1).is_scalar()
    assert not cue.O3(0, -1).is_scalar()

    assert cue.O3(0, 1).is_trivial()
    assert not cue.O3(0, -1).is_trivial()

    r = cue.O3(0, 1)
    assert cue.Rep.is_scalar(r)
    assert cue.Rep.is_trivial(r)
    r = cue.O3(0, -1)
    assert not cue.Rep.is_scalar(r)
    assert not cue.Rep.is_trivial(r)


def test_exp_map():
    r = cue.SO3.from_string("1")
    np.testing.assert_allclose(
        r.exp_map(np.array([0.0, 0.0, 0.0]), np.array([])), np.eye(3)
    )
    np.testing.assert_allclose(
        r.exp_map(np.array([0.0, np.pi, 0.0]), np.array([])),
        np.diag([-1.0, 1.0, -1.0]),
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        r.exp_map(np.array([2 * np.pi, 0.0, 0.0]), np.array([])),
        np.eye(3),
        rtol=1e-15,
        atol=1e-15,
    )
    r = cue.O3.from_string("1o")
    np.testing.assert_allclose(
        r.exp_map(np.array([0.0, 0.0, 0.0]), np.array([1])), -np.eye(3)
    )
