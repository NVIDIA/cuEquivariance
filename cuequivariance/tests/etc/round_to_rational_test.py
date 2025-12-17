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
import numpy as np
from cuequivariance.etc import linalg


def test_as_approx_integer_ratio():
    big = 1 << 62
    toobig = 1 << 63

    n, d = linalg.as_approx_integer_ratio(
        np.array([-0.5, 0.0, 0.1, 1.0, 10.0, big, toobig, 1 / toobig])
    )
    np.testing.assert_equal(
        n, [-1, 0, 3602879701896397, 1, 36028797018963968, big, 0, 0]
    )
    np.testing.assert_equal(d, [2, 1, 36028797018963968, 1, 3602879701896397, 1, 0, 0])


def test_round_to_rational():
    x0 = np.array([-12.0, -0.5, 0.0, 0.5, 1e10, 1e16, 0.75 + 1e-13])
    x1 = np.array([-12.0, -0.5, 0.0, 0.5, 1e10, 1e16, 0.75])
    np.testing.assert_equal(linalg.round_to_rational(x0), x1)


def test_round_to_sqrt_rational():
    x0 = np.array([0.0, -0.5, np.sqrt(3) + 1e-13])
    x1 = np.array([0.0, -0.5, np.sqrt(3)])
    np.testing.assert_equal(linalg.round_to_sqrt_rational(x0), x1)


def test_limit_denominator():
    np.testing.assert_equal(linalg.limit_denominator(1, 7, 7), [1, 7])
    np.testing.assert_equal(linalg.limit_denominator(4, 7 * 5, 7), [1, 7])
