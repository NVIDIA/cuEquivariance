# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cuequivariance as cue


def test_operation_transpose():
    ope = cue.Operation([0, 1, 2, 3])
    assert ope.transpose([True, False, False], [True]) == cue.Operation([3, 0, 1, 2])

    ope = cue.Operation([0, 1, 2, 4])
    assert ope.transpose([True, False, False, True], [True]) == cue.Operation(
        [3, 0, 1, 2]
    )

    ope = cue.Operation([4, 2, 3, 1])
    assert ope.transpose([True, False, False, True], [True]) == cue.Operation(
        [2, 1, 4, 0]
    )


def test_operation_cube():
    ope = cue.Operation([0, 0, 0, 1])  # x^3
    jvps = ope.jvp([True])
    assert jvps == [
        cue.Operation((1, 0, 0, 2)),
        cue.Operation((0, 1, 0, 2)),
        cue.Operation((0, 0, 1, 2)),
    ]

    [(mul, ope)] = cue.Operation.group_by_operational_symmetries(
        [
            (0, 1, 2, 3),
            (0, 2, 1, 3),
            (1, 0, 2, 3),
            (1, 2, 0, 3),
            (2, 0, 1, 3),
            (2, 1, 0, 3),
        ],
        jvps,
    )
    assert mul == 3
    assert ope == cue.Operation((0, 0, 1, 2))  # d/dx (x^3) = 3x^2 dx

    ope = ope.transpose([False, True], [True])
    assert ope == cue.Operation((0, 0, 2, 1))

    [(groupped_operands, [ope])] = cue.Operation.group_by_idential_buffers([ope])
    assert groupped_operands == frozenset(
        {frozenset({3}), frozenset({0, 1}), frozenset({2})}
    )
    assert ope == cue.Operation((0, 0, 2, 1))
