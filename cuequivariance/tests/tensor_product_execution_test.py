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
from cuequivariance.tensor_product_execution import InBuffer, OutBuffer


def test_group_by_symmetries():
    # x^3
    exe = cue.TensorProductExecution(
        [(InBuffer(0), InBuffer(0), InBuffer(0), OutBuffer(0))]
    )
    mul, exe = next(
        exe.jvp([True]).group_by_symmetries(
            [
                (0, 1, 2, 3),
                (0, 2, 1, 3),
                (1, 0, 2, 3),
                (1, 2, 0, 3),
                (2, 0, 1, 3),
                (2, 1, 0, 3),
            ]
        )
    )
    # d/dx (x^3) = 3x^2
    assert mul == 3
    expected = cue.TensorProductExecution(
        [(InBuffer(1), InBuffer(0), InBuffer(0), OutBuffer(0))]
    )
    assert exe == expected
