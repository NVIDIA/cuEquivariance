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
import pytest

import cuequivariance as cue


def test_init():
    irreps = cue.Irreps("SU2", "1/2 + 16x2")
    array = np.zeros((128, irreps.dim))

    cue.NumpyIrrepsArray(irreps, array, cue.mul_ir)
    cue.NumpyIrrepsArray(irreps, array, cue.ir_mul)
    cue.NumpyIrrepsArray(irreps, array, "mul_ir")
    cue.NumpyIrrepsArray(irreps, array, "ir_mul")

    with pytest.raises(ValueError):
        cue.NumpyIrrepsArray(irreps, array, "invalid")
    with pytest.raises(ValueError):
        cue.NumpyIrrepsArray(irreps, np.zeros((128, 128, 128)), cue.mul_ir)


def test_sort():
    x = cue.NumpyIrrepsArray(cue.Irreps("SU2", "1/2 + 0 + 1"), np.arange(6), cue.mul_ir)
    y = x.sort()

    assert y.irreps == cue.Irreps("SU2", "0 + 1/2 + 1")
    np.testing.assert_array_equal(x.segments[0], y.segments[1])
    np.testing.assert_array_equal(x.segments[1], y.segments[0])
    np.testing.assert_array_equal(x.segments[2], y.segments[2])
