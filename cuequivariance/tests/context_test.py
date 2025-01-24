# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def test_rep_collection_context():
    explicit = cue.Irreps("O3", "0e + 1o + 2e")
    with cue.assume("O3"):
        implicit = cue.Irreps("0e + 1o + 2e")

    assert explicit == implicit

    with pytest.raises(ValueError):
        cue.Irreps("0e + 1o + 2e")


def test_layout_context():
    explicit = cue.NumpyIrrepsArray(cue.Irreps("O3", "0e + 1o"), np.ones(4), cue.ir_mul)

    with cue.assume(cue.ir_mul):
        implicit = cue.NumpyIrrepsArray(cue.Irreps("O3", "0e + 1o"), np.ones(4))

    assert explicit == implicit

    with pytest.raises(ValueError):
        cue.NumpyIrrepsArray(cue.Irreps("O3", "0e + 1o"), np.ones(4))


def test_rep_collection_and_layout_context():
    explicit = cue.NumpyIrrepsArray(cue.Irreps("O3", "0e + 1o"), np.ones(4), cue.ir_mul)

    with cue.assume("O3", cue.ir_mul):
        implicit = cue.NumpyIrrepsArray("0e + 1o", np.ones(4))

    assert explicit == implicit


def test_decorator():
    @cue.assume("O3", cue.ir_mul)
    def func():
        assert cue.get_irrep_scope() == cue.O3
        assert cue.get_layout_scope() == cue.ir_mul

    assert cue.get_irrep_scope(False) is None
    assert cue.get_layout_scope(False) is None
    func()
    assert cue.get_irrep_scope(False) is None
    assert cue.get_layout_scope(False) is None
