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

import jax
import numpy as np
import pytest

import cuequivariance as cue
import cuequivariance_jax as cuex

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("shape", [(2, 3), ()])
def test_spherical_harmonics(shape):
    x = cuex.RepArray(cue.Irreps(cue.O3, "1o"), np.random.randn(*shape, 3), cue.ir_mul)
    y = cuex.spherical_harmonics([0, 1, 2], x)
    assert y.shape == shape + (9,)
    assert y.irreps == cue.Irreps(cue.O3, "0e + 1o + 2e")
