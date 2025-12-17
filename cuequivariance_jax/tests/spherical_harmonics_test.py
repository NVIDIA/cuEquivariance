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

import jax
import numpy as np
import pytest

import cuequivariance as cue
import cuequivariance_jax as cuex

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("shape", [(2, 3), (), (10,), (10, 0, 3)])
def test_spherical_harmonics(shape):
    x = cuex.RepArray(cue.Irreps(cue.O3, "1o"), np.random.randn(*shape, 3), cue.ir_mul)
    y = cuex.spherical_harmonics([0, 1, 2], x)
    assert y.shape == shape + (9,)
    assert y.irreps == cue.Irreps(cue.O3, "0e + 1o + 2e")


@pytest.mark.parametrize("squared", [True, False])
def test_norm(squared):
    """Test the norm function with a batch of vectors."""
    # Create a RepArray with known values
    irreps = cue.Irreps(cue.O3, "1o")
    shape = (10,)

    # Create batch data with specific values in first two positions
    data = np.zeros(shape + (3,))
    data[0] = [3.0, 4.0, 0.0]  # norm = 5.0
    data[1] = [0.0, 0.0, 0.0]  # norm = 0.0
    data[2:] = np.random.randn(shape[0] - 2, 3)  # random values

    array = cuex.RepArray(irreps, data, cue.ir_mul)

    # Calculate norm
    norm_result = cuex.norm(array, squared=squared)

    # Verify basic properties
    assert norm_result.irreps.dim == 1
    for _, ir in norm_result.irreps:
        assert ir.is_trivial()
    assert norm_result.shape == shape + (1,)

    # Verify specific test values
    expected_first = 25.0 if squared else 5.0  # norm of [3,4,0]
    expected_second = 0.0  # norm of [0,0,0]

    np.testing.assert_allclose(norm_result.array[0, 0], expected_first, rtol=1e-6)
    np.testing.assert_allclose(norm_result.array[1, 0], expected_second, rtol=1e-6)
