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
import jax.numpy as jnp
import pytest
from cuequivariance_jax.activation import (
    function_parity,
    normalize_function,
    normalspace,
    scalar_activation,
)

import cuequivariance as cue
import cuequivariance_jax as cuex


def test_normalspace():
    """Test the normalspace function."""
    n = 100
    points = normalspace(n)

    assert points.shape == (n,)
    assert jnp.all(jnp.diff(points) > 0)  # Points in ascending order
    assert jnp.isclose(jnp.mean(points), 0.0, atol=1e-6)


def test_normalize_function():
    """Test the normalize_function."""
    # Test with constant and non-constant functions
    norm_const = normalize_function(lambda x: jnp.ones_like(x))
    norm_linear = normalize_function(lambda x: x)

    test_points = normalspace(1001)

    # Check normalization
    assert jnp.isclose(jnp.mean(norm_const(test_points) ** 2), 1.0, atol=1e-2)
    assert jnp.isclose(jnp.mean(norm_linear(test_points) ** 2), 1.0, atol=5e-2)

    # Test zero function (should raise ValueError)
    with pytest.raises(ValueError):
        normalize_function(lambda x: jnp.zeros_like(x))


def test_function_parity():
    """Test the function_parity function."""
    # Test even, odd, and neither functions
    assert function_parity(jnp.cos) == 1  # Even
    assert function_parity(jnp.sin) == -1  # Odd
    assert function_parity(jnp.exp) == 0  # Neither


@cue.assume("SO3", cue.ir_mul)
def test_scalar_activation():
    """Test scalar_activation function."""
    # Create test data
    irreps = cue.Irreps("SO3", "2x0 + 0")
    x = cuex.randn(jax.random.key(42), irreps, (5,))

    # Test with a single activation
    y = scalar_activation(x, lambda x: 2 * x)
    assert y.irreps == x.irreps
    assert y.shape == x.shape

    # Test with multiple activations
    y = scalar_activation(x, [jnp.sin, jnp.cos])
    assert y.irreps == x.irreps

    # Test with a dict of activations
    y = scalar_activation(x, {cue.SO3(0): jnp.sin})
    assert y.shape == x.shape

    # Test with non-scalar irreps
    irreps_with_vectors = cue.Irreps("SO3", "0 + 1")
    x_with_vectors = cuex.randn(jax.random.key(43), irreps_with_vectors, (5,))

    # Should assert when trying to apply activation to non-scalar
    with pytest.raises(AssertionError):
        scalar_activation(x_with_vectors, jnp.sin)

    # Should work with None for non-scalar components
    y = scalar_activation(x_with_vectors, [jnp.sin, None])
    assert y.irreps == x_with_vectors.irreps
