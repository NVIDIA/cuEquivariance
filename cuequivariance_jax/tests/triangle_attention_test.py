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
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

import cuequivariance_jax as cuex


def create_test_data(
    batch_size=2, n_nodes=4, n_heads=2, seq_len_qo=8, seq_len_kv=6, d_model=32
):
    """Create test data for triangle attention."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    q = jax.random.normal(keys[0], (batch_size, n_nodes, n_heads, seq_len_qo, d_model))
    k = jax.random.normal(keys[1], (batch_size, n_nodes, n_heads, seq_len_kv, d_model))
    v = jax.random.normal(keys[2], (batch_size, n_nodes, n_heads, seq_len_kv, d_model))

    # Create a boolean mask (True means valid)
    mask = jax.random.bernoulli(keys[3], 0.8, (batch_size, n_nodes, 1, 1, seq_len_kv))

    bias = (
        jax.random.normal(keys[4], (batch_size, 1, n_heads, seq_len_qo, seq_len_kv))
        * 0.1
    )

    scale = d_model**-0.5

    return q, k, v, mask, bias, scale


def test_gradient_correctness_finite_differences():
    """Test gradient correctness using finite differences."""
    q, k, v, mask, bias, scale = create_test_data()

    def fn(q, k, v, bias):
        output, _, _ = cuex.experimental.triangle_attention(
            q, k, v, mask, bias, scale, precision=jax.lax.Precision.HIGH
        )
        return jnp.sum(output)

    check_grads(
        fn, (q, k, v, bias), order=1, modes=["rev"], eps=1e-1, atol=1e-2, rtol=1e-2
    )


def test_basic_functionality():
    """Basic test to ensure the function works."""
    q, k, v, mask, bias, scale = create_test_data()
    output, lse, amax = cuex.experimental.triangle_attention(q, k, v, mask, bias, scale)

    # Basic shape checks
    assert output.shape == q.shape
    assert lse.shape == q.shape[:-1] + (1,)
    assert amax.shape == q.shape[:-1] + (1,)
