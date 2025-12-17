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
import pytest
from jax.test_util import check_grads

import cuequivariance_jax as cuex

jax.config.update("jax_enable_x64", True)


def create_test_data(
    platform: str,
    batch_size=2,
    n_nodes=4,
    n_heads=2,
    seq_len_qo=8,
    seq_len_kv=6,
    d_model=32,
    dtype=jnp.float32,
):
    """Create test data for triangle attention."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    q = jax.random.normal(
        keys[0], (batch_size, n_nodes, n_heads, seq_len_qo, d_model), dtype
    )
    k = jax.random.normal(
        keys[1], (batch_size, n_nodes, n_heads, seq_len_kv, d_model), dtype
    )
    v = jax.random.normal(
        keys[2], (batch_size, n_nodes, n_heads, seq_len_kv, d_model), dtype
    )
    bias = jax.random.normal(
        keys[3], (batch_size, 1, n_heads, seq_len_qo, seq_len_kv), dtype
    )
    mask = jax.random.bernoulli(keys[4], 0.8, (batch_size, n_nodes, 1, 1, seq_len_kv))
    scale = d_model**-0.5

    [q, k, v, mask, bias] = jax.tree.map(
        lambda x: jax.device_put(x, jax.local_devices(backend=platform)[0]),
        [q, k, v, mask, bias],
    )

    return q, k, v, bias, mask, scale


def require_platform(platform: str):
    """Helper function to check GPU requirement based on platform parameter."""
    if platform == "cuda" and jnp.ones(()).devices().pop().platform != "gpu":
        pytest.skip("This test requires a CUDA device.")


# Test configurations
SHAPE_CONFIGS = [
    (2, 4, 2, 8, 6, 32),  # Default
    (1, 2, 1, 4, 4, 16),  # Small
    (2, 8, 4, 16, 12, 32),  # Large
]

PRECISION_CONFIGS = [
    jax.lax.Precision.DEFAULT,
    jax.lax.Precision.HIGH,
    jax.lax.Precision.HIGHEST,
]


@pytest.mark.parametrize("platform", ["cpu", "cuda"])
@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_gradient_correctness_finite_differences(platform, precision):
    """Test gradient correctness using finite differences."""
    require_platform(platform)

    # Skip HIGHEST precision on CUDA due to backward pass limitation
    if platform == "cuda" and precision == jax.lax.Precision.HIGHEST:
        pytest.skip("HIGHEST precision not supported for backward pass on CUDA")

    q, k, v, bias, mask, scale = create_test_data(platform)

    def fn(q, k, v, bias):
        output, _, _ = cuex.triangle_attention(
            q, k, v, bias, mask, scale, precision=precision
        )
        return jnp.sum(output)

    check_grads(
        fn, (q, k, v, bias), order=1, modes=["rev"], eps=1e-1, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("platform", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "batch_size, n_nodes, n_heads, seq_len_qo, seq_len_kv, d_model", SHAPE_CONFIGS
)
def test_basic_functionality(
    platform, batch_size, n_nodes, n_heads, seq_len_qo, seq_len_kv, d_model
):
    """Basic test to ensure the function works."""
    require_platform(platform)

    q, k, v, bias, mask, scale = create_test_data(
        platform, batch_size, n_nodes, n_heads, seq_len_qo, seq_len_kv, d_model
    )

    def fn(q, k, v, bias, mask):
        return cuex.triangle_attention(q, k, v, bias, mask, scale)

    output, lse, amax = fn(q, k, v, bias, mask)
    assert output.shape == q.shape
    assert lse.shape == q.shape[:-1] + (1,)
    assert amax.shape == q.shape[:-1] + (1,)


@pytest.mark.parametrize("platform", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_dtype_support(platform, dtype, precision):
    """Test triangle attention with different dtypes and precision values."""
    require_platform(platform)

    q, k, v, bias, mask, scale = create_test_data(platform, dtype=dtype)

    output, lse, amax = cuex.triangle_attention(
        q, k, v, bias, mask, scale, precision=precision
    )
    assert output.shape == q.shape
    assert output.dtype == dtype
    assert lse.shape == q.shape[:-1] + (1,)
    assert amax.shape == q.shape[:-1] + (1,)


@pytest.mark.parametrize("platform", ["cpu", "cuda"])
def test_vmap(platform):
    require_platform(platform)

    batch_size = 2
    vmap_size = 2
    n_nodes = 2
    n_heads = 2
    seq_len_qo = 3
    seq_len_kv = 4
    d_model = 8

    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 5)
    dtype = jnp.float32

    # Create data with extra vmap dimension
    q = jax.random.normal(
        keys[0], (vmap_size, batch_size, n_nodes, n_heads, seq_len_qo, d_model), dtype
    )
    k = jax.random.normal(
        keys[1], (vmap_size, batch_size, n_nodes, n_heads, seq_len_kv, d_model), dtype
    )
    # purpusfully not vmapping v to test
    v = jax.random.normal(
        keys[2], (batch_size, n_nodes, n_heads, seq_len_kv, d_model), dtype
    )
    bias = jax.random.normal(
        keys[3], (vmap_size, batch_size, 1, n_heads, seq_len_qo, seq_len_kv), dtype
    )
    mask = jax.random.bernoulli(
        keys[4], 0.8, (vmap_size, batch_size, n_nodes, 1, 1, seq_len_kv)
    )
    scale = d_model**-0.5

    def grad_fn(q, k, v, bias, mask):
        def loss_fn(q, k, v, bias):
            output, _, _ = cuex.triangle_attention(
                q, k, v, bias, mask, scale, precision=jax.lax.Precision.HIGH
            )
            return jnp.sum(output)

        return jax.grad(loss_fn, argnums=(0, 1, 2, 3))(q, k, v, bias)

    dq, dk, dv, dbias = jax.vmap(grad_fn, (0, 0, None, 0, 0))(q, k, v, bias, mask)
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == (vmap_size,) + v.shape
    assert dbias.shape == bias.shape
