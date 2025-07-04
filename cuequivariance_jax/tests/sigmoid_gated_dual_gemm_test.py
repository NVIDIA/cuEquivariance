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
import jax.numpy as jnp
import numpy as np
import pytest
from jax import test_util

from cuequivariance_jax.triangle.sigmoid_gated_dual_gemm import (
    Precision,
    _sigmoid_gated_dual_gemm_reference,
    sigmoid_gated_dual_gemm,
    sigmoid_gated_dual_gemm_dual_x,
)


def test_sigmoid_gated_dual_gemm_comprehensive():
    """Comprehensive test covering API, shapes, batching, and basic functionality."""
    key = jax.random.PRNGKey(42)
    M, N, K = 64, 128, 256
    B = 4

    # Test data
    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)
    x2 = jax.random.normal(jax.random.split(key, 4)[0], (M, K), dtype=jnp.float32)

    # Batch data
    x_batch = jax.random.normal(
        jax.random.split(key, 5)[0], (B, M, K), dtype=jnp.float32
    )
    x2_batch = jax.random.normal(
        jax.random.split(key, 6)[0], (B, M, K), dtype=jnp.float32
    )

    # Test single input API
    output = sigmoid_gated_dual_gemm(x, w1, w2)
    assert output.shape == (M, N)

    # Test with mask
    output_masked = sigmoid_gated_dual_gemm(x, w1, w2, mask=mask)
    assert output_masked.shape == (M, N)

    # Test transpose_out
    output_transposed = sigmoid_gated_dual_gemm(x, w1, w2, transpose_out=True)
    assert output_transposed.shape == (N, M)
    assert jnp.allclose(output_transposed, output.T, atol=1e-5)

    # Test dual input API
    output_dual = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2)
    assert output_dual.shape == (M, N)

    # Test dual input with transpose
    output_dual_transposed = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, transpose_out=True
    )
    assert output_dual_transposed.shape == (N, M)
    assert jnp.allclose(output_dual_transposed, output_dual.T, atol=1e-5)

    # Test batch processing
    output_batch = sigmoid_gated_dual_gemm(x_batch, w1, w2)
    assert output_batch.shape == (B, M, N)

    output_dual_batch = sigmoid_gated_dual_gemm_dual_x(x_batch, x2_batch, w1, w2)
    assert output_dual_batch.shape == (B, M, N)

    # Test reference implementation
    output_ref = _sigmoid_gated_dual_gemm_reference(
        x,
        None,
        w1,
        w2,
        None,
        two_inputs=False,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output_ref.shape == (M, N)

    output_ref_dual = _sigmoid_gated_dual_gemm_reference(
        x,
        x2,
        w1,
        w2,
        None,
        two_inputs=True,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output_ref_dual.shape == (M, N)


def test_sigmoid_gated_dual_gemm_correctness():
    """Test correctness against manual computation for both single and dual input modes."""
    key = jax.random.PRNGKey(42)
    M, N, K = 4, 32, 32  # Use dimensions compatible with tile sizes

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)
    x2 = jax.random.normal(jax.random.split(key, 4)[0], (M, K), dtype=jnp.float32)

    tol = 1e-5

    # Test single input correctness
    expected_single = _sigmoid_gated_dual_gemm_reference(
        x, None, w1, w2, None, False, False, Precision.IEEE
    )
    output_single = sigmoid_gated_dual_gemm(x, w1, w2, precision=Precision.IEEE)
    np.testing.assert_allclose(output_single, expected_single, rtol=tol, atol=tol)

    # Test single input with mask
    expected_masked = expected_single * mask[:, None]
    output_masked = sigmoid_gated_dual_gemm(
        x, w1, w2, mask=mask, precision=Precision.IEEE
    )
    np.testing.assert_allclose(output_masked, expected_masked, rtol=tol, atol=tol)

    # Test dual input correctness
    expected_dual = _sigmoid_gated_dual_gemm_reference(
        x, x2, w1, w2, None, True, False, Precision.IEEE
    )
    output_dual = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, precision=Precision.IEEE
    )
    np.testing.assert_allclose(output_dual, expected_dual, rtol=tol, atol=tol)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_sigmoid_gated_dual_gemm_gradients_1(backend):
    """Test gradient computation for all modes."""
    M, N, K = 8, 32, 32  # Use smaller dimensions for faster gradient checking

    x = jax.random.normal(jax.random.key(0), (M, K), dtype=jnp.float32)
    x2 = jax.random.normal(jax.random.key(1), (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.key(2), (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.key(3), (N, K), dtype=jnp.float32)

    if backend == "cpu":
        device = jax.devices("cpu")[0]
    elif backend == "gpu":
        device = jax.devices("gpu")[0]

    [x, x2, w1, w2] = jax.device_put([x, x2, w1, w2], device)

    # Test single input gradients
    def single_input_fn(x, w1, w2):
        return jnp.sum(sigmoid_gated_dual_gemm(x, w1, w2, precision=Precision.IEEE))

    grads = jax.grad(single_input_fn, argnums=(0, 1, 2))(x, w1, w2)
    assert grads[0].shape == x.shape
    assert grads[1].shape == w1.shape
    assert grads[2].shape == w2.shape

    test_util.check_grads(single_input_fn, (x, w1, w2), order=1, modes=["rev"])

    # Test dual input gradients
    def dual_input_fn(x1, x2, w1, w2):
        return jnp.sum(
            sigmoid_gated_dual_gemm_dual_x(x1, x2, w1, w2, precision=Precision.IEEE)
        )

    grads_dual = jax.grad(dual_input_fn, argnums=(0, 1, 2, 3))(x, x2, w1, w2)
    assert grads_dual[0].shape == x.shape
    assert grads_dual[1].shape == x2.shape
    assert grads_dual[2].shape == w1.shape
    assert grads_dual[3].shape == w2.shape

    test_util.check_grads(dual_input_fn, (x, x2, w1, w2), order=1, modes=["rev"])


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_sigmoid_gated_dual_gemm_gradients_2(backend):
    """Test gradient computation for all modes."""
    key = jax.random.PRNGKey(42)
    M, N, K = 8, 32, 32  # Use smaller dimensions for faster gradient checking

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)

    if backend == "cpu":
        device = jax.devices("cpu")[0]
    elif backend == "gpu":
        device = jax.devices("gpu")[0]

    [x, w1, w2, mask] = jax.device_put([x, w1, w2, mask], device)

    # Test masked input gradients
    def masked_fn(x, w1, w2, mask):
        return jnp.sum(
            sigmoid_gated_dual_gemm(x, w1, w2, mask=mask, precision=Precision.IEEE)
        )

    test_util.check_grads(masked_fn, (x, w1, w2, mask), order=1, modes=["rev"])


@pytest.mark.parametrize(
    "precision", [Precision.DEFAULT, Precision.TF32, Precision.IEEE]
)
def test_sigmoid_gated_dual_gemm_precision_modes(precision):
    """Test different precision modes."""
    key = jax.random.PRNGKey(42)
    M, N, K = 32, 64, 128

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    x2 = jax.random.normal(jax.random.split(key, 3)[0], (M, K), dtype=jnp.float32)

    # Test single input with different precision
    output = sigmoid_gated_dual_gemm(x, w1, w2, precision=precision)
    assert output.shape == (M, N)

    # Test dual input with different precision
    output_dual = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2, precision=precision)
    assert output_dual.shape == (M, N)
