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

import os
from contextlib import contextmanager

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


@contextmanager
def temp_env_vars(**env_vars):
    """Context manager to temporarily set environment variables."""
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value

    try:
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_value


def create_test_data(M=32, N=64, K=128, include_mask=False, batch_size=None):
    """Create standard test data for sigmoid_gated_dual_gemm tests."""
    key = jax.random.key(42)
    data = {
        "x": jax.random.normal(key, (M, K), dtype=jnp.float32),
        "w1": jax.random.normal(jax.random.key(1), (N, K), dtype=jnp.float32),
        "w2": jax.random.normal(jax.random.key(2), (N, K), dtype=jnp.float32),
        "x2": jax.random.normal(jax.random.key(3), (M, K), dtype=jnp.float32),
    }

    if include_mask:
        data["mask"] = jax.random.uniform(jax.random.key(4), (M,), dtype=jnp.float32)

    if batch_size is not None:
        data["x_batch"] = jax.random.normal(
            jax.random.key(5), (batch_size, M, K), dtype=jnp.float32
        )
        data["x2_batch"] = jax.random.normal(
            jax.random.key(6), (batch_size, M, K), dtype=jnp.float32
        )

    return data


def validate_output(output, expected_shape, output_name="output"):
    """Validate output shape and check for NaN values."""
    assert output.shape == expected_shape, f"{output_name} shape mismatch"
    assert not jnp.any(jnp.isnan(output)), f"{output_name} contains NaN values"


def test_sigmoid_gated_dual_gemm_comprehensive():
    """Comprehensive test covering API, shapes, batching, and basic functionality."""
    M, N, K = 64, 128, 256
    B = 4

    # Create test data
    test_data = create_test_data(M, N, K, include_mask=True, batch_size=B)
    x, w1, w2, x2, mask = (
        test_data["x"],
        test_data["w1"],
        test_data["w2"],
        test_data["x2"],
        test_data["mask"],
    )
    x_batch, x2_batch = test_data["x_batch"], test_data["x2_batch"]

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
    M, N, K = 4, 32, 32  # Use dimensions compatible with tile sizes

    # Create test data
    test_data = create_test_data(M, N, K, include_mask=True)
    x, w1, w2, x2, mask = (
        test_data["x"],
        test_data["w1"],
        test_data["w2"],
        test_data["x2"],
        test_data["mask"],
    )

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
def test_sigmoid_gated_dual_gemm_gradients(backend):
    """Test gradient computation for all modes."""
    M, N, K = 8, 32, 32  # Use smaller dimensions for faster gradient checking

    # Create test data
    test_data = create_test_data(M, N, K, include_mask=True)
    x, w1, w2, x2, mask = (
        test_data["x"],
        test_data["w1"],
        test_data["w2"],
        test_data["x2"],
        test_data["mask"],
    )

    if backend == "cpu":
        device = jax.devices("cpu")[0]
    elif backend == "gpu":
        device = jax.devices("gpu")[0]

    [x, x2, w1, w2, mask] = jax.device_put([x, x2, w1, w2, mask], device)

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
    M, N, K = 32, 64, 128

    # Create test data
    test_data = create_test_data(M, N, K)
    x, w1, w2, x2 = test_data["x"], test_data["w1"], test_data["w2"], test_data["x2"]

    # Test single input with different precision
    output = sigmoid_gated_dual_gemm(x, w1, w2, precision=precision)
    assert output.shape == (M, N)

    # Test dual input with different precision
    output_dual = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2, precision=precision)
    assert output_dual.shape == (M, N)


@pytest.mark.parametrize("tuning_mode", ["AOT", "ONDEMAND", None])
def test_sigmoid_gated_dual_gemm_triton_tuning_modes(tuning_mode):
    """Test sigmoid_gated_dual_gemm with different CUEQ_TRITON_TUNING environment variable values."""
    env_vars = {
        "CUEQ_TRITON_TUNING": tuning_mode,
        "CUEQ_TRITON_IGNORE_EXISTING_CACHE": "1",
    }

    with temp_env_vars(**env_vars):
        # Create test data
        M, N, K = 32, 64, 128
        test_data = create_test_data(M, N, K)

        # Test single input mode
        output = sigmoid_gated_dual_gemm(
            test_data["x"], test_data["w1"], test_data["w2"]
        )
        validate_output(output, (M, N), "single input output")

        # Test dual input mode
        output_dual = sigmoid_gated_dual_gemm_dual_x(
            test_data["x"], test_data["x2"], test_data["w1"], test_data["w2"]
        )
        validate_output(output_dual, (M, N), "dual input output")


if __name__ == "__main__":
    test_sigmoid_gated_dual_gemm_triton_tuning_modes("ONDEMAND")
