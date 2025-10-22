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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cuequivariance_jax.triangle import Precision
from cuequivariance_jax.triangle._sigmoid_gated_dual_gemm import (
    sigmoid_gated_dual_gemm,
    sigmoid_gated_dual_gemm_dual_x,
    sigmoid_gated_dual_gemm_reference,
)
from jax import test_util

# Enable x64 support but test with fp32
jax.config.update("jax_enable_x64", True)
os.environ["CUEQ_TRITON_IGNORE_EXISTING_CACHE"] = (
    "1"  # needed to be set before the first call to the API
)


def create_test_data(
    M=32, N=64, K=128, include_mask=False, include_bias=False, batch_size=None
):
    """Create standard test data for sigmoid_gated_dual_gemm tests."""
    key = jax.random.key(42)
    data = {
        "x": jax.random.normal(key, (M, K), dtype=jnp.float32),
        "w1": jax.random.normal(jax.random.key(1), (N, K), dtype=jnp.float32),
        "w2": jax.random.normal(jax.random.key(2), (N, K), dtype=jnp.float32),
        "x2": jax.random.normal(jax.random.key(3), (M, K), dtype=jnp.float32),
    }

    if include_bias:
        data["b1"] = jax.random.normal(jax.random.key(7), (N,), dtype=jnp.float32)
        data["b2"] = jax.random.normal(jax.random.key(8), (N,), dtype=jnp.float32)

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
    M, N, K = 32, 64, 128
    B = 2

    # Create test data
    test_data = create_test_data(
        M, N, K, include_mask=True, include_bias=True, batch_size=B
    )
    x, w1, w2, x2, mask = (
        test_data["x"],
        test_data["w1"],
        test_data["w2"],
        test_data["x2"],
        test_data["mask"],
    )
    b1, b2 = test_data["b1"], test_data["b2"]
    x_batch, x2_batch = test_data["x_batch"], test_data["x2_batch"]

    # Test single input API
    output = sigmoid_gated_dual_gemm(x, w1, w2)
    assert output.shape == (M, N)

    # Test single input API with bias
    output_bias = sigmoid_gated_dual_gemm(x, w1, w2, b1=b1, b2=b2)
    assert output_bias.shape == (M, N)
    # Outputs should be different when bias is added
    assert not jnp.allclose(output, output_bias, atol=1e-5)

    # Test with mask
    output_masked = sigmoid_gated_dual_gemm(x, w1, w2, mask=mask)
    assert output_masked.shape == (M, N)

    # Test with mask and bias
    output_masked_bias = sigmoid_gated_dual_gemm(x, w1, w2, b1=b1, b2=b2, mask=mask)
    assert output_masked_bias.shape == (M, N)

    # Test transpose_out
    output_transposed = sigmoid_gated_dual_gemm(x, w1, w2, transpose_out=True)
    assert output_transposed.shape == (N, M)
    assert jnp.allclose(output_transposed, output.T, atol=1e-5)

    # Test transpose_out with bias
    output_transposed_bias = sigmoid_gated_dual_gemm(
        x, w1, w2, b1=b1, b2=b2, transpose_out=True
    )
    assert output_transposed_bias.shape == (N, M)
    assert jnp.allclose(output_transposed_bias, output_bias.T, atol=1e-5)

    # Test dual input API
    output_dual = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2)
    assert output_dual.shape == (M, N)

    # Test dual input API with bias
    output_dual_bias = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2, b1=b1, b2=b2)
    assert output_dual_bias.shape == (M, N)
    # Outputs should be different when bias is added
    assert not jnp.allclose(output_dual, output_dual_bias, atol=1e-5)

    # Test dual input with transpose
    output_dual_transposed = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, transpose_out=True
    )
    assert output_dual_transposed.shape == (N, M)
    assert jnp.allclose(output_dual_transposed, output_dual.T, atol=1e-5)

    # Test batch processing
    output_batch = sigmoid_gated_dual_gemm(x_batch, w1, w2)
    assert output_batch.shape == (B, M, N)

    # Test batch processing with bias
    output_batch_bias = sigmoid_gated_dual_gemm(x_batch, w1, w2, b1=b1, b2=b2)
    assert output_batch_bias.shape == (B, M, N)

    output_dual_batch = sigmoid_gated_dual_gemm_dual_x(x_batch, x2_batch, w1, w2)
    assert output_dual_batch.shape == (B, M, N)

    # Test batch processing dual input with bias
    output_dual_batch_bias = sigmoid_gated_dual_gemm_dual_x(
        x_batch, x2_batch, w1, w2, b1=b1, b2=b2
    )
    assert output_dual_batch_bias.shape == (B, M, N)

    # Test reference implementation
    output_ref = sigmoid_gated_dual_gemm_reference(
        x,
        None,
        w1,
        w2,
        None,
        None,
        None,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output_ref.shape == (M, N)

    # Test reference implementation with bias
    output_ref_bias = sigmoid_gated_dual_gemm_reference(
        x,
        None,
        w1,
        w2,
        b1,
        b2,
        None,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output_ref_bias.shape == (M, N)

    output_ref_dual = sigmoid_gated_dual_gemm_reference(
        x,
        x2,
        w1,
        w2,
        None,
        None,
        None,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output_ref_dual.shape == (M, N)

    # Test reference implementation dual with bias
    output_ref_dual_bias = sigmoid_gated_dual_gemm_reference(
        x,
        x2,
        w1,
        w2,
        b1,
        b2,
        None,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output_ref_dual_bias.shape == (M, N)


def test_sigmoid_gated_dual_gemm_correctness():
    """Test correctness against manual computation for both single and dual input modes."""
    M, N, K = 4, 32, 32

    # Create test data
    test_data = create_test_data(M, N, K, include_mask=True, include_bias=True)
    x, w1, w2, x2, mask = (
        test_data["x"],
        test_data["w1"],
        test_data["w2"],
        test_data["x2"],
        test_data["mask"],
    )
    b1, b2 = test_data["b1"], test_data["b2"]  # TODO insert above?

    tol = 1e-5

    # Test single input correctness without bias
    expected_single = sigmoid_gated_dual_gemm_reference(
        x, None, w1, w2, None, None, None, transpose_out=False, precision=Precision.IEEE
    )
    output_single = sigmoid_gated_dual_gemm(x, w1, w2, precision=Precision.IEEE)
    np.testing.assert_allclose(output_single, expected_single, rtol=tol, atol=tol)

    # Test single input correctness with bias
    expected_single_bias = sigmoid_gated_dual_gemm_reference(
        x, None, w1, w2, b1, b2, None, transpose_out=False, precision=Precision.IEEE
    )
    output_single_bias = sigmoid_gated_dual_gemm(
        x, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE
    )
    np.testing.assert_allclose(
        output_single_bias, expected_single_bias, rtol=tol, atol=tol
    )

    # Test single input with mask
    expected_masked = expected_single * mask[:, None]
    output_masked = sigmoid_gated_dual_gemm(
        x, w1, w2, mask=mask, precision=Precision.IEEE
    )
    np.testing.assert_allclose(output_masked, expected_masked, rtol=tol, atol=tol)

    # Test single input with mask and bias
    expected_masked_bias = expected_single_bias * mask[:, None]
    output_masked_bias = sigmoid_gated_dual_gemm(
        x, w1, w2, b1=b1, b2=b2, mask=mask, precision=Precision.IEEE
    )
    np.testing.assert_allclose(
        output_masked_bias, expected_masked_bias, rtol=tol, atol=tol
    )

    # Test dual input correctness without bias
    expected_dual = sigmoid_gated_dual_gemm_reference(
        x, x2, w1, w2, None, None, None, transpose_out=False, precision=Precision.IEEE
    )
    output_dual = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, precision=Precision.IEEE
    )
    np.testing.assert_allclose(output_dual, expected_dual, rtol=tol, atol=tol)

    # Test dual input correctness with bias
    expected_dual_bias = sigmoid_gated_dual_gemm_reference(
        x, x2, w1, w2, b1, b2, None, transpose_out=False, precision=Precision.IEEE
    )
    output_dual_bias = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE
    )
    np.testing.assert_allclose(output_dual_bias, expected_dual_bias, rtol=tol, atol=tol)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_sigmoid_gated_dual_gemm_gradients(backend):
    """Test gradient computation for all modes."""
    M, N, K = 4, 32, 32

    # Create test data
    test_data = create_test_data(M, N, K, include_mask=True, include_bias=True)
    x, w1, w2, x2, mask = (
        test_data["x"],
        test_data["w1"],
        test_data["w2"],
        test_data["x2"],
        test_data["mask"],
    )
    b1, b2 = test_data["b1"], test_data["b2"]

    if backend == "cpu":
        device = jax.devices("cpu")[0]
    elif backend == "gpu":
        try:
            device = jax.devices("gpu")[0]
        except RuntimeError:
            pytest.skip("No GPU available for testing")

    [x, x2, w1, w2, b1, b2, mask] = jax.device_put(
        [x, x2, w1, w2, b1, b2, mask], device
    )

    # Test single input gradients without bias
    def single_input_fn(x, w1, w2):
        return jnp.sum(sigmoid_gated_dual_gemm(x, w1, w2, precision=Precision.IEEE))

    grads = jax.grad(single_input_fn, argnums=(0, 1, 2))(x, w1, w2)
    assert grads[0].shape == x.shape
    assert grads[1].shape == w1.shape
    assert grads[2].shape == w2.shape

    # Test gradient correctness
    test_util.check_grads(single_input_fn, (x, w1, w2), order=1, modes=["rev"])

    # Test single input gradients with bias
    def single_input_bias_fn(x, w1, w2, b1, b2):
        return jnp.sum(
            sigmoid_gated_dual_gemm(x, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE)
        )

    grads_bias = jax.grad(single_input_bias_fn, argnums=(0, 1, 2, 3, 4))(
        x, w1, w2, b1, b2
    )
    assert grads_bias[0].shape == x.shape
    assert grads_bias[1].shape == w1.shape
    assert grads_bias[2].shape == w2.shape
    assert grads_bias[3].shape == b1.shape
    assert grads_bias[4].shape == b2.shape

    # Test dual input gradients without bias
    def dual_input_fn(x1, x2, w1, w2):
        return jnp.sum(
            sigmoid_gated_dual_gemm_dual_x(x1, x2, w1, w2, precision=Precision.IEEE)
        )

    grads_dual = jax.grad(dual_input_fn, argnums=(0, 1, 2, 3))(x, x2, w1, w2)
    assert grads_dual[0].shape == x.shape
    assert grads_dual[1].shape == x2.shape
    assert grads_dual[2].shape == w1.shape
    assert grads_dual[3].shape == w2.shape

    # Test dual input gradients with bias
    def dual_input_bias_fn(x1, x2, w1, w2, b1, b2):
        return jnp.sum(
            sigmoid_gated_dual_gemm_dual_x(
                x1, x2, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE
            )
        )

    grads_dual_bias = jax.grad(dual_input_bias_fn, argnums=(0, 1, 2, 3, 4, 5))(
        x, x2, w1, w2, b1, b2
    )
    assert grads_dual_bias[0].shape == x.shape
    assert grads_dual_bias[1].shape == x2.shape
    assert grads_dual_bias[2].shape == w1.shape
    assert grads_dual_bias[3].shape == w2.shape
    assert grads_dual_bias[4].shape == b1.shape
    assert grads_dual_bias[5].shape == b2.shape

    # Test masked input gradients
    def masked_fn(x, w1, w2, mask):
        return jnp.sum(
            sigmoid_gated_dual_gemm(x, w1, w2, mask=mask, precision=Precision.IEEE)
        )

    grads_masked = jax.grad(masked_fn, argnums=(0, 1, 2, 3))(x, w1, w2, mask)
    assert grads_masked[0].shape == x.shape
    assert grads_masked[1].shape == w1.shape
    assert grads_masked[2].shape == w2.shape
    assert grads_masked[3].shape == mask.shape


@pytest.mark.parametrize(
    "precision", [Precision.DEFAULT, Precision.TF32, Precision.IEEE]
)
def test_sigmoid_gated_dual_gemm_precision_modes(precision):
    """Test different precision modes."""
    M, N, K = 32, 64, 128

    # Create test data
    test_data = create_test_data(M, N, K, include_bias=True)
    x, w1, w2, x2 = test_data["x"], test_data["w1"], test_data["w2"], test_data["x2"]
    b1, b2 = test_data["b1"], test_data["b2"]

    # Test single input with different precision
    output = sigmoid_gated_dual_gemm(x, w1, w2, precision=precision)
    assert output.shape == (M, N)

    # Test single input with bias and different precision
    output_bias = sigmoid_gated_dual_gemm(x, w1, w2, b1=b1, b2=b2, precision=precision)
    assert output_bias.shape == (M, N)

    # Test dual input with different precision
    output_dual = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2, precision=precision)
    assert output_dual.shape == (M, N)

    # Test dual input with bias and different precision
    output_dual_bias = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, b1=b1, b2=b2, precision=precision
    )
    assert output_dual_bias.shape == (M, N)


@pytest.mark.slow
@pytest.mark.parametrize("tuning_mode", ["ONDEMAND", None])  # "AOT" is too slow to test
def test_sigmoid_gated_dual_gemm_triton_tuning_modes(tuning_mode, monkeypatch):
    """Test sigmoid_gated_dual_gemm with different CUEQ_TRITON_TUNING environment variable values."""
    assert os.environ["CUEQ_TRITON_IGNORE_EXISTING_CACHE"] == "1"

    # Configure environment variables using pytest's monkeypatch fixture
    if tuning_mode is None:
        monkeypatch.delenv("CUEQ_TRITON_TUNING", raising=False)
    else:
        monkeypatch.setenv("CUEQ_TRITON_TUNING", tuning_mode)

    # Create test data
    M, N, K = 32, 64, 128
    test_data = create_test_data(M, N, K)

    # Test single input mode
    output = sigmoid_gated_dual_gemm(test_data["x"], test_data["w1"], test_data["w2"])
    validate_output(output, (M, N), "single input output")

    # Test dual input mode
    output_dual = sigmoid_gated_dual_gemm_dual_x(
        test_data["x"], test_data["x2"], test_data["w1"], test_data["w2"]
    )
    validate_output(output_dual, (M, N), "dual input output")

    # Test backward pass for single input mode
    def single_input_fn(x, w1, w2):
        return jnp.sum(sigmoid_gated_dual_gemm(x, w1, w2))

    grads = jax.grad(single_input_fn, argnums=(0, 1, 2))(
        test_data["x"], test_data["w1"], test_data["w2"]
    )
    assert grads[0].shape == test_data["x"].shape
    assert grads[1].shape == test_data["w1"].shape
    assert grads[2].shape == test_data["w2"].shape

    # Test backward pass for dual input mode
    def dual_input_fn(x1, x2, w1, w2):
        return jnp.sum(sigmoid_gated_dual_gemm_dual_x(x1, x2, w1, w2))

    grads_dual = jax.grad(dual_input_fn, argnums=(0, 1, 2, 3))(
        test_data["x"], test_data["x2"], test_data["w1"], test_data["w2"]
    )
    assert grads_dual[0].shape == test_data["x"].shape
    assert grads_dual[1].shape == test_data["x2"].shape
    assert grads_dual[2].shape == test_data["w1"].shape
    assert grads_dual[3].shape == test_data["w2"].shape


def test_sigmoid_gated_dual_gemm_batched_mask_reshaping():
    """Test mask reshaping with batched inputs."""
    B, M, N, K = 2, 64, 64, 64
    key = jax.random.key(42)
    x_batch = jax.random.normal(key, (B, M, K), dtype=jnp.float32)
    x2_batch = jax.random.normal(jax.random.key(1), (B, M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.key(2), (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.key(3), (N, K), dtype=jnp.float32)
    mask_batch = jax.random.uniform(jax.random.key(4), (B, M), dtype=jnp.float32)

    # Test both single and dual input modes
    output_single = sigmoid_gated_dual_gemm(
        x_batch, w1, w2, mask=mask_batch, precision=Precision.IEEE
    )
    output_dual = sigmoid_gated_dual_gemm_dual_x(
        x_batch, x2_batch, w1, w2, mask=mask_batch, precision=Precision.IEEE
    )

    validate_output(output_single, (B, M, N), "batched single input")
    validate_output(output_dual, (B, M, N), "batched dual input")

    # Verify correctness against manual masking
    output_no_mask = sigmoid_gated_dual_gemm(x_batch, w1, w2, precision=Precision.IEEE)
    expected_masked = output_no_mask * mask_batch[..., None]
    np.testing.assert_allclose(output_single, expected_masked, rtol=1e-5, atol=1e-5)


def test_sigmoid_gated_dual_gemm_4d_input_3d_mask():
    """Test 4D input with 3D mask (original triangle_multiplicative_update bug case)."""
    B, H, W, D = 1, 32, 32, 64
    N = 2 * D
    key = jax.random.key(42)
    x_4d = jax.random.normal(key, (B, H, W, D), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.key(1), (N, D), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.key(2), (N, D), dtype=jnp.float32)
    mask_3d = jnp.ones((B, H, W), dtype=jnp.float32)

    # Test both single and dual input modes with transpose_out=True
    output_single = sigmoid_gated_dual_gemm(
        x_4d, w1, w2, mask=mask_3d, transpose_out=True, precision=Precision.IEEE
    )
    x2_4d = jax.random.normal(jax.random.key(3), (B, H, W, D), dtype=jnp.float32)
    output_dual = sigmoid_gated_dual_gemm_dual_x(
        x_4d, x2_4d, w1, w2, mask=mask_3d, transpose_out=True, precision=Precision.IEEE
    )

    expected_shape = (N, B, H, W)
    validate_output(output_single, expected_shape, "4D input with 3D mask (single)")
    validate_output(output_dual, expected_shape, "4D input with 3D mask (dual)")

    # Verify correctness: mask is all ones so should match unmasked output
    output_no_mask = sigmoid_gated_dual_gemm(
        x_4d, w1, w2, transpose_out=True, precision=Precision.IEEE
    )
    np.testing.assert_allclose(output_single, output_no_mask, rtol=1e-5, atol=1e-5)


def test_sigmoid_gated_dual_gemm_cpu_execution():
    """Test CPU execution with jax.device_put (verifies _reference_forward path)."""
    cpu_device = jax.devices("cpu")[0]
    B, H, W, D = 1, 32, 32, 64
    N = 128

    # Create test data and move to CPU
    key = jax.random.key(42)
    x_cpu = jax.device_put(
        jax.random.normal(key, (B, H, W, D), dtype=jnp.float32), cpu_device
    )
    w1_cpu = jax.device_put(
        jax.random.normal(jax.random.key(1), (N, D), dtype=jnp.float32), cpu_device
    )
    w2_cpu = jax.device_put(
        jax.random.normal(jax.random.key(2), (N, D), dtype=jnp.float32), cpu_device
    )
    mask_cpu = jax.device_put(jnp.ones((B, H, W), dtype=jnp.float32), cpu_device)

    # Test both modes on CPU (this was failing before our fix)
    output_single = sigmoid_gated_dual_gemm(
        x_cpu,
        w1_cpu,
        w2_cpu,
        mask=mask_cpu,
        transpose_out=True,
        precision=Precision.IEEE,
    )
    x2_cpu = jax.device_put(
        jax.random.normal(jax.random.key(3), (B, H, W, D), dtype=jnp.float32),
        cpu_device,
    )
    output_dual = sigmoid_gated_dual_gemm_dual_x(
        x_cpu,
        x2_cpu,
        w1_cpu,
        w2_cpu,
        mask=mask_cpu,
        transpose_out=True,
        precision=Precision.IEEE,
    )

    # Verify outputs are on CPU with correct shapes
    assert output_single.device == cpu_device
    assert output_dual.device == cpu_device
    expected_shape = (N, B, H, W)
    validate_output(output_single, expected_shape, "CPU single input")
    validate_output(output_dual, expected_shape, "CPU dual input")

    # Verify correctness: all-ones mask should match unmasked output
    output_no_mask = sigmoid_gated_dual_gemm(
        x_cpu, w1_cpu, w2_cpu, transpose_out=True, precision=Precision.IEEE
    )
    np.testing.assert_allclose(output_single, output_no_mask, rtol=1e-6, atol=1e-6)


def test_sigmoid_gated_dual_gemm_bias_functionality():
    """Comprehensive test for bias functionality."""
    M, N, K = 32, 64, 128

    # Create test data
    test_data = create_test_data(M, N, K, include_mask=True, include_bias=True)
    x, w1, w2, x2, mask = (
        test_data["x"],
        test_data["w1"],
        test_data["w2"],
        test_data["x2"],
        test_data["mask"],
    )
    b1, b2 = test_data["b1"], test_data["b2"]

    # Test that bias changes the output
    output_no_bias = sigmoid_gated_dual_gemm(
        x, w1, w2, precision=Precision.IEEE, fallback=True
    )
    output_with_bias = sigmoid_gated_dual_gemm(
        x, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE, fallback=True
    )

    # Outputs should be different when bias is added
    assert not jnp.allclose(output_no_bias, output_with_bias, atol=1e-6)

    # Test that only b1 changes output
    output_b1_only = sigmoid_gated_dual_gemm(
        x, w1, w2, b1=b1, precision=Precision.IEEE, fallback=True
    )
    assert not jnp.allclose(output_no_bias, output_b1_only, atol=1e-6)
    assert not jnp.allclose(output_with_bias, output_b1_only, atol=1e-6)

    # Test that only b2 changes output
    output_b2_only = sigmoid_gated_dual_gemm(
        x, w1, w2, b2=b2, precision=Precision.IEEE, fallback=True
    )
    assert not jnp.allclose(output_no_bias, output_b2_only, atol=1e-6)
    assert not jnp.allclose(output_with_bias, output_b2_only, atol=1e-6)
    assert not jnp.allclose(output_b1_only, output_b2_only, atol=1e-6)

    # Test dual input mode
    output_dual_no_bias = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, precision=Precision.IEEE, fallback=True
    )
    output_dual_with_bias = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE, fallback=True
    )

    # Outputs should be different when bias is added
    assert not jnp.allclose(output_dual_no_bias, output_dual_with_bias, atol=1e-6)

    # Test bias gradients
    def loss_fn_bias(b1, b2):
        return jnp.sum(
            sigmoid_gated_dual_gemm(
                x, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE, fallback=True
            )
        )

    grad_fn = jax.grad(loss_fn_bias, argnums=(0, 1))
    grad_b1, grad_b2 = grad_fn(b1, b2)

    # Gradients should have correct shapes and not be zero
    assert grad_b1.shape == b1.shape
    assert grad_b2.shape == b2.shape
    assert not jnp.allclose(grad_b1, 0.0, atol=1e-6)
    assert not jnp.allclose(grad_b2, 0.0, atol=1e-6)

    # Test bias gradients for dual input
    def dual_loss_fn_bias(b1, b2):
        return jnp.sum(
            sigmoid_gated_dual_gemm_dual_x(
                x, x2, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE, fallback=True
            )
        )

    dual_grad_fn = jax.grad(dual_loss_fn_bias, argnums=(0, 1))
    dual_grad_b1, dual_grad_b2 = dual_grad_fn(b1, b2)

    # Gradients should have correct shapes and not be zero
    assert dual_grad_b1.shape == b1.shape
    assert dual_grad_b2.shape == b2.shape
    assert not jnp.allclose(dual_grad_b1, 0.0, atol=1e-6)
    assert not jnp.allclose(dual_grad_b2, 0.0, atol=1e-6)

    # Test bias with masking
    output_masked_no_bias = sigmoid_gated_dual_gemm(
        x, w1, w2, mask=mask, precision=Precision.IEEE, fallback=True
    )
    output_masked_with_bias = sigmoid_gated_dual_gemm(
        x, w1, w2, b1=b1, b2=b2, mask=mask, precision=Precision.IEEE, fallback=True
    )

    # Outputs should be different when bias is added, even with masking
    assert not jnp.allclose(output_masked_no_bias, output_masked_with_bias, atol=1e-6)

    # Test that bias works with different shapes
    B = 2
    x_batch = jax.random.normal(jax.random.key(42), (B, M, K), dtype=jnp.float32)
    output_batch_no_bias = sigmoid_gated_dual_gemm(
        x_batch, w1, w2, precision=Precision.IEEE, fallback=True
    )
    output_batch_with_bias = sigmoid_gated_dual_gemm(
        x_batch, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE, fallback=True
    )

    assert output_batch_no_bias.shape == (B, M, N)
    assert output_batch_with_bias.shape == (B, M, N)
    assert not jnp.allclose(output_batch_no_bias, output_batch_with_bias, atol=1e-6)


def test_sigmoid_gated_dual_gemm_vmap():
    """Test vmap functionality with batching rule."""
    M, N, K = 16, 32, 64
    B = 2

    # Create test data
    key = jax.random.key(42)
    x_batch = jax.random.normal(key, (B, M, K), dtype=jnp.float32)
    x2_batch = jax.random.normal(jax.random.key(1), (B, M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.key(2), (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.key(3), (N, K), dtype=jnp.float32)
    b1 = jax.random.normal(jax.random.key(4), (N,), dtype=jnp.float32)
    b2 = jax.random.normal(jax.random.key(5), (N,), dtype=jnp.float32)
    mask_batch = jax.random.uniform(jax.random.key(6), (B, M), dtype=jnp.float32)

    # Test vmap for single input mode without bias
    vmapped_single = jax.vmap(
        lambda x: sigmoid_gated_dual_gemm(x, w1, w2, precision=Precision.IEEE)
    )(x_batch)

    # Test vmap for single input mode with bias
    vmapped_single_bias = jax.vmap(
        lambda x: sigmoid_gated_dual_gemm(
            x, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE
        )
    )(x_batch)

    # Test vmap for single input mode with mask
    vmapped_single_mask = jax.vmap(
        lambda x, mask: sigmoid_gated_dual_gemm(
            x, w1, w2, mask=mask, precision=Precision.IEEE
        )
    )(x_batch, mask_batch)

    # Test vmap for single input mode with bias and mask
    vmapped_single_bias_mask = jax.vmap(
        lambda x, mask: sigmoid_gated_dual_gemm(
            x, w1, w2, b1=b1, b2=b2, mask=mask, precision=Precision.IEEE
        )
    )(x_batch, mask_batch)

    # Test vmap for dual input mode without bias
    vmapped_dual = jax.vmap(
        lambda x1, x2: sigmoid_gated_dual_gemm_dual_x(
            x1, x2, w1, w2, precision=Precision.IEEE
        )
    )(x_batch, x2_batch)

    # Test vmap for dual input mode with bias
    vmapped_dual_bias = jax.vmap(
        lambda x1, x2: sigmoid_gated_dual_gemm_dual_x(
            x1, x2, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE
        )
    )(x_batch, x2_batch)

    # Test vmap for dual input mode with mask
    vmapped_dual_mask = jax.vmap(
        lambda x1, x2, mask: sigmoid_gated_dual_gemm_dual_x(
            x1, x2, w1, w2, mask=mask, precision=Precision.IEEE
        )
    )(x_batch, x2_batch, mask_batch)

    # Test vmap for dual input mode with bias and mask
    vmapped_dual_bias_mask = jax.vmap(
        lambda x1, x2, mask: sigmoid_gated_dual_gemm_dual_x(
            x1, x2, w1, w2, b1=b1, b2=b2, mask=mask, precision=Precision.IEEE
        )
    )(x_batch, x2_batch, mask_batch)

    # Test vmap with transpose_out=True
    vmapped_single_transpose = jax.vmap(
        lambda x: sigmoid_gated_dual_gemm(
            x, w1, w2, transpose_out=True, precision=Precision.IEEE
        )
    )(x_batch)

    vmapped_dual_transpose = jax.vmap(
        lambda x1, x2: sigmoid_gated_dual_gemm_dual_x(
            x1, x2, w1, w2, transpose_out=True, precision=Precision.IEEE
        )
    )(x_batch, x2_batch)

    # Verify shapes
    validate_output(vmapped_single, (B, M, N), "vmap single input")
    validate_output(vmapped_single_bias, (B, M, N), "vmap single input with bias")
    validate_output(vmapped_single_mask, (B, M, N), "vmap single input with mask")
    validate_output(
        vmapped_single_bias_mask, (B, M, N), "vmap single input with bias and mask"
    )
    validate_output(vmapped_dual, (B, M, N), "vmap dual input")
    validate_output(vmapped_dual_bias, (B, M, N), "vmap dual input with bias")
    validate_output(vmapped_dual_mask, (B, M, N), "vmap dual input with mask")
    validate_output(
        vmapped_dual_bias_mask, (B, M, N), "vmap dual input with bias and mask"
    )
    validate_output(vmapped_single_transpose, (B, N, M), "vmap single input transpose")
    validate_output(vmapped_dual_transpose, (B, N, M), "vmap dual input transpose")


def test_sigmoid_gated_dual_gemm_vmap_backward():
    """Test vmap functionality for backward pass (gradient computation)."""
    M, N, K = 16, 32, 64
    B = 2

    # Create test data
    key = jax.random.key(42)
    x_batch = jax.random.normal(key, (B, M, K), dtype=jnp.float32)
    x2_batch = jax.random.normal(jax.random.key(1), (B, M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.key(2), (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.key(3), (N, K), dtype=jnp.float32)
    b1 = jax.random.normal(jax.random.key(4), (N,), dtype=jnp.float32)
    b2 = jax.random.normal(jax.random.key(5), (N,), dtype=jnp.float32)
    mask_batch = jax.random.uniform(jax.random.key(6), (B, M), dtype=jnp.float32)

    # Test vmap for single input gradient computation without bias
    def single_grad_fn(x, w1, w2):
        def loss_fn(x, w1, w2):
            return jnp.sum(sigmoid_gated_dual_gemm(x, w1, w2, precision=Precision.IEEE))

        return jax.grad(loss_fn, argnums=(0, 1, 2))(x, w1, w2)

    vmapped_single_grads = jax.vmap(single_grad_fn, in_axes=(0, None, None))(
        x_batch, w1, w2
    )

    # Verify gradient shapes
    grad_x, grad_w1, grad_w2 = vmapped_single_grads
    validate_output(grad_x, (B, M, K), "vmap single input grad_x")
    validate_output(grad_w1, (B, N, K), "vmap single input grad_w1")
    validate_output(grad_w2, (B, N, K), "vmap single input grad_w2")

    # Test vmap for single input gradient computation with bias
    def single_bias_grad_fn(x, w1, w2, b1, b2):
        def loss_fn(x, w1, w2, b1, b2):
            return jnp.sum(
                sigmoid_gated_dual_gemm(
                    x, w1, w2, b1=b1, b2=b2, precision=Precision.IEEE
                )
            )

        return jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(x, w1, w2, b1, b2)

    vmapped_single_bias_grads = jax.vmap(
        single_bias_grad_fn, in_axes=(0, None, None, None, None)
    )(x_batch, w1, w2, b1, b2)

    # Verify gradient shapes
    grad_x, grad_w1, grad_w2, grad_b1, grad_b2 = vmapped_single_bias_grads
    validate_output(grad_x, (B, M, K), "vmap single input bias grad_x")
    validate_output(grad_w1, (B, N, K), "vmap single input bias grad_w1")
    validate_output(grad_w2, (B, N, K), "vmap single input bias grad_w2")
    validate_output(grad_b1, (B, N), "vmap single input bias grad_b1")
    validate_output(grad_b2, (B, N), "vmap single input bias grad_b2")

    # Test vmap for dual input gradient computation without bias
    def dual_grad_fn(x1, x2, w1, w2):
        def loss_fn(x1, x2, w1, w2):
            return jnp.sum(
                sigmoid_gated_dual_gemm_dual_x(x1, x2, w1, w2, precision=Precision.IEEE)
            )

        return jax.grad(loss_fn, argnums=(0, 1, 2, 3))(x1, x2, w1, w2)

    vmapped_dual_grads = jax.vmap(dual_grad_fn, in_axes=(0, 0, None, None))(
        x_batch, x2_batch, w1, w2
    )

    # Verify gradient shapes
    grad_x1, grad_x2, grad_w1, grad_w2 = vmapped_dual_grads
    validate_output(grad_x1, (B, M, K), "vmap dual input grad_x1")
    validate_output(grad_x2, (B, M, K), "vmap dual input grad_x2")
    validate_output(grad_w1, (B, N, K), "vmap dual input grad_w1")
    validate_output(grad_w2, (B, N, K), "vmap dual input grad_w2")

    # Test vmap for dual input gradient computation with bias and mask
    def dual_bias_mask_grad_fn(x1, x2, w1, w2, b1, b2, mask):
        def loss_fn(x1, x2, w1, w2, b1, b2, mask):
            return jnp.sum(
                sigmoid_gated_dual_gemm_dual_x(
                    x1, x2, w1, w2, b1=b1, b2=b2, mask=mask, precision=Precision.IEEE
                )
            )

        return jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(
            x1, x2, w1, w2, b1, b2, mask
        )

    vmapped_dual_bias_mask_grads = jax.vmap(
        dual_bias_mask_grad_fn, in_axes=(0, 0, None, None, None, None, 0)
    )(x_batch, x2_batch, w1, w2, b1, b2, mask_batch)

    # Verify gradient shapes
    grad_x1, grad_x2, grad_w1, grad_w2, grad_b1, grad_b2, grad_mask = (
        vmapped_dual_bias_mask_grads
    )
    validate_output(grad_x1, (B, M, K), "vmap dual input bias mask grad_x1")
    validate_output(grad_x2, (B, M, K), "vmap dual input bias mask grad_x2")
    validate_output(grad_w1, (B, N, K), "vmap dual input bias mask grad_w1")
    validate_output(grad_w2, (B, N, K), "vmap dual input bias mask grad_w2")
    validate_output(grad_b1, (B, N), "vmap dual input bias mask grad_b1")
    validate_output(grad_b2, (B, N), "vmap dual input bias mask grad_b2")
    validate_output(grad_mask, (B, M), "vmap dual input bias mask grad_mask")

    # Test vmap with transpose_out=True
    def single_transpose_grad_fn(x, w1, w2):
        def loss_fn(x, w1, w2):
            return jnp.sum(
                sigmoid_gated_dual_gemm(
                    x, w1, w2, transpose_out=True, precision=Precision.IEEE
                )
            )

        return jax.grad(loss_fn, argnums=(0, 1, 2))(x, w1, w2)

    vmapped_transpose_grads = jax.vmap(
        single_transpose_grad_fn, in_axes=(0, None, None)
    )(x_batch, w1, w2)

    # Verify gradient shapes (should be same as non-transposed case)
    grad_x, grad_w1, grad_w2 = vmapped_transpose_grads
    validate_output(grad_x, (B, M, K), "vmap transpose grad_x")
    validate_output(grad_w1, (B, N, K), "vmap transpose grad_w1")
    validate_output(grad_w2, (B, N, K), "vmap transpose grad_w2")
