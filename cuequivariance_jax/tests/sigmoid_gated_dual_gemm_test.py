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
import pytest
from jax import test_util

from cuequivariance_jax.triangle.sigmoid_gated_dual_gemm import (
    Precision,
    sigmoid_gated_dual_gemm,
    sigmoid_gated_dual_gemm_dual_x,
    sigmoid_gated_dual_gemm_reference_forward,
)


def test_sigmoid_gated_dual_gemm_reference():
    """Test the reference implementation."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    M, N, K = 64, 128, 256

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)

    # Test single input mode
    # Test without mask
    output = sigmoid_gated_dual_gemm_reference_forward(
        x,
        None,
        w1,
        w2,
        None,
        two_inputs=False,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output.shape == (M, N)

    # Test with mask
    output_masked = sigmoid_gated_dual_gemm_reference_forward(
        x,
        None,
        w1,
        w2,
        mask,
        two_inputs=False,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output_masked.shape == (M, N)

    # Test transpose_out
    output_transposed = sigmoid_gated_dual_gemm_reference_forward(
        x,
        None,
        w1,
        w2,
        None,
        two_inputs=False,
        transpose_out=True,
        precision=Precision.DEFAULT,
    )
    assert output_transposed.shape == (N, M)

    # Test two input mode
    x2 = jax.random.normal(jax.random.split(key, 4)[0], (M, K), dtype=jnp.float32)
    output_dual = sigmoid_gated_dual_gemm_reference_forward(
        x,
        x2,
        w1,
        w2,
        None,
        two_inputs=True,
        transpose_out=False,
        precision=Precision.DEFAULT,
    )
    assert output_dual.shape == (M, N)


def test_sigmoid_gated_dual_gemm_api():
    """Test the high-level API functions."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    M, N, K = 64, 128, 256

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)

    # Test single input API
    output = sigmoid_gated_dual_gemm(x, w1, w2)
    assert output.shape == (M, N)

    # Test with mask
    output_masked = sigmoid_gated_dual_gemm(x, w1, w2, mask=mask)
    assert output_masked.shape == (M, N)

    # Test transpose_out
    output_transposed = sigmoid_gated_dual_gemm(x, w1, w2, transpose_out=True)
    assert output_transposed.shape == (N, M)

    # Test dual input API
    x2 = jax.random.normal(jax.random.split(key, 4)[0], (M, K), dtype=jnp.float32)
    output_dual = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2)
    assert output_dual.shape == (M, N)


def test_sigmoid_gated_dual_gemm_batch():
    """Test batch processing."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    B, M, N, K = 4, 32, 64, 128

    x = jax.random.normal(key, (B, M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)

    # Test single input batch
    output = sigmoid_gated_dual_gemm(x, w1, w2)
    assert output.shape == (B, M, N)

    # Test dual input batch
    x2 = jax.random.normal(jax.random.split(key, 3)[0], (B, M, K), dtype=jnp.float32)
    output_dual = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2)
    assert output_dual.shape == (B, M, N)


def test_sigmoid_gated_dual_gemm_gradients():
    """Test gradient computation."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    M, N, K = 32, 64, 128

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)

    # Test single input gradients
    def single_input_fn(x, w1, w2):
        return jnp.sum(sigmoid_gated_dual_gemm(x, w1, w2))

    grad_fn = jax.grad(single_input_fn, argnums=(0, 1, 2))
    grads = grad_fn(x, w1, w2)

    assert grads[0].shape == x.shape
    assert grads[1].shape == w1.shape
    assert grads[2].shape == w2.shape

    # Test dual input gradients
    x2 = jax.random.normal(jax.random.split(key, 3)[0], (M, K), dtype=jnp.float32)

    def dual_input_fn(x1, x2, w1, w2):
        return jnp.sum(sigmoid_gated_dual_gemm_dual_x(x1, x2, w1, w2))

    grad_fn_dual = jax.grad(dual_input_fn, argnums=(0, 1, 2, 3))
    grads_dual = grad_fn_dual(x, x2, w1, w2)

    assert grads_dual[0].shape == x.shape
    assert grads_dual[1].shape == x2.shape
    assert grads_dual[2].shape == w1.shape
    assert grads_dual[3].shape == w2.shape


def test_sigmoid_gated_dual_gemm_correctness():
    """Test correctness against manual computation."""
    # Set up simple test case
    key = jax.random.PRNGKey(42)
    M, N, K = 4, 32, 32  # Use dimensions compatible with tile sizes

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)

    # Manual computation
    acc_1 = jnp.dot(x, w1.T)
    acc_2 = jnp.dot(x, w2.T)
    acc_sig = jax.nn.sigmoid(acc_1)
    expected = acc_sig * acc_2

    # Our implementation
    output = sigmoid_gated_dual_gemm(x, w1, w2, precision=Precision.IEEE)

    # Check correctness
    diff = jnp.abs(output - expected)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    assert max_diff < 5e-2, f"Max difference too large: {max_diff}"
    assert mean_diff < 5e-3, f"Mean difference too large: {mean_diff}"


def test_sigmoid_gated_dual_gemm_with_mask():
    """Test correctness with mask."""
    # Set up simple test case
    key = jax.random.PRNGKey(42)
    M, N, K = 4, 32, 32  # Use dimensions compatible with tile sizes

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)

    # Manual computation
    acc_1 = jnp.dot(x, w1.T)
    acc_2 = jnp.dot(x, w2.T)
    acc_sig = jax.nn.sigmoid(acc_1)
    output_unmasked = acc_sig * acc_2
    expected = output_unmasked * mask[:, None]

    # Our implementation
    output = sigmoid_gated_dual_gemm(x, w1, w2, mask=mask, precision=Precision.IEEE)

    # Check correctness
    diff = jnp.abs(output - expected)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    # TODO  the tolerance is high, probably due to a bug?
    assert max_diff < 1e-2, f"Max difference too large: {max_diff}"
    assert mean_diff < 1e-3, f"Mean difference too large: {mean_diff}"


def test_sigmoid_gated_dual_gemm_dual_x_correctness():
    """Test correctness of dual input mode."""
    # Set up simple test case
    key = jax.random.PRNGKey(42)
    M, N, K = 4, 32, 32  # Use dimensions compatible with tile sizes

    x1 = jax.random.normal(key, (M, K), dtype=jnp.float32)
    x2 = jax.random.normal(jax.random.split(key, 1)[0], (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 3)[0], (N, K), dtype=jnp.float32)

    # Manual computation
    acc_1 = jnp.dot(x1, w1.T)
    acc_2 = jnp.dot(x2, w2.T)
    acc_sig = jax.nn.sigmoid(acc_1)
    expected = acc_sig * acc_2

    # Our implementation
    output = sigmoid_gated_dual_gemm_dual_x(x1, x2, w1, w2, precision=Precision.IEEE)

    # Check correctness
    diff = jnp.abs(output - expected)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    assert max_diff < 1e-2, f"Max difference too large: {max_diff}"
    assert mean_diff < 1e-3, f"Mean difference too large: {mean_diff}"


@pytest.mark.parametrize("precision", [Precision.DEFAULT, Precision.TF32])
def test_sigmoid_gated_dual_gemm_precision_modes(precision):
    """Test different precision modes."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    M, N, K = 32, 64, 128

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)

    # Test single input with different precision
    output = sigmoid_gated_dual_gemm(x, w1, w2, precision=precision)
    assert output.shape == (M, N)

    # Test dual input with different precision
    x2 = jax.random.normal(jax.random.split(key, 3)[0], (M, K), dtype=jnp.float32)
    output_dual = sigmoid_gated_dual_gemm_dual_x(x, x2, w1, w2, precision=precision)
    assert output_dual.shape == (M, N)


def test_sigmoid_gated_dual_gemm_transpose_out():
    """Test transpose_out functionality."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    M, N, K = 16, 32, 64

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)

    # Test single input with transpose
    output_normal = sigmoid_gated_dual_gemm(x, w1, w2, transpose_out=False)
    output_transposed = sigmoid_gated_dual_gemm(x, w1, w2, transpose_out=True)

    assert output_normal.shape == (M, N)
    assert output_transposed.shape == (N, M)

    # Check that transposed output equals transpose of normal output
    assert jnp.allclose(output_transposed, output_normal.T, atol=1e-5)

    # Test dual input with transpose
    x2 = jax.random.normal(jax.random.split(key, 3)[0], (M, K), dtype=jnp.float32)
    output_dual_normal = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, transpose_out=False
    )
    output_dual_transposed = sigmoid_gated_dual_gemm_dual_x(
        x, x2, w1, w2, transpose_out=True
    )

    assert output_dual_normal.shape == (M, N)
    assert output_dual_transposed.shape == (N, M)

    # Check that transposed output equals transpose of normal output
    assert jnp.allclose(output_dual_transposed, output_dual_normal.T, atol=1e-5)


def test_sigmoid_gated_dual_gemm_check_grads():
    """Test gradient correctness using jax.test_util.check_grads."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    M, N, K = 8, 32, 32  # Use smaller dimensions for faster gradient checking

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)

    # Test single input mode without mask
    def single_input_fn(x, w1, w2):
        return jnp.sum(sigmoid_gated_dual_gemm(x, w1, w2, precision=Precision.IEEE))

    test_util.check_grads(single_input_fn, (x, w1, w2), order=1, modes=["rev"])

    # Test single input mode with mask
    def single_input_masked_fn(x, w1, w2, mask):
        return jnp.sum(
            sigmoid_gated_dual_gemm(x, w1, w2, mask=mask, precision=Precision.IEEE)
        )

    # TODO fix later the gradient for masked input
    test_util.check_grads(
        single_input_masked_fn, (x, w1, w2, mask), order=1, modes=["rev"]
    )

    # Test dual input mode
    x2 = jax.random.normal(jax.random.split(key, 4)[0], (M, K), dtype=jnp.float32)

    def dual_input_fn(x1, x2, w1, w2):
        return jnp.sum(
            sigmoid_gated_dual_gemm_dual_x(x1, x2, w1, w2, precision=Precision.IEEE)
        )

    test_util.check_grads(dual_input_fn, (x, x2, w1, w2), order=1, modes=["rev"])

    # Test with transpose_out
    def single_input_transpose_fn(x, w1, w2):
        return jnp.sum(
            sigmoid_gated_dual_gemm(
                x, w1, w2, transpose_out=True, precision=Precision.IEEE
            )
        )

    test_util.check_grads(
        single_input_transpose_fn, (x, w1, w2), order=1, modes=["rev"]
    )

    # Test dual input with transpose_out
    def dual_input_transpose_fn(x1, x2, w1, w2):
        return jnp.sum(
            sigmoid_gated_dual_gemm_dual_x(
                x1, x2, w1, w2, transpose_out=True, precision=Precision.IEEE
            )
        )

    test_util.check_grads(
        dual_input_transpose_fn, (x, x2, w1, w2), order=1, modes=["rev"]
    )


def test_sigmoid_gated_dual_gemm_debug_mask_gradient():
    """Debug test to understand mask gradient computation."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    M, N, K = 4, 32, 32  # Dimensions compatible with tile sizes

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)

    # Reference implementation using pure JAX
    def reference_fn(x, w1, w2, mask):
        acc_1 = jnp.dot(x, w1.T)
        acc_2 = jnp.dot(x, w2.T)
        acc_sig = jax.nn.sigmoid(acc_1)
        output = acc_sig * acc_2
        if mask is not None:
            output = output * mask[:, None]
        return jnp.sum(output)

    # Our implementation
    def our_fn(x, w1, w2, mask):
        return jnp.sum(
            sigmoid_gated_dual_gemm(x, w1, w2, mask=mask, precision=Precision.IEEE)
        )

    # Compare forward pass
    ref_output = reference_fn(x, w1, w2, mask)
    our_output = our_fn(x, w1, w2, mask)

    print(f"Reference output: {ref_output}")
    print(f"Our output: {our_output}")
    print(f"Forward diff: {jnp.abs(ref_output - our_output)}")

    # Compare gradients
    ref_grad = jax.grad(reference_fn, argnums=(0, 1, 2, 3))(x, w1, w2, mask)
    our_grad = jax.grad(our_fn, argnums=(0, 1, 2, 3))(x, w1, w2, mask)

    print(f"\nReference mask gradient: {ref_grad[3]}")
    print(f"Our mask gradient: {our_grad[3]}")
    print(f"Mask gradient diff: {jnp.abs(ref_grad[3] - our_grad[3])}")
    print(f"Max mask gradient diff: {jnp.max(jnp.abs(ref_grad[3] - our_grad[3]))}")

    # Check gradient using check_grads on reference function
    try:
        test_util.check_grads(reference_fn, (x, w1, w2, mask), order=1, modes=["rev"])
        print("Reference function gradient check: PASSED")
    except Exception as e:
        print(f"Reference function gradient check: FAILED - {e}")

    # Check gradient using check_grads on our function
    try:
        test_util.check_grads(our_fn, (x, w1, w2, mask), order=1, modes=["rev"])
        print("Our function gradient check: PASSED")
    except Exception as e:
        print(f"Our function gradient check: FAILED - {e}")


def test_sigmoid_gated_dual_gemm_reference_only():
    """Test using only the reference implementation (no Triton)."""
    # Set up test data
    key = jax.random.PRNGKey(42)
    M, N, K = 4, 32, 32  # Dimensions compatible with tile sizes

    x = jax.random.normal(key, (M, K), dtype=jnp.float32)
    w1 = jax.random.normal(jax.random.split(key, 1)[0], (N, K), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.split(key, 2)[0], (N, K), dtype=jnp.float32)
    mask = jax.random.uniform(jax.random.split(key, 3)[0], (M,), dtype=jnp.float32)

    # Reference implementation using pure JAX
    def reference_fn(x, w1, w2, mask):
        acc_1 = jnp.dot(x, w1.T)
        acc_2 = jnp.dot(x, w2.T)
        acc_sig = jax.nn.sigmoid(acc_1)
        output = acc_sig * acc_2
        if mask is not None:
            output = output * mask[:, None]
        return jnp.sum(output)

    # Direct call to reference implementation
    def our_reference_fn(x, w1, w2, mask):
        return jnp.sum(
            sigmoid_gated_dual_gemm_reference_forward(
                x, None, w1, w2, mask, False, False, Precision.IEEE
            )
        )

    # Compare forward pass
    ref_output = reference_fn(x, w1, w2, mask)
    our_output = our_reference_fn(x, w1, w2, mask)

    print(f"Reference output: {ref_output}")
    print(f"Our reference output: {our_output}")
    print(f"Forward diff: {jnp.abs(ref_output - our_output)}")

    # Compare gradients
    ref_grad = jax.grad(reference_fn, argnums=(0, 1, 2, 3))(x, w1, w2, mask)
    our_grad = jax.grad(our_reference_fn, argnums=(0, 1, 2, 3))(x, w1, w2, mask)

    print(f"\nReference mask gradient: {ref_grad[3]}")
    print(f"Our reference mask gradient: {our_grad[3]}")
    print(f"Mask gradient diff: {jnp.abs(ref_grad[3] - our_grad[3])}")
    print(f"Max mask gradient diff: {jnp.max(jnp.abs(ref_grad[3] - our_grad[3]))}")

    # Check gradient using check_grads on our reference function
    try:
        test_util.check_grads(
            our_reference_fn, (x, w1, w2, mask), order=1, modes=["rev"]
        )
        print("Our reference function gradient check: PASSED")
    except Exception as e:
        print(f"Our reference function gradient check: FAILED - {e}")
