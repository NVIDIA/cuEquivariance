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
import jax.test_util
import numpy as np
import pytest
from cuequivariance_jax.triangle import (
    Precision,
)
from cuequivariance_jax.triangle import (
    triangle_multiplicative_update as triangle_multiplicative_update_jax,
)

jax.config.update("jax_enable_x64", True)


def create_weights(hidden_dim, seed=42, device=None, include_bias=False):
    """Helper function to create test weights."""
    np.random.seed(seed)

    weights_np = {
        "norm_in_weight": np.ones(hidden_dim),
        "norm_in_bias": np.zeros(hidden_dim),
        "norm_out_weight": np.ones(hidden_dim),
        "norm_out_bias": np.zeros(hidden_dim),
        "p_in_weight": np.random.randn(2 * hidden_dim, hidden_dim) * 0.1,
        "g_in_weight": np.random.randn(2 * hidden_dim, hidden_dim) * 0.1,
        "p_out_weight": np.random.randn(hidden_dim, hidden_dim) * 0.1,
        "g_out_weight": np.random.randn(hidden_dim, hidden_dim) * 0.1,
    }

    if include_bias:
        weights_np.update(
            {
                "p_in_bias": np.random.randn(2 * hidden_dim) * 0.1,
                "g_in_bias": np.random.randn(2 * hidden_dim) * 0.1,
                "p_out_bias": np.random.randn(hidden_dim) * 0.1,
                "g_out_bias": np.random.randn(hidden_dim) * 0.1,
            }
        )

    if device == "torch":
        import torch

        return {
            k: torch.tensor(v, dtype=torch.float32, device="cuda")
            for k, v in weights_np.items()
        }
    else:
        return {k: jnp.array(v, jnp.float32) for k, v in weights_np.items()}


@pytest.mark.slow
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("include_bias", [False, True])
def test_compare_with_pytorch(direction, use_mask, include_bias):
    """Compare JAX and PyTorch implementations with and without bias."""
    pytest.skip("Hard to get JAX and PyTorch to run in the same environment.")
    try:
        import torch
        from cuequivariance_ops_torch import (
            triangle_multiplicative_update as triangle_multiplicative_update_torch,
        )
    except ImportError:
        pytest.skip("torch or cuequivariance_ops_torch not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size, seq_len, hidden_dim = 1, 128, 64
    eps = 1e-5

    # Create test inputs
    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, seq_len, hidden_dim).astype(np.float32)
    x_torch = torch.tensor(x_np, dtype=torch.float32, device="cuda")
    x_jax = jnp.array(x_np)

    mask_torch = mask_jax = None
    if use_mask:
        mask_np = np.random.rand(batch_size, seq_len, seq_len).astype(np.float32)
        mask_torch = torch.tensor(mask_np, dtype=torch.float32, device="cuda")
        mask_jax = jnp.array(mask_np)

    # Create weights
    weights_torch = create_weights(
        hidden_dim, device="torch", include_bias=include_bias
    )
    weights_jax = create_weights(hidden_dim, include_bias=include_bias)

    # Run both versions
    with torch.no_grad():
        out_torch = triangle_multiplicative_update_torch(
            x_torch,
            direction=direction,
            mask=mask_torch,
            **weights_torch,
            eps=eps,
            precision="IEEE",
        )

    out_jax = triangle_multiplicative_update_jax(
        x_jax,
        direction=direction,
        mask=mask_jax,
        **weights_jax,
        eps=eps,
        precision=Precision.IEEE,
    )

    # Compare outputs
    np.testing.assert_allclose(
        out_torch.cpu().numpy(), np.array(out_jax), rtol=5e-3, atol=5e-3
    )


@pytest.mark.parametrize(
    "x_shape,mask_shape,expected_shape",
    [
        ((8, 8, 64), None, (8, 8, 64)),
        ((1, 8, 8, 64), None, (1, 8, 8, 64)),
        ((2, 8, 8, 64), (2, 8, 8), (2, 8, 8, 64)),
        ((2, 1, 4, 8, 8, 64), (3, 1, 8, 8), (2, 3, 4, 8, 8, 64)),  # broadcast
    ],
)
def test_shapes(x_shape, mask_shape, expected_shape):
    """Test different input and output shapes."""
    x = jnp.ones(x_shape, dtype=jnp.float32)
    mask = jnp.ones(mask_shape) if mask_shape else None
    weights = create_weights(x_shape[-1])

    output = triangle_multiplicative_update_jax(
        x, direction="outgoing", mask=mask, **weights
    )
    assert output.shape == expected_shape


def test_basic_functionality():
    """Test basic functionality, directions, and weight initialization."""
    batch_size, seq_len, hidden_dim = 1, 4, 64
    key = jax.random.key(0)
    key_x, key_init1, key_init2 = jax.random.split(key, 3)

    x = jax.random.normal(
        key_x, (batch_size, seq_len, seq_len, hidden_dim), dtype=jnp.float32
    )
    weights = create_weights(hidden_dim)

    # Test different directions produce different results
    out_outgoing = triangle_multiplicative_update_jax(
        x, direction="outgoing", **weights
    )
    out_incoming = triangle_multiplicative_update_jax(
        x, direction="incoming", **weights
    )
    assert not jnp.allclose(out_outgoing, out_incoming)

    # Test weight initialization with keys
    with pytest.raises(ValueError, match="Random key is required"):
        triangle_multiplicative_update_jax(x, direction="outgoing")

    output1 = triangle_multiplicative_update_jax(x, direction="outgoing", key=key_init1)
    output2 = triangle_multiplicative_update_jax(x, direction="outgoing", key=key_init2)
    assert output1.shape == x.shape
    assert not jnp.allclose(output1, output2)  # Different keys -> different results


@pytest.mark.parametrize(
    "error_match,test_input",
    [
        (
            "direction must be either",
            {"direction": "invalid", "x_shape": (1, 8, 8, 64)},
        )
    ],
)
def test_errors(error_match, test_input):
    """Test error handling."""
    x_shape = test_input["x_shape"]
    x = jnp.ones(x_shape)
    hidden_dim = x_shape[-1] if len(x_shape) <= 4 else 64
    weights = create_weights(hidden_dim)
    mask = jnp.ones(test_input["mask_shape"]) if "mask_shape" in test_input else None

    with pytest.raises(ValueError, match=error_match):
        triangle_multiplicative_update_jax(
            x, direction=test_input["direction"], mask=mask, **weights
        )


@pytest.mark.parametrize(
    "precision", [Precision.DEFAULT, Precision.TF32, Precision.IEEE]
)
def test_precision_modes(precision):
    """Test different precision modes."""
    x = jax.random.normal(jax.random.key(0), (1, 4, 4, 64), dtype=jnp.float32)
    weights = create_weights(64)
    output = triangle_multiplicative_update_jax(
        x, direction="outgoing", precision=precision, **weights
    )
    assert output.shape == x.shape


def test_bias_functionality():
    """Test bias parameters functionality."""
    x = jax.random.normal(jax.random.key(0), (1, 4, 4, 64), dtype=jnp.float32)
    weights_no_bias = create_weights(64)
    weights_with_bias = create_weights(64, include_bias=True)

    # Test with and without bias
    output_no_bias = triangle_multiplicative_update_jax(
        x, direction="outgoing", **weights_no_bias
    )
    output_with_bias = triangle_multiplicative_update_jax(
        x, direction="outgoing", **weights_with_bias
    )
    assert not jnp.allclose(output_no_bias, output_with_bias, atol=1e-6)

    # Test None bias parameters
    output_none = triangle_multiplicative_update_jax(
        x,
        direction="outgoing",
        p_in_bias=None,
        g_in_bias=None,
        p_out_bias=None,
        g_out_bias=None,
        **weights_no_bias,
    )
    assert output_none.shape == x.shape

    # Test mixed bias
    output_mixed = triangle_multiplicative_update_jax(
        x,
        direction="outgoing",
        p_in_bias=weights_with_bias["p_in_bias"],
        g_out_bias=weights_with_bias["g_out_bias"],
        **weights_no_bias,
    )
    assert not jnp.allclose(output_no_bias, output_mixed, atol=1e-6)

    # Test only output bias (input biases set to None)
    output_only_out_bias = triangle_multiplicative_update_jax(
        x,
        direction="outgoing",
        p_in_bias=None,
        g_in_bias=None,
        p_out_bias=weights_with_bias["p_out_bias"],
        g_out_bias=weights_with_bias["g_out_bias"],
        **weights_no_bias,
    )
    assert not jnp.allclose(output_no_bias, output_only_out_bias, atol=1e-6)
    assert not jnp.allclose(output_with_bias, output_only_out_bias, atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("include_bias", [False, True])
def test_gradients(direction, include_bias):
    """Test gradient computation with and without bias."""
    x = jax.random.normal(jax.random.key(0), (1, 4, 4, 64), dtype=jnp.float32)
    weights = create_weights(64, include_bias=include_bias)

    def f_input(x):
        output = triangle_multiplicative_update_jax(x, direction=direction, **weights)
        return jnp.sum(output**2)

    jax.test_util.check_grads(
        f_input, (x,), order=1, eps=1e-2, modes="rev", atol=0.2, rtol=0.2
    )

    # Test bias gradients if available
    if include_bias:

        def f_bias_only(p_in_bias, g_in_bias, p_out_bias, g_out_bias):
            weights_no_bias = {
                k: v
                for k, v in weights.items()
                if not k.endswith("_bias") or k in ["norm_in_bias", "norm_out_bias"]
            }
            output = triangle_multiplicative_update_jax(
                x,
                direction=direction,
                p_in_bias=p_in_bias,
                g_in_bias=g_in_bias,
                p_out_bias=p_out_bias,
                g_out_bias=g_out_bias,
                **weights_no_bias,
            )
            return jnp.sum(output**2)

        jax.test_util.check_grads(
            f_bias_only,
            (
                weights["p_in_bias"],
                weights["g_in_bias"],
                weights["p_out_bias"],
                weights["g_out_bias"],
            ),
            order=1,
            eps=1e-2,
            modes="rev",
            atol=0.2,
            rtol=0.2,
        )


@pytest.mark.parametrize(
    "dtype,direction", [(jnp.float16, "outgoing"), (jnp.float32, "incoming")]
)
def test_mixed_precision(dtype, direction):
    """Test with different precisions."""
    x = jax.random.normal(jax.random.key(42), (1, 32, 32, 64), dtype=dtype)
    weights = create_weights(64, include_bias=True)
    weights_typed = {k: v.astype(dtype) for k, v in weights.items()}

    output = triangle_multiplicative_update_jax(
        x, direction=direction, **weights_typed, precision=Precision.IEEE
    )
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert jnp.isfinite(output).all()


def test_fallback_and_optimized_paths():
    """Test automatic fallback vs optimized path selection."""
    key = jax.random.key(42)
    weights = create_weights(64, include_bias=True)

    # Small sequence (fallback)
    x_small = jax.random.normal(key, (1, 32, 32, 64), dtype=jnp.float32)
    output_small = triangle_multiplicative_update_jax(
        x_small, direction="outgoing", **weights
    )
    assert output_small.shape == x_small.shape
    assert jnp.isfinite(output_small).all()

    # Large sequence (optimized)
    key, subkey = jax.random.split(key)
    x_large = jax.random.normal(subkey, (1, 200, 200, 64), dtype=jnp.float32)
    mask_large = jnp.ones((1, 200, 200))

    output_large = triangle_multiplicative_update_jax(
        x_large,
        direction="outgoing",
        mask=mask_large,
        **weights,
        precision=Precision.IEEE,
    )
    assert output_large.shape == x_large.shape
    assert jnp.isfinite(output_large).all()


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_jit_compilation(direction):
    """Test JAX JIT compilation."""
    x = jax.random.normal(jax.random.key(42), (1, 32, 32, 64), dtype=jnp.float32)
    weights = create_weights(64, include_bias=True)

    @jax.jit
    def jit_triangle_mul(x, **kwargs):
        return triangle_multiplicative_update_jax(x, direction=direction, **kwargs)

    output_jit = jit_triangle_mul(x, **weights)
    output_no_jit = triangle_multiplicative_update_jax(
        x, direction=direction, **weights
    )

    np.testing.assert_allclose(output_jit, output_no_jit, rtol=1e-3, atol=1e-3)
    assert output_jit.shape == x.shape
    assert jnp.isfinite(output_jit).all()


def test_edge_cases_and_memory():
    """Test edge cases, accuracy tolerance, and memory efficiency."""
    key = jax.random.key(42)
    hidden_dim = 64
    weights = create_weights(hidden_dim, include_bias=True)

    # Test different value scales
    for scale in [1e-3, 1.0, 3.0]:
        x = jax.random.normal(key, (1, 16, 16, hidden_dim), dtype=jnp.float32) * scale
        output = triangle_multiplicative_update_jax(x, direction="outgoing", **weights)
        assert jnp.isfinite(output).all()

    # Test memory efficiency - multiple runs should give identical results
    x = jax.random.normal(key, (1, 32, 32, hidden_dim), dtype=jnp.float32)
    outputs = [
        triangle_multiplicative_update_jax(
            x, direction="outgoing", **weights, precision=Precision.IEEE
        )
        for _ in range(3)
    ]

    for output in outputs[1:]:
        np.testing.assert_allclose(outputs[0], output, rtol=1e-6, atol=1e-6)

    # Test different batch sizes
    for batch in [1, 2]:
        key, subkey = jax.random.split(key)
        x_batch = jax.random.normal(
            subkey, (batch, 32, 32, hidden_dim), dtype=jnp.float32
        )
        output_batch = triangle_multiplicative_update_jax(
            x_batch, direction="outgoing", **weights
        )
        assert output_batch.shape == x_batch.shape
