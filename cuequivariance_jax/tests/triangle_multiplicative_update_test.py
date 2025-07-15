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


def create_weights(hidden_dim, seed=42, device=None):
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
def test_compare_with_pytorch(direction, use_mask):
    """Compare JAX and PyTorch implementations."""
    try:
        import torch
        from cuequivariance_ops_torch import (
            triangle_multiplicative_update as triangle_multiplicative_update_torch,
        )
        from cuequivariance_ops_torch.triangle_multiplicative_update import (
            Precision as PrecisionTorch,
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

    mask_torch = None
    mask_jax = None
    if use_mask:
        mask_np = np.random.rand(batch_size, seq_len, seq_len).astype(np.float32)
        mask_torch = torch.tensor(mask_np, dtype=torch.float32, device="cuda")
        mask_jax = jnp.array(mask_np)
    else:
        # PyTorch implementation doesn't handle mask=None properly, so provide a mask of ones
        # This should give equivalent results to no masking
        mask_torch = torch.ones(
            batch_size, seq_len, seq_len, dtype=torch.float32, device="cuda"
        )
        # For JAX, we keep mask_jax as None to test the proper handling

    # Create weights
    weights_torch = create_weights(hidden_dim, device="torch")
    weights_jax = create_weights(hidden_dim)

    # Run both versions
    with torch.no_grad():
        out_torch = triangle_multiplicative_update_torch(
            x_torch,
            direction=direction,
            mask=mask_torch,
            **weights_torch,
            eps=eps,
            precision=PrecisionTorch.DEFAULT,
        )

    out_jax = triangle_multiplicative_update_jax(
        x_jax,
        direction=direction,
        mask=mask_jax,
        **weights_jax,
        eps=eps,
        precision=Precision.DEFAULT,
    )

    # Compare outputs
    np.testing.assert_allclose(
        out_torch.cpu().numpy(),
        np.array(out_jax),
        rtol=5e-3,
        atol=5e-3,
        err_msg=f"Outputs differ for direction={direction}, use_mask={use_mask}",
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "x_shape,mask_shape,expected_shape",
    [
        ((8, 8, 64), None, (1, 8, 8, 64)),  # 3D input -> 4D output
        ((1, 8, 8, 64), None, (1, 8, 8, 64)),  # 4D input
        ((2, 8, 8, 64), (2, 8, 8), (2, 8, 8, 64)),  # With mask
        ((8, 8, 64), (8, 8), (1, 8, 8, 64)),  # 3D with 2D mask -> 4D output
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


def test_directions_produce_different_results():
    """Test that different directions produce different outputs."""
    batch_size, seq_len, hidden_dim = 1, 4, 64

    np.random.seed(42)
    x = jnp.array(
        np.random.randn(batch_size, seq_len, seq_len, hidden_dim).astype(np.float32)
    )
    weights = create_weights(hidden_dim)

    # Get outputs for both directions
    out_outgoing = triangle_multiplicative_update_jax(
        x, direction="outgoing", **weights
    )
    out_incoming = triangle_multiplicative_update_jax(
        x, direction="incoming", **weights
    )

    # They should produce different results
    assert not jnp.allclose(out_outgoing, out_incoming)


def test_initialization_with_key():
    """Test weight initialization with JAX random keys."""
    batch_size, seq_len, hidden_dim = 1, 4, 64

    key = jax.random.key(0)
    key_x, key_init1, key_init2 = jax.random.split(key, 3)
    x = jax.random.normal(
        key_x, (batch_size, seq_len, seq_len, hidden_dim), dtype=jnp.float32
    )

    # Should raise without key when weights are None
    with pytest.raises(ValueError, match="Random key is required"):
        triangle_multiplicative_update_jax(x, direction="outgoing")

    # Should work with keys
    output1 = triangle_multiplicative_update_jax(x, direction="outgoing", key=key_init1)
    output2 = triangle_multiplicative_update_jax(x, direction="outgoing", key=key_init2)

    assert output1.shape == (batch_size, seq_len, seq_len, hidden_dim)
    assert not jnp.allclose(output1, output2)  # Different keys -> different results


@pytest.mark.parametrize(
    "error_match,test_input",
    [
        (
            "direction must be either",
            {"direction": "invalid", "x_shape": (1, 8, 8, 64)},
        ),
        (
            "must be 4-dimensional",
            {"direction": "outgoing", "x_shape": (1, 2, 8, 8, 64)},
        ),
        (
            "mask must be 3-dimensional",
            {
                "direction": "outgoing",
                "x_shape": (1, 8, 8, 64),
                "mask_shape": (1, 2, 8, 8),
            },
        ),
    ],
)
def test_errors(error_match, test_input):
    """Test error handling."""
    x_shape = test_input["x_shape"]
    x = jnp.ones(x_shape)
    hidden_dim = x_shape[-1] if len(x_shape) <= 4 else 64
    weights = create_weights(hidden_dim)

    mask = None
    if "mask_shape" in test_input:
        mask = jnp.ones(test_input["mask_shape"])

    with pytest.raises(ValueError, match=error_match):
        triangle_multiplicative_update_jax(
            x, direction=test_input["direction"], mask=mask, **weights
        )


@pytest.mark.parametrize(
    "precision", [Precision.DEFAULT, Precision.TF32, Precision.TF32x3, Precision.IEEE]
)
def test_precision_modes(precision):
    """Test different precision modes."""
    batch_size, seq_len, hidden_dim = 1, 4, 64
    x = jax.random.normal(
        jax.random.key(0), (batch_size, seq_len, seq_len, hidden_dim), dtype=jnp.float32
    )
    weights = create_weights(hidden_dim)

    output = triangle_multiplicative_update_jax(
        x, direction="outgoing", precision=precision, **weights
    )

    assert output.shape == x.shape


@pytest.mark.slow
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_gradients(direction):
    """Test gradient computation and numerical gradient checking."""
    batch_size, seq_len, hidden_dim = 1, 4, 64
    x = jax.random.normal(
        jax.random.key(0), (batch_size, seq_len, seq_len, hidden_dim), dtype=jnp.float32
    )
    weights = create_weights(hidden_dim)

    def f_input(x):
        output = triangle_multiplicative_update_jax(x, direction=direction, **weights)
        return jnp.sum(output**2)

    jax.test_util.check_grads(
        f_input, (x,), order=1, eps=1e-2, modes="rev", atol=0.2, rtol=0.2
    )
