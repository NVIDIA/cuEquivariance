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

import math
import os

import jax
import jax.numpy as jnp

from cuequivariance_jax.triangle._layer_norm_transpose import (
    layer_norm_transpose,
)
from cuequivariance_jax.triangle._sigmoid_gated_dual_gemm import (
    sigmoid_gated_dual_gemm,
    sigmoid_gated_dual_gemm_dual_x,
)
from cuequivariance_jax.triangle._utils import Precision

CUEQ_TRIMUL_FALLBACK_THRESHOLD: int = int(
    os.getenv("CUEQ_TRIMUL_FALLBACK_THRESHOLD", "50")
)


def ensure_dims(ten: jax.Array, n: int) -> jax.Array:
    """Ensure tensor has at least n dimensions by adding 1-sized dimensions at the beginning.

    Args:
        ten: Input tensor
        n: Target number of dimensions

    Returns:
        Tensor with at least n dimensions
    """
    while len(ten.shape) < n:
        ten = jnp.expand_dims(ten, 0)
    return ten


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    """Calculate fan-in or fan-out for weight initialization.

    Args:
        linear_weight_shape: Shape tuple (fan_out, fan_in)
        fan: One of "fan_in", "fan_out", or "fan_avg"

    Returns:
        Calculated fan value
    """
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError(
            f"Invalid fan option: {fan}. Must be one of 'fan_in', 'fan_out', or 'fan_avg'"
        )
    return f


def trunc_normal_init(shape, key, scale=1.0, fan="fan_in", dtype=jnp.float32):
    """Initialize weights with truncated normal distribution.

    Args:
        shape: Shape of the weight tensor
        key: JAX random key
        scale: Scale factor for initialization
        fan: Fan mode ("fan_in", "fan_out", or "fan_avg")
        dtype: Data type for the weights

    Returns:
        Initialized weight tensor
    """
    if key is None:
        raise ValueError("Random key is required for weight initialization")

    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)

    # For truncated normal with bounds [-2, 2], the standard deviation is approximately 0.87962566
    # This is the standard deviation of a standard truncated normal distribution with these bounds
    truncnorm_std = 0.87962566

    # Calculate the desired standard deviation
    std = math.sqrt(scale)

    # Generate truncated normal samples and scale them
    samples = jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype=dtype)
    # Scale the samples to have the desired standard deviation
    samples = samples * (std / truncnorm_std)

    return samples


def lecun_normal_init(shape, key, dtype=jnp.float32):
    """LeCun normal initialization."""
    return trunc_normal_init(shape, key, scale=1.0, dtype=dtype)


def bias_init_zero(shape, dtype=jnp.float32):
    """Initialize bias with zeros."""
    return jnp.zeros(shape, dtype=dtype)


def bias_init_one(shape, dtype=jnp.float32):
    """Initialize bias with ones."""
    return jnp.ones(shape, dtype=dtype)


def triangle_multiplicative_update(
    x: jax.Array,
    direction: str = "outgoing",
    key: jax.Array | None = None,
    mask: jax.Array | None = None,
    norm_in_weight: jax.Array | None = None,
    norm_in_bias: jax.Array | None = None,
    p_in_weight: jax.Array | None = None,
    p_in_bias: jax.Array | None = None,
    g_in_weight: jax.Array | None = None,
    g_in_bias: jax.Array | None = None,
    norm_out_weight: jax.Array | None = None,
    norm_out_bias: jax.Array | None = None,
    p_out_weight: jax.Array | None = None,
    p_out_bias: jax.Array | None = None,
    g_out_weight: jax.Array | None = None,
    g_out_bias: jax.Array | None = None,
    eps: float = 1e-5,
    precision: Precision = Precision.DEFAULT,
    fallback: bool | None = None,
) -> jax.Array:
    """Apply triangle multiplicative update operation.

    This function performs a triangle multiplicative update operation, which is a key component
    in the AlphaFold2 architecture. The operation consists of:

    1. Input normalization and gating
    2. Triangular projection (either outgoing or incoming)
    3. Output normalization and gating

    Args:
        x (jax.Array): Input tensor of shape (B, N, N, D) where:
            - B is the batch size
            - N is the sequence length
            - D is the hidden dimension
            Can also be 3D (N, N, D) which will be expanded to 4D.
        direction (str): Direction of the triangular projection. Must be either "outgoing" or "incoming".
        key (jax.Array, optional): JAX random key for weight initialization. Required if any weights are None.
        mask (jax.Array, optional): Optional mask tensor of shape (B, N, N) for masking the output.
            Can also be 2D (N, N) which will be expanded to 3D.
        norm_in_weight (jax.Array, optional): Weight tensor for input normalization of shape (D,).
            If None, initialized to ones.
        norm_in_bias (jax.Array, optional): Bias tensor for input normalization of shape (D,).
            If None, initialized to zeros.
        p_in_weight (jax.Array, optional): Weight tensor for input projection of shape (2D, D).
            If None, initialized with LeCun normal distribution.
        p_in_bias (jax.Array, optional): Bias tensor for input projection of shape (2D,).
            If None, no bias is applied to the input projection.
        g_in_weight (jax.Array, optional): Weight tensor for input gating of shape (2D, D).
            If None, initialized with LeCun normal distribution.
        g_in_bias (jax.Array, optional): Bias tensor for input gating of shape (2D,).
            If None, no bias is applied to the input gating.
        norm_out_weight (jax.Array, optional): Weight tensor for output normalization of shape (D,).
            If None, initialized to ones.
        norm_out_bias (jax.Array, optional): Bias tensor for output normalization of shape (D,).
            If None, initialized to zeros.
        p_out_weight (jax.Array, optional): Weight tensor for output projection of shape (D, D).
            If None, initialized with LeCun normal distribution.
        p_out_bias (jax.Array, optional): Bias tensor for output projection of shape (D,).
            If None, no bias is applied to the output projection.
        g_out_weight (jax.Array, optional): Weight tensor for output gating of shape (D, D).
            If None, initialized with LeCun normal distribution.
        g_out_bias (jax.Array, optional): Bias tensor for output gating of shape (D,).
            If None, no bias is applied to the output gating.
        eps (float): Small constant for numerical stability in normalization. Defaults to 1e-5.
        precision (Precision): Precision mode for matrix multiplications.
            Available options:
            - DEFAULT: Use default precision setting
            - TF32: Use TensorFloat-32 precision
            - TF32x3: Use TensorFloat-32 precision with 3x accumulation
            - IEEE: Use IEEE 754 precision

    Returns:
        jax.Array: Output tensor of shape (B, N, N, D). Always returns 4D tensor even if input was 3D.

    Notes:
        - Unlike PyTorch, JAX arrays are immutable, so weight initialization returns new arrays
        - Hidden dimension D must be divisible by 64 for the BND_BND layout in layer normalization
        - If weights are not provided, they are initialized with appropriate values, but in practice
          you should pass learned parameters

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from cuequivariance_jax import triangle_multiplicative_update
        >>> # Create input tensor
        >>> key = jax.random.key(0)
        >>> key, subkey = jax.random.split(key)
        >>> batch_size, seq_len, hidden_dim = 1, 128, 128
        >>> x = jax.random.normal(subkey, (batch_size, seq_len, seq_len, hidden_dim), dtype=jnp.float32)
        >>> # Create mask (1 for valid positions, 0 for masked)
        >>> mask = jnp.ones((batch_size, seq_len, seq_len))
        >>> # Create weight parameters (in practice, these would be learned)
        >>> norm_in_weight = jnp.ones(hidden_dim)
        >>> norm_in_bias = jnp.zeros(hidden_dim)
        >>> # Optional bias parameters for projection and gating layers
        >>> p_in_bias = jnp.zeros(2 * hidden_dim)  # Optional input projection bias
        >>> g_in_bias = jnp.zeros(2 * hidden_dim)  # Optional input gating bias
        >>> p_out_bias = jnp.zeros(hidden_dim)     # Optional output projection bias
        >>> g_out_bias = jnp.zeros(hidden_dim)     # Optional output gating bias
        >>> # Initialize other weights using the key
        >>> key, subkey = jax.random.split(key)
        >>> # Perform triangular multiplication
        >>> output = triangle_multiplicative_update(
        ...     x=x,
        ...     direction="outgoing",  # or "incoming"
        ...     key=subkey,  # Only needed if some weights are None
        ...     mask=mask,
        ...     norm_in_weight=norm_in_weight,
        ...     norm_in_bias=norm_in_bias,
        ...     p_in_bias=p_in_bias,  # Can be None to skip bias
        ...     g_in_bias=g_in_bias,  # Can be None to skip bias
        ...     p_out_bias=p_out_bias,  # Can be None to skip bias
        ...     g_out_bias=g_out_bias,  # Can be None to skip bias
        ...     # ... pass other weights or let them initialize ...
        ... )
        >>> print(output.shape)
        (1, 128, 128, 128)
    """
    # Input validation
    if direction not in ["outgoing", "incoming"]:
        raise ValueError("direction must be either 'outgoing' or 'incoming'")

    # Ensure x has 4 dimensions
    x = ensure_dims(x, 4)
    if x.ndim != 4:
        raise ValueError(
            "x must be 4-dimensional (batch_size, seq_len, seq_len, hidden_dim) or lower dimensional where first dimensions with size 1 are omitted"
        )

    if mask is not None:
        # Ensure mask has 3 dimensions
        mask = ensure_dims(mask, 3)
        if mask.ndim != 3:
            raise ValueError(
                "mask must be 3-dimensional (batch_size, seq_len, seq_len) or lower dimensional where first dimensions with size 1 are omitted"
            )

    # Initialize default weights if not provided
    hidden_dim = x.shape[-1]

    # Check hidden dimension constraint for BND_BND layout
    if hidden_dim % 32 != 0:
        raise ValueError(
            f"Hidden dimension must be divisible by 32 for BND_BND layout in layer normalization. "
            f"Got hidden_dim={hidden_dim}"
        )

    # Validate weight dimensions if provided
    if norm_in_weight is not None and norm_in_weight.shape != (hidden_dim,):
        raise ValueError(
            f"norm_in_weight must have shape ({hidden_dim},), got {norm_in_weight.shape}"
        )
    if norm_in_bias is not None and norm_in_bias.shape != (hidden_dim,):
        raise ValueError(
            f"norm_in_bias must have shape ({hidden_dim},), got {norm_in_bias.shape}"
        )
    if p_in_weight is not None and p_in_weight.shape != (2 * hidden_dim, hidden_dim):
        raise ValueError(
            f"p_in_weight must have shape ({2 * hidden_dim}, {hidden_dim}), got {p_in_weight.shape}"
        )
    if g_in_weight is not None and g_in_weight.shape != (2 * hidden_dim, hidden_dim):
        raise ValueError(
            f"g_in_weight must have shape ({2 * hidden_dim}, {hidden_dim}), got {g_in_weight.shape}"
        )
    if norm_out_weight is not None and norm_out_weight.shape != (hidden_dim,):
        raise ValueError(
            f"norm_out_weight must have shape ({hidden_dim},), got {norm_out_weight.shape}"
        )
    if norm_out_bias is not None and norm_out_bias.shape != (hidden_dim,):
        raise ValueError(
            f"norm_out_bias must have shape ({hidden_dim},), got {norm_out_bias.shape}"
        )
    if p_out_weight is not None and p_out_weight.shape != (hidden_dim, hidden_dim):
        raise ValueError(
            f"p_out_weight must have shape ({hidden_dim}, {hidden_dim}), got {p_out_weight.shape}"
        )
    if g_out_weight is not None and g_out_weight.shape != (hidden_dim, hidden_dim):
        raise ValueError(
            f"g_out_weight must have shape ({hidden_dim}, {hidden_dim}), got {g_out_weight.shape}"
        )

    # Validate bias dimensions if provided
    if p_in_bias is not None and p_in_bias.shape != (2 * hidden_dim,):
        raise ValueError(
            f"p_in_bias must have shape ({2 * hidden_dim},), got {p_in_bias.shape}"
        )
    if g_in_bias is not None and g_in_bias.shape != (2 * hidden_dim,):
        raise ValueError(
            f"g_in_bias must have shape ({2 * hidden_dim},), got {g_in_bias.shape}"
        )
    if p_out_bias is not None and p_out_bias.shape != (hidden_dim,):
        raise ValueError(
            f"p_out_bias must have shape ({hidden_dim},), got {p_out_bias.shape}"
        )
    if g_out_bias is not None and g_out_bias.shape != (hidden_dim,):
        raise ValueError(
            f"g_out_bias must have shape ({hidden_dim},), got {g_out_bias.shape}"
        )

    # If we need to initialize weights and no key is provided, raise an error
    needs_init = (
        p_in_weight is None
        or g_in_weight is None
        or p_out_weight is None
        or g_out_weight is None
    )
    if needs_init and key is None:
        raise ValueError("Random key is required for weight initialization")

    # Split keys for each weight initialization if needed
    if needs_init:
        keys = jax.random.split(key, 4)
        key_p_in, key_g_in, key_p_out, key_g_out = keys

    if norm_in_weight is None:
        norm_in_weight = bias_init_one(hidden_dim, dtype=x.dtype)
    if norm_in_bias is None:
        norm_in_bias = bias_init_zero(hidden_dim, dtype=x.dtype)
    if p_in_weight is None:
        p_in_weight = lecun_normal_init(
            (2 * hidden_dim, hidden_dim), key_p_in, dtype=x.dtype
        )
    if g_in_weight is None:
        g_in_weight = lecun_normal_init(
            (2 * hidden_dim, hidden_dim), key_g_in, dtype=x.dtype
        )
    if norm_out_weight is None:
        norm_out_weight = bias_init_one(hidden_dim, dtype=x.dtype)
    if norm_out_bias is None:
        norm_out_bias = bias_init_zero(hidden_dim, dtype=x.dtype)
    if p_out_weight is None:
        p_out_weight = lecun_normal_init(
            (hidden_dim, hidden_dim), key_p_out, dtype=x.dtype
        )
    if g_out_weight is None:
        g_out_weight = lecun_normal_init(
            (hidden_dim, hidden_dim), key_g_out, dtype=x.dtype
        )

    if fallback is None:
        fallback = x.shape[-2] <= CUEQ_TRIMUL_FALLBACK_THRESHOLD

    # Input normalization
    x = layer_norm_transpose(
        x, norm_in_weight, norm_in_bias, eps=eps, layout="bijd->bijd", fallback=fallback
    )
    x_in = x

    # Gated dual gemm
    ab = sigmoid_gated_dual_gemm(
        x,
        g_in_weight,
        p_in_weight,
        b1=g_in_bias,
        b2=p_in_bias,
        mask=mask,
        transpose_out=True,
        precision=precision,
        fallback=fallback,
    )
    a, b = jnp.split(ab, 2, axis=0)

    # Triangular projection
    if direction == "outgoing":
        x = jnp.einsum("dbik,dbjk->dbij", a, b)
    else:
        x = jnp.einsum("dbki,dbkj->dbij", a, b)

    # Output normalization
    x_out = layer_norm_transpose(
        x,
        norm_out_weight,
        norm_out_bias,
        eps=eps,
        layout="dbij->bijd",
        fallback=fallback,
    )

    # Output gating
    x = sigmoid_gated_dual_gemm_dual_x(
        x_in,
        x_out,
        g_out_weight,
        p_out_weight,
        b1=g_out_bias,
        b2=p_out_bias,
        precision=precision,
        fallback=fallback,
    )

    return x
