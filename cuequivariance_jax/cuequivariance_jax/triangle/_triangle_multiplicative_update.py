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
    os.getenv("CUEQ_TRIMUL_FALLBACK_THRESHOLD", "100")
)


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
        x (jax.Array): Input tensor of shape (..., N, N, D_in) where:
            - ... represents arbitrary batch dimensions
            - N is the sequence length
            - D_in is the input hidden dimension
        direction (str): Direction of the triangular projection. Must be either "outgoing" or "incoming".
        key (jax.Array, optional): JAX random key for weight initialization. Required if any weights are None.
        mask (jax.Array, optional): Optional mask tensor of shape (..., N, N) for masking the output.
            Must be broadcastable with the input tensor's batch dimensions.
        norm_in_weight (jax.Array, optional): Weight tensor for input normalization of shape (D_in,).
            If None, initialized to ones.
        norm_in_bias (jax.Array, optional): Bias tensor for input normalization of shape (D_in,).
            If None, initialized to zeros.
        p_in_weight (jax.Array, optional): Weight tensor for input projection of shape (2*D_in, D_in).
            If None, initialized with LeCun normal distribution.
        p_in_bias (jax.Array, optional): Bias tensor for input projection of shape (2*D_in,).
            If None, no bias is applied to the input projection.
        g_in_weight (jax.Array, optional): Weight tensor for input gating of shape (2*D_in, D_in).
            If None, initialized with LeCun normal distribution.
        g_in_bias (jax.Array, optional): Bias tensor for input gating of shape (2*D_in,).
            If None, no bias is applied to the input gating.
        norm_out_weight (jax.Array, optional): Weight tensor for output normalization of shape (D_in,).
            If None, initialized to ones.
        norm_out_bias (jax.Array, optional): Bias tensor for output normalization of shape (D_in,).
            If None, initialized to zeros.
        p_out_weight (jax.Array, optional): Weight tensor for output projection of shape (D_out, D_in).
            If None, initialized with LeCun normal distribution.
        p_out_bias (jax.Array, optional): Bias tensor for output projection of shape (D_out,).
            If None, no bias is applied to the output projection.
        g_out_weight (jax.Array, optional): Weight tensor for output gating of shape (D_out, D_in).
            If None, initialized with LeCun normal distribution.
        g_out_bias (jax.Array, optional): Bias tensor for output gating of shape (D_out,).
            If None, no bias is applied to the output gating.
        eps (float): Small constant for numerical stability in normalization. Defaults to 1e-5.
        precision (Precision): Precision mode for matrix multiplications.
            Available options:
            - DEFAULT: Use default precision setting
            - TF32: Use TensorFloat-32 precision
            - TF32x3: Use TensorFloat-32 precision with 3x accumulation
            - IEEE: Use IEEE 754 precision

    Returns:
        jax.Array: Output tensor of shape (..., N, N, D_out) where D_out is determined by
                   the first dimension of g_out_weight. If g_out_weight is not provided,
                   D_out equals D_in (the input hidden dimension).

    Notes:
        - Unlike PyTorch, JAX arrays are immutable, so weight initialization returns new arrays
        - If output weights are not provided, they are initialized with D_out = D_in (preserving input dimension)
        - If weights are not provided, they are initialized with appropriate values, but in practice
          you should pass learned parameters
        - Supports arbitrary batch dimensions through broadcasting

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from cuequivariance_jax import triangle_multiplicative_update
        >>> # Create input tensor with arbitrary batch dimensions
        >>> key = jax.random.key(0)
        >>> key, subkey = jax.random.split(key)
        >>> batch_dim1, batch_dim2, seq_len, D_in = 2, 3, 128, 128
        >>> x = jax.random.normal(subkey, (batch_dim1, batch_dim2, seq_len, seq_len, D_in), dtype=jnp.float32)
        >>> # Create mask (1 for valid positions, 0 for masked)
        >>> mask = jnp.ones((batch_dim1, batch_dim2, seq_len, seq_len))
        >>> # Create weight parameters (in practice, these would be learned)
        >>> norm_in_weight = jnp.ones(D_in)
        >>> norm_in_bias = jnp.zeros(D_in)
        >>> # Optional bias parameters for projection and gating layers
        >>> p_in_bias = jnp.zeros(2 * D_in)  # Optional input projection bias
        >>> g_in_bias = jnp.zeros(2 * D_in)  # Optional input gating bias
        >>> p_out_bias = jnp.zeros(D_in)     # Optional output projection bias (would be D_out if dimension changes)
        >>> g_out_bias = jnp.zeros(D_in)     # Optional output gating bias (would be D_out if dimension changes)
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
        (2, 3, 128, 128, 128)
        >>> # Example with dimension change: input 128 -> output 256
        >>> g_out_weight_256 = jax.random.normal(jax.random.key(1), (256, 128))
        >>> p_out_weight_256 = jax.random.normal(jax.random.key(2), (256, 128))
        >>> key, subkey2 = jax.random.split(key)
        >>> output_256 = triangle_multiplicative_update(
        ...     x=x,
        ...     direction="outgoing",
        ...     key=subkey2,  # Key needed for other weight initialization
        ...     g_out_weight=g_out_weight_256,
        ...     p_out_weight=p_out_weight_256,
        ... )
        >>> print(output_256.shape)
        (2, 3, 128, 128, 256)

    .. note::
        This operation uses a custom CUDA kernel for performance. When using this function
        on multiple devices, manual sharding is required to achieve proper performance.
        Without explicit sharding, performance will be significantly degraded. See
        `JAX shard_map documentation <https://docs.jax.dev/en/latest/notebooks/shard_map.html>`_
        for details on manual parallelism.
    """
    # Input validation
    if direction not in ["outgoing", "incoming"]:
        raise ValueError("direction must be either 'outgoing' or 'incoming'")

    seq_len, seq_len_other, D_in = x.shape[-3:]
    assert seq_len == seq_len_other
    batch_dims = x.shape[:-3]

    if mask is not None:
        assert mask.shape[-2:] == (seq_len, seq_len)
        batch_dims = jnp.broadcast_shapes(batch_dims, mask.shape[:-2])
        x = jnp.broadcast_to(x, batch_dims + (seq_len, seq_len, D_in))
        mask = jnp.broadcast_to(mask, batch_dims + (seq_len, seq_len))

    batch_size = math.prod(batch_dims)
    x = x.reshape(batch_size, seq_len, seq_len, D_in)
    if mask is not None:
        mask = mask.reshape(batch_size, seq_len, seq_len)

    # Validate weight dimensions if provided
    if norm_in_weight is not None and norm_in_weight.shape != (D_in,):
        raise ValueError(
            f"norm_in_weight must have shape ({D_in},), got {norm_in_weight.shape}"
        )
    if norm_in_bias is not None and norm_in_bias.shape != (D_in,):
        raise ValueError(
            f"norm_in_bias must have shape ({D_in},), got {norm_in_bias.shape}"
        )
    if p_in_weight is not None and p_in_weight.shape != (2 * D_in, D_in):
        raise ValueError(
            f"p_in_weight must have shape ({2 * D_in}, {D_in}), got {p_in_weight.shape}"
        )
    if g_in_weight is not None and g_in_weight.shape != (2 * D_in, D_in):
        raise ValueError(
            f"g_in_weight must have shape ({2 * D_in}, {D_in}), got {g_in_weight.shape}"
        )
    if norm_out_weight is not None and norm_out_weight.shape != (D_in,):
        raise ValueError(
            f"norm_out_weight must have shape ({D_in},), got {norm_out_weight.shape}"
        )
    if norm_out_bias is not None and norm_out_bias.shape != (D_in,):
        raise ValueError(
            f"norm_out_bias must have shape ({D_in},), got {norm_out_bias.shape}"
        )
    if p_out_weight is not None and p_out_weight.shape[1] != D_in:
        raise ValueError(
            f"p_out_weight must have shape (output_dim, {D_in}), got {p_out_weight.shape}"
        )
    if g_out_weight is not None and g_out_weight.shape[1] != D_in:
        raise ValueError(
            f"g_out_weight must have shape (output_dim, {D_in}), got {g_out_weight.shape}"
        )
    if (
        p_out_weight is not None
        and g_out_weight is not None
        and p_out_weight.shape[0] != g_out_weight.shape[0]
    ):
        raise ValueError(
            f"p_out_weight and g_out_weight must have the same output dimension, got {p_out_weight.shape[0]} and {g_out_weight.shape[0]}"
        )

    # Validate bias dimensions if provided
    if p_in_bias is not None and p_in_bias.shape != (2 * D_in,):
        raise ValueError(
            f"p_in_bias must have shape ({2 * D_in},), got {p_in_bias.shape}"
        )
    if g_in_bias is not None and g_in_bias.shape != (2 * D_in,):
        raise ValueError(
            f"g_in_bias must have shape ({2 * D_in},), got {g_in_bias.shape}"
        )
    # Get output dimension for bias validation
    D_out = D_in  # default to input dimension
    if p_out_weight is not None:
        D_out = p_out_weight.shape[0]
    elif g_out_weight is not None:
        D_out = g_out_weight.shape[0]

    if p_out_bias is not None and p_out_bias.shape != (D_out,):
        raise ValueError(
            f"p_out_bias must have shape ({D_out},), got {p_out_bias.shape}"
        )
    if g_out_bias is not None and g_out_bias.shape != (D_out,):
        raise ValueError(
            f"g_out_bias must have shape ({D_out},), got {g_out_bias.shape}"
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
        norm_in_weight = bias_init_one(D_in, dtype=x.dtype)
    if norm_in_bias is None:
        norm_in_bias = bias_init_zero(D_in, dtype=x.dtype)
    if p_in_weight is None:
        p_in_weight = lecun_normal_init((2 * D_in, D_in), key_p_in, dtype=x.dtype)
    if g_in_weight is None:
        g_in_weight = lecun_normal_init((2 * D_in, D_in), key_g_in, dtype=x.dtype)
    if norm_out_weight is None:
        norm_out_weight = bias_init_one(D_in, dtype=x.dtype)
    if norm_out_bias is None:
        norm_out_bias = bias_init_zero(D_in, dtype=x.dtype)
    if p_out_weight is None:
        p_out_weight = lecun_normal_init((D_out, D_in), key_p_out, dtype=x.dtype)
    if g_out_weight is None:
        g_out_weight = lecun_normal_init((D_out, D_in), key_g_out, dtype=x.dtype)

    if fallback is None:
        fallback = seq_len <= CUEQ_TRIMUL_FALLBACK_THRESHOLD

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

    # Reshape back to original batch dimensions with output hidden dimension
    x = x.reshape(batch_dims + (seq_len, seq_len, D_out))

    return x
