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
from typing import Optional, Tuple

import torch

try:
    from cuequivariance_ops_torch import TriMulPrecision
except ImportError:
    import enum

    class TriMulPrecision(enum.IntEnum):  # type: ignore
        """Fallback precision enum when cuequivariance_ops_torch is not available."""

        NONE = -1
        DEFAULT = 0
        TF32 = 1
        TF32x3 = 2
        IEEE = 3


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_aux: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Triangle Attention

    .. math::

        \text{Attention}_q(Q, K, V, B, M) = \sum_k\left[\text{softmax}_k\left(\begin{cases} 
        s\, Q_q \cdot K_k + B_{qk} & \text{if } M_k = 1 \\
        -10^9 & \text{otherwise}
        \end{cases}\right) V_k \right]


    Args:
        q (torch.Tensor): Query tensor of shape (B, N, H, Q, D). For B=1, can also be (N, H, Q, D).
        k (torch.Tensor): Key tensor of shape (B, N, H, K, D). For B=1, can also be (N, H, K, D).
        v (torch.Tensor): Value tensor of shape (B, N, H, K, D). For B=1, can also be (N, H, K, D).
        bias (torch.Tensor): Bias tensor of shape (B, 1, H, Q, K), For B=1, can also be (1, H, Q, K).
            Will be cast to float32 internally.
        mask (torch.Tensor, optional): Mask tensor of shape (B, N, 1, 1, K). For B=1, can also be (N, 1, 1, K).
            Will be cast to bool internally.
        scale (float, optional): Float scale for q (s in the equation). If None, value 1/sqrt(d) is used.
        return_aux (bool): If True, two auxiliary tensors are returned along with the result.
            Defaults to False.

    Note:
        - B: batch size
        - N: number of tokens
        - H: number of heads
        - Q: number of query tokens
        - K: number of key tokens
        - D: attention dimension

    Returns:
        - output(torch.Tensor): Output tensor of shape (B, N, H, Q, D). dtype=q.dtype
        - lse(torch.Tensor): Auxiliary result (for special use only). dtype=float32
        - max(torch.Tensor): Auxiliary result (for special use only). dtype=float32

    Notes:
        (1) Context is saved for backward pass. You don't need to save it manually.
        (2) Kernel precision (fp32, bf16, fp16) is based on input dtypes. For tf32, set it from torch global scope
        (3) Triangle attention kernel supports: all hidden_dim<=32 and divisible by 4 for tf32/fp32, and for all hidden_dim<=128 and divisible by 8 for bf16/fp16. In the rare instance that the kernel does not support an input config, fallback to torch is enabled instead of erroring out.
        (4) Blackwell-optimized kernels (for compute capabilities 10.0 and 10.3) provide superior performance especially for long sequences and higher head dimensions. These kernels require the sequence length N to be a multiple of 8 for the forward pass; pad the sequence if necessary. Currently, this feature is supported only for cu13 builds.

    Example:
        >>> import torch
        >>> import math
        >>> from cuequivariance_torch import triangle_attention
        >>> if torch.cuda.is_available():  # doctest: +SKIP
        ...     device = torch.device("cuda")
        ...     # Set up dimensions
        ...     batch_size, seq_len, num_heads, hidden_dim = 1, 128, 2, 32
        ...     # Create input tensors on GPU with float16 precision
        ...     q = torch.randn(batch_size, seq_len, num_heads, seq_len, hidden_dim,
        ...                     device=device, dtype=torch.float16, requires_grad=True)
        ...     k = torch.randn(batch_size, seq_len, num_heads, seq_len, hidden_dim,
        ...                     device=device, dtype=torch.float16, requires_grad=True)
        ...     v = torch.randn(batch_size, seq_len, num_heads, seq_len, hidden_dim,
        ...                     device=device, dtype=torch.float16, requires_grad=True)
        ...     bias = torch.randn(batch_size, 1, num_heads, seq_len, seq_len,
        ...                        device=device, dtype=torch.float32, requires_grad=True)
        ...     # Create optional mask
        ...     mask = torch.rand(batch_size, seq_len, 1, 1, seq_len,
        ...                       device=device) < 0.5
        ...     # Calculate scale
        ...     scale = 1 / math.sqrt(hidden_dim)
        ...     # Forward pass
        ...     output, lse, max_val = triangle_attention(
        ...         q=q, k=k, v=v, bias=bias, mask=mask, scale=scale, return_aux=True)
        ...     print(output.shape)  # torch.Size([1, 128, 2, 128, 32])
        ...     # Create gradient tensor and perform backward pass
        ...     grad_out = torch.randn_like(output)
        ...     output.backward(grad_out)
        ...     # Access gradients
        ...     print(q.grad.shape)  # torch.Size([1, 128, 2, 128, 32])
        ...     print(k.grad.shape)  # torch.Size([1, 128, 2, 128, 32])
        ...     print(v.grad.shape)  # torch.Size([1, 128, 2, 128, 32])
        ...     print(bias.grad.shape)  # torch.Size([1, 1, 2, 128, 128])
        torch.Size([1, 128, 2, 128, 32])
        torch.Size([1, 128, 2, 128, 32])
        torch.Size([1, 128, 2, 128, 32])
        torch.Size([1, 128, 2, 128, 32])
        torch.Size([1, 1, 2, 128, 128])
    """

    try:
        from cuequivariance_ops_torch import triangle_attention as f
    except Exception:
        raise ImportError(
            "Error importing triangle_attention from cuequivariance_ops_torch."
        )
    else:
        return f(q, k, v, bias, mask, scale, return_aux)


def triangle_multiplicative_update(
    x: torch.Tensor,
    direction: str = "outgoing",
    mask: Optional[torch.Tensor] = None,
    norm_in_weight: Optional[torch.Tensor] = None,
    norm_in_bias: Optional[torch.Tensor] = None,
    p_in_weight: Optional[torch.Tensor] = None,
    p_in_bias: Optional[torch.Tensor] = None,
    g_in_weight: Optional[torch.Tensor] = None,
    g_in_bias: Optional[torch.Tensor] = None,
    norm_out_weight: Optional[torch.Tensor] = None,
    norm_out_bias: Optional[torch.Tensor] = None,
    p_out_weight: Optional[torch.Tensor] = None,
    p_out_bias: Optional[torch.Tensor] = None,
    g_out_weight: Optional[torch.Tensor] = None,
    g_out_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    precision: Optional[TriMulPrecision] = None,
) -> torch.Tensor:
    """Apply triangle multiplicative update operation.

    This function performs a triangle multiplicative update operation, which is a key component
    in the AlphaFold2 architecture. The operation consists of:

    1. Input normalization and gating
    2. Triangular projection (either outgoing or incoming)
    3. Output normalization and gating

    The function supports both ahead-of-time (AOT) tuning and just-in-time (JIT) tuning.
    Auto-tuning behavior can be controlled through environment variables:

    - Quick testing: Default configuration where tuning configs, if existent, are looked-up. If not, then falls back to default kernel parameters. No tuning is performed.
    - On-Demand tuning: Set `CUEQ_TRITON_TUNING= "ONDEMAND"` to auto-tune for new shapes encountered on first run (may take several minutes)
    - AOT tuning: Set `CUEQ_TRITON_TUNING= "AOT"` to perform full ahead-of-time tuning for optimal performance **(may take several hours)**
    - Ignore user cache: Set CUEQ_TRITON_IGNORE_EXISTING_CACHE to ignore both the default settings that come with the package and any user-local settings previously saved with AOT/ONDEMAND tuning. May be used to regenerate optimal settings for a particular setup.
    - Cache directory: Set `CUEQ_TRITON_CACHE_DIR` to specify where tuning configurations are stored
    - Note: When using Docker with default or on-demand tuning enabled, commit the container to persist tuning changes

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, N, D) where:
            B is the batch size
            N is the sequence length
            D is the hidden dimension
        direction (str): Direction of the triangular projection. Must be either "outgoing" or "incoming".
        mask (torch.Tensor): Optional Mask tensor of shape (B, N, N) for masking the output.
        norm_in_weight (torch.Tensor): Optional weight tensor for input normalization of shape (D,).
        norm_in_bias (torch.Tensor): Optional bias tensor for input normalization of shape (D,).
        p_in_weight (torch.Tensor): Optional weight tensor for input projection of shape (2D, D).
        p_in_bias (torch.Tensor): Optional bias tensor for input projection of shape (2D,).
        g_in_weight (torch.Tensor): Optional weight tensor for input gating of shape (2D, D).
        g_in_bias (torch.Tensor): Optional bias tensor for input gating of shape (2D,).
        norm_out_weight (torch.Tensor): Optional weight tensor for output normalization of shape (D,).
        norm_out_bias (torch.Tensor): Optional bias tensor for output normalization of shape (D,).
        p_out_weight (torch.Tensor): Optional weight tensor for output projection of shape (D, D).
        p_out_bias (torch.Tensor): Optional bias tensor for output projection of shape (D,).
        g_out_weight (torch.Tensor): Optional weight tensor for output gating of shape (D, D).
        g_out_bias (torch.Tensor): Optional bias tensor for output gating of shape (D,).
        eps (float, optional): Small constant for numerical stability in normalization. Defaults to 1e-5.
        precision (TriMulPrecision, optional): Precision mode for matrix multiplications.
            Available options:
            - None: Defaults to triton language dot's default for non-32b input and for 32b input, tf32/tf32x3 based on 1/0 value set in torch.backends.cuda.matmul.allow_tf32
            - IEEE: Use IEEE 754 precision

    Returns:
        Output tensor of shape (batch_size, seq_len, seq_len, hidden_dim)

    Notes:
        (1) Context is saved for backward pass. You don't need to save it manually.
        (2) Kernel precision (fp32, bf16, fp16) is based on input dtypes. For tf32, set it from torch global scope using torch.backends.cuda.matmul.allow_tf32
        (3) **Limitation**: Currently only supports hidden_dim values that are multiples of 32.
        (4) We have moved away from the default round-towards-zero (RZ) implementation to round-nearest (RN) for better tf32 accuracy in cuex.triangle_multiplicative_update. In rare circumstances, this may cause minor differences in results observed.
        (5) When using torch compile, use `cueuivariance_ops_torch.init_triton_cache()` to initialize triton cache before calling torch compiled triangular multiplicative update.
        (6) Although the example demonstrates the most common case of one batch dimension, the API supports variable number of leading batch dimensions.

    Example:
        >>> import torch
        >>> from cuequivariance_torch import triangle_multiplicative_update
        >>> if torch.cuda.is_available():  # doctest: +SKIP
        ...     device = torch.device("cuda")
        ...     batch_size, seq_len, hidden_dim = 1, 128, 128
        ...     # Create input tensor
        ...     x = torch.randn(batch_size, seq_len, seq_len, hidden_dim, requires_grad=True, device=device)
        ...     # Create mask (1 for valid positions, 0 for masked)
        ...     mask = torch.ones(batch_size, seq_len, seq_len, device=device)
        ...     # Perform triangular multiplication
        ...     output = triangle_multiplicative_update(
        ...         x=x,
        ...         direction="outgoing",  # or "incoming"
        ...         mask=mask,
        ...     )
        ...     print(output.shape)  # torch.Size([1, 128, 128, 128])
        ...     # Create gradient tensor and perform backward pass
        ...     grad_out = torch.randn_like(output)
        ...     output.backward(grad_out)
        ...     # Access gradients
        ...     print(x.grad.shape)  # torch.Size([1, 128, 128, 128])
        torch.Size([1, 128, 128, 128])
        torch.Size([1, 128, 128, 128])
    """
    try:
        from cuequivariance_ops_torch import triangle_multiplicative_update as f
    except Exception:
        raise ImportError(
            "Error importing triangle_multiplicative_update from cuequivariance_ops_torch."
        )
    else:
        return f(
            x,
            direction,
            mask=mask,
            norm_in_weight=norm_in_weight,
            norm_in_bias=norm_in_bias,
            p_in_weight=p_in_weight,
            p_in_bias=p_in_bias,
            g_in_weight=g_in_weight,
            g_in_bias=g_in_bias,
            norm_out_weight=norm_out_weight,
            norm_out_bias=norm_out_bias,
            p_out_weight=p_out_weight,
            p_out_bias=p_out_bias,
            g_out_weight=g_out_weight,
            g_out_bias=g_out_bias,
            eps=eps,
            precision=precision,
        )


def attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    w_proj_z: Optional[torch.Tensor],
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    w_ln_z: Optional[torch.Tensor] = None,
    b_ln_z: Optional[torch.Tensor] = None,
    b_proj_z: Optional[torch.Tensor] = None,
    b_proj_g: Optional[torch.Tensor] = None,
    b_proj_o: Optional[torch.Tensor] = None,
    inf: float = 1e6,
    eps: float = 1e-5,
    attn_scale: Optional[float] = None,
    return_z_proj: bool = True,
    is_cached_z_proj: bool = False,
):
    """Compute attention with pairwise bias for diffusion models.

    This function implements attention with pairwise bias, which is commonly used
    in diffusion models. The function automatically chooses between optimized
    Triton kernels (for long sequences) and PyTorch fallback (for short sequences)
    based on sequence length.

    Args:
        s: Input sequence tensor of shape (B * M, S, D) where B is batch size,
            M is multiplicity (diffusion steps), S is sequence length, and D is
            feature dimension.
        q: Query tensor of shape (B * M, H, U, DH) where H is number of heads,
            U is query sequence length, and DH is head dimension.
        k: Key tensor of shape (B * M, H, V, DH) where V is key sequence length.
        v: Value tensor of shape (B * M, H, V, DH).
        z: Pairwise tensor of shape (B, U, V, z_dim) containing pairwise interactions,
            where z_dim can be arbitrary. This is the main input for the pairwise bias computation. If return_z_proj is True, z should be of shape (B, H, U, V).
        mask: Attention mask of shape (B, V) or (B * M, V) indicating which positions
            should be masked (0 = masked, 1 = unmasked).
        num_heads: Number of attention heads.
        w_proj_z: Weight matrix for z projection of shape (H, z_dim).
        w_proj_g: Weight matrix for gating projection of shape (D, D).
        w_proj_o: Weight matrix for output projection of shape (D, D).
        w_ln_z: Weight for layer normalization of z tensor of shape (z_dim,).
        b_ln_z: Bias for layer normalization of z tensor of shape (z_dim,).
        b_proj_z: Bias for z projection of shape (H,). Defaults to None.
        b_proj_g: Bias for gating projection of shape (D,). Defaults to None.
        b_proj_o: Bias for output projection of shape (D,). Defaults to None.
        inf: Large value used for masking invalid attention positions. Defaults to 1e6.
        eps: Epsilon value for layer normalization. Defaults to 1e-5.
        attn_scale: Scaling factor for attention scores. If None, uses 1/sqrt(head_dim).
            Defaults to None.
        return_z_proj: Whether to return the projected z tensor as the second output. Defaults to True.
        is_cached_z_proj: Whether the z tensor is already projected and cached.
            If True, z should be of shape (B, H, U, V). Defaults to False.

    Returns:
        - **output** (:class:`torch.Tensor`): Attention output of shape (B * M, S, D)
          with pairwise bias applied.
        - **proj_z** (:class:`torch.Tensor`): Projected z tensor of shape (B, H, U, V)
          containing the pairwise bias tensor with mask applied.

    Notes:
        - For short sequences (≤ CUEQ_ATTENTION_PAIR_BIAS_FALLBACK_THRESHOLD),
          uses PyTorch fallback implementation.
        - For long sequences, uses optimized Triton kernels with automatic
          backend selection (CUDNN, Flash Attention, Efficient Attention).
        - Multiplicity (M) is computed automatically from tensor shapes to allow
          processing multiple diffusion timesteps in a single forward pass.
        - The proj_z output is experimental to prevent breakage when caching
          of pair bias tensor is enabled in the next release.
        - Tested for bf16, fp16, fp32 and tf32. torch.set_float32_matmul_precision maybe used to toggle between fp32/tf32.
        - Currently, the kernel provides superior performance only when DH (head dimension) is a multiple of 32.
          For non-multiples of 32, we also recommend using graph compilation techniques like torch.compile, in addition.

    Examples:
        Basic usage without caching:

        >>> import torch
        >>> from cuequivariance_torch import attention_pair_bias
        >>> if torch.cuda.is_available():  # doctest: +SKIP
        ...     device = torch.device("cuda")
        ...     batch_size, seq_len, num_heads, heads_dim, hidden_dim = 1, 32, 2, 32, 64
        ...     query_len, key_len, z_dim = 32, 32, 16
        ...     # Create input tensors on GPU
        ...     s = torch.randn(batch_size, seq_len, hidden_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     q = torch.randn(batch_size, num_heads, query_len, heads_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     k = torch.randn(batch_size, num_heads, key_len, heads_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     v = torch.randn(batch_size, num_heads, key_len, heads_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     z = torch.randn(batch_size, query_len, key_len, z_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     mask = torch.rand(batch_size, key_len,
        ...                       device=device) < 0.5
        ...     w_proj_z = torch.randn(num_heads, z_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     w_proj_g = torch.randn(hidden_dim, hidden_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     w_proj_o = torch.randn(hidden_dim, hidden_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     w_ln_z = torch.randn(z_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     b_ln_z = torch.randn(z_dim,
        ...                     device=device, dtype=torch.bfloat16)
        ...     # Perform operation
        ...     output = attention_pair_bias(
        ...         s=s,
        ...         q=q,
        ...         k=k,
        ...         v=v,
        ...         z=z,
        ...         mask=mask,
        ...         num_heads=num_heads,
        ...         w_proj_z=w_proj_z,
        ...         w_proj_g=w_proj_g,
        ...         w_proj_o=w_proj_o,
        ...         w_ln_z=w_ln_z,
        ...         b_ln_z=b_ln_z,
        ...         return_z_proj=False,
        ...     )
        ...     print(output.shape)  # torch.Size([1, 32, 64])
        torch.Size([1, 32, 64])

        Example with caching (recommended for inference when z doesn't change):

        >>> # Check cache and determine if z is already projected
        >>> if model_cache is not None and "proj_z" in model_cache:  # doctest: +SKIP
        ...     z = model_cache["proj_z"]
        ...     is_cached_z = True
        ... else:
        ...     is_cached_z = False
        >>>
        >>> # Call attention_pair_bias
        >>> o, proj_z = attention_pair_bias(  # doctest: +SKIP
        ...     s=s, q=q, k=k, v=v, z=z, mask=mask,
        ...     num_heads=num_heads,
        ...     w_proj_z=w_proj_z if not is_cached_z else None,
        ...     w_proj_g=w_proj_g,
        ...     w_proj_o=w_proj_o,
        ...     w_ln_z=w_ln_z if not is_cached_z else None,
        ...     b_ln_z=b_ln_z if not is_cached_z else None,
        ...     return_z_proj=True,
        ...     is_cached_z_proj=is_cached_z,
        ... )
        >>>
        >>> # Cache proj_z for next call
        >>> if model_cache is not None and "proj_z" not in model_cache:  # doctest: +SKIP
        ...     model_cache["proj_z"] = proj_z
    """

    try:
        from cuequivariance_ops_torch.attention_pair_bias_torch import (
            attention_pair_bias as f,
        )
    except Exception:
        raise ImportError(
            "Error importing attention_pair_bias from cuequivariance_ops_torch."
        )
    else:
        return f(
            s,
            q,
            k,
            v,
            z,
            mask,
            num_heads,
            w_proj_z=w_proj_z,
            w_proj_g=w_proj_g,
            w_proj_o=w_proj_o,
            w_ln_z=w_ln_z,
            b_ln_z=b_ln_z,
            b_proj_z=b_proj_z,
            b_proj_g=b_proj_g,
            b_proj_o=b_proj_o,
            inf=inf,
            eps=eps,
            attn_scale=attn_scale,
            return_z_proj=return_z_proj,
            is_cached_z_proj=is_cached_z_proj,
        )
