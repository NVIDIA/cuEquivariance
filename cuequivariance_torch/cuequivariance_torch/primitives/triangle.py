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
        (3) **Limitation**: Full FP32 is not supported for backward pass. Please set `torch.backends.cuda.matmul.allow_tf32=True`.

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
    g_in_weight: Optional[torch.Tensor] = None,
    norm_out_weight: Optional[torch.Tensor] = None,
    norm_out_bias: Optional[torch.Tensor] = None,
    p_out_weight: Optional[torch.Tensor] = None,
    g_out_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply triangle multiplicative update operation.

    This function performs a triangle multiplicative update operation, which is a key component
    in the AlphaFold2 architecture. The operation consists of:

    1. Input normalization and gating
    2. Triangular projection (either outgoing or incoming)
    3. Output normalization and gating

    The function supports both ahead-of-time (AOT) tuning and just-in-time (JIT) tuning.
    Auto-tuning behavior can be controlled through environment variables:

    - Default: Full Ahead-of-Time (AOT) auto-tuning enabled for optimal performance **(may take several hours)**
    - Quick testing: Set `CUEQ_DISABLE_AOT_TUNING = 1` and `CUEQ_DEFAULT_CONFIG = 1` to disable all tuning
    - On-Demand tuning: `CUEQ_DISABLE_AOT_TUNING = 1`, auto-tunes for new shapes encountered on first run. (may take several minutes)
    - Note: When using Docker with default or on-demand tuning enabled, commit the container to persist tuning changes

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, N, D) where:
            B is the batch size
            N is the sequence length
            D is the hidden dimension
        direction (str): Direction of the triangular projection. Must be either "outgoing" or "incoming".
        mask (torch.Tensor): Optional Mask tensor of shape (B, N, N) for masking the output.
        norm_in_weight (torch.Tensor): Weight tensor for input normalization of shape (D,).
        norm_in_bias (torch.Tensor): Bias tensor for input normalization of shape (D,).
        p_in_weight (torch.Tensor): Weight tensor for input projection of shape (2D, D).
        g_in_weight (torch.Tensor): Weight tensor for input gating of shape (2D, D).
        norm_out_weight (torch.Tensor): Weight tensor for output normalization of shape (D,).
        norm_out_bias (torch.Tensor): Bias tensor for output normalization of shape (D,).
        p_out_weight (torch.Tensor): Weight tensor for output projection of shape (D, D).
        g_out_weight (torch.Tensor): Weight tensor for output gating of shape (D, D).
        eps (float, optional): Small constant for numerical stability in normalization. Defaults to 1e-5.

    Returns:
        Output tensor of shape (batch_size, seq_len, seq_len, hidden_dim)

    Notes:
        (1) Context is saved for backward pass. You don't need to save it manually.
        (2) Kernel precision (fp32, bf16,fp16) is based on input dtypes. For tf32, set it from torch global scope
        (3) **Limitation** Currently only supports hidden_dim values that are multiples of 32.

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
        ...     ) #If CUEQ_DISABLE_AOT_TUNING is not set to 1, will default to json config look-up if config is available. If not, then proceeds to auto-tuning using Ahead Of Time Compilation.
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
            mask,
            norm_in_weight,
            norm_in_bias,
            p_in_weight,
            g_in_weight,
            norm_out_weight,
            norm_out_bias,
            p_out_weight,
            g_out_weight,
            eps,
        )
