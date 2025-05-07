# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from typing import List, Optional

import pytest
import torch

from cuequivariance_torch import triangle_flash_attention


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
    return_lse: bool = False,
) -> torch.Tensor:
    """PyTorch reference implementation of attention."""
    # Permute key for matrix multiplication
    key = permute_final_dims(key, (1, 0))

    # Compute attention scores
    a = torch.matmul(query, key)

    # Add biases
    for b in biases:
        a += b

    if return_lse:
        # Calculate log-sum-exp for numerical stability
        amax = a.max(dim=-1, keepdim=True)[0]
        lse = torch.logsumexp(a - amax, dim=-1, keepdim=True)
        a = torch.exp(a - amax - lse)
    else:
        amax = None
        lse = None
        a = torch.nn.functional.softmax(a, -1)

    # Apply attention to values
    a = torch.matmul(a, value)

    if return_lse:
        return a, lse, amax

    return a


class TFA(torch.nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_bias: torch.Tensor,
        triangle_bias: torch.Tensor,
        sm_scale: Optional[float] = None,
        use_tf32: bool = False,
        return_softmax_lse: bool = False,
        return_softmax_maximums: bool = False,
    ):
        return triangle_flash_attention(
            q,
            k,
            v,
            mask_bias,
            triangle_bias,
            sm_scale=sm_scale,
            use_tf32=use_tf32,  # Whether to use TF32 precision (False = use FP32)
            return_softmax_lse=return_softmax_lse,
            return_softmax_maximums=return_softmax_maximums,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("use_tf32", [False, True], ids=["fp32", "tf32"])
def test_compilation_methods(use_tf32):
    """Test that triangle attention works with different PyTorch compilation methods."""
    # Setup dimensions
    B = 2  # batch size
    T = 128  # sequence length
    H = 4  # number of heads
    D = 32  # head dimension (fixed)

    # Create input tensors
    torch.manual_seed(42)
    q = torch.randn(B, T, H, T, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, T, H, T, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, T, H, T, D, device="cuda", dtype=torch.float32)
    mask_bias = torch.zeros(B, T, 1, 1, T, device="cuda", dtype=torch.float32)
    triangle_bias = torch.randn(B, 1, H, T, T, device="cuda", dtype=torch.float32)

    args = (q, k, v, mask_bias, triangle_bias, 1.0, use_tf32)

    tfa = TFA()
    tfa_compiled = torch.compile(tfa, fullgraph=True)

    output, lse, maximums = tfa_compiled(
        *args, return_softmax_lse=True, return_softmax_maximums=True
    )

    tfa_scripted = torch.jit.script(tfa)
    output1, lse1, maximums1 = tfa_scripted(
        *args, return_softmax_lse=True, return_softmax_maximums=True
    )

    torch.testing.assert_close(output1, output)
    torch.testing.assert_close(lse1, lse)
    torch.testing.assert_close(maximums1, maximums)

    tfa_exported = torch.export.export(tfa, args=args).module()
    output2 = tfa_exported(*args)

    torch.testing.assert_close(output2, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("use_tf32", [False, True], ids=["fp32", "tf32"])
def test_vs_reference_implementation(use_tf32):
    """Test that triangle attention matches the PyTorch reference implementation."""

    torch.manual_seed(1100)
    device = torch.device("cuda")

    # Define dimensions
    B, N, H, D = 2, 186, 4, 32

    # Create tensors
    q = torch.randn((B, N, H, N, D), device=device)
    k = torch.randn((B, N, H, N, D), device=device)
    v = torch.randn((B, N, H, N, D), device=device)
    q_scaled = q / math.sqrt(D)

    # Create masks/biases
    mask_bias = torch.zeros((B, N, 1, 1, N), device=device)
    triangle_bias = torch.randn((B, 1, H, N, N), device=device)
    biases = [mask_bias, triangle_bias]

    with torch.no_grad():
        pytorch_output = reference_attention(q_scaled, k, v, biases)

    with torch.no_grad():
        cudnn_output, _ = triangle_flash_attention(
            q=q_scaled,
            k=k,
            v=v,
            mask_bias=mask_bias,
            triangle_bias=triangle_bias,
            sm_scale=1.0,  # q has been scaled already
            return_softmax_lse=True,
            use_tf32=use_tf32,
        )

    if use_tf32:
        assert torch.allclose(cudnn_output, pytorch_output, atol=1e-2, rtol=1e-2), (
            "Outputs don't match"
        )
    else:
        assert torch.allclose(cudnn_output, pytorch_output, atol=1e-6, rtol=1e-6), (
            "Outputs don't match"
        )
