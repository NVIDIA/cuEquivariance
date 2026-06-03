# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch

import cuequivariance_torch as cuet


def test_triangle_attention():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Set up dimensions
        batch_size, seq_len, num_heads, hidden_dim = 1, 16, 2, 32
        # Create input tensors on GPU with float16 precision
        q = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            seq_len,
            hidden_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        k = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            seq_len,
            hidden_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        v = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            seq_len,
            hidden_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        bias = torch.randn(
            batch_size,
            1,
            num_heads,
            seq_len,
            seq_len,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        # Create optional mask
        mask = torch.rand(batch_size, seq_len, 1, 1, seq_len, device=device) < 0.5
        # Calculate scale
        scale = 1 / math.sqrt(hidden_dim)
        # Forward pass
        output, lse, max_val = cuet.triangle_attention(
            q=q, k=k, v=v, bias=bias, mask=mask, scale=scale, return_aux=True
        )
        assert output.shape == torch.Size(
            [batch_size, seq_len, num_heads, seq_len, hidden_dim]
        )
        # Create gradient tensor and perform backward pass
        grad_out = torch.randn_like(output)
        output.backward(grad_out)
        # Access gradients
        assert q.grad.shape == torch.Size(
            [batch_size, seq_len, num_heads, seq_len, hidden_dim]
        )
        assert k.grad.shape == torch.Size(
            [batch_size, seq_len, num_heads, seq_len, hidden_dim]
        )
        assert v.grad.shape == torch.Size(
            [batch_size, seq_len, num_heads, seq_len, hidden_dim]
        )
        assert bias.grad.shape == torch.Size(
            [batch_size, 1, num_heads, seq_len, seq_len]
        )


def test_triangle_multiplicative_update():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size, seq_len, hidden_dim = 1, 32, 32
        # Create input tensor
        x = torch.randn(
            batch_size, seq_len, seq_len, hidden_dim, requires_grad=True, device=device
        )
        # Create mask (1 for valid positions, 0 for masked)
        mask = torch.ones(batch_size, seq_len, seq_len, device=device)
        # Perform triangular multiplication
        output = cuet.triangle_multiplicative_update(
            x=x,
            direction="outgoing",  # or "incoming"
            mask=mask,
        )
        assert output.shape == torch.Size([batch_size, seq_len, seq_len, hidden_dim])
        # Create gradient tensor and perform backward pass
        grad_out = torch.randn_like(output)
        output.backward(grad_out)
        # Access gradients
        assert x.grad.shape == torch.Size([batch_size, seq_len, seq_len, hidden_dim])


def test_attention_pair_bias():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size, seq_len, num_heads, heads_dim, hidden_dim = 1, 32, 2, 32, 64
        query_len, key_len, z_dim = 32, 32, 16
        # Create input tensors on GPU
        s = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32
        )
        q = torch.randn(
            batch_size,
            num_heads,
            query_len,
            heads_dim,
            device=device,
            dtype=torch.float32,
        )
        k = torch.randn(
            batch_size,
            num_heads,
            key_len,
            heads_dim,
            device=device,
            dtype=torch.float32,
        )
        v = torch.randn(
            batch_size,
            num_heads,
            key_len,
            heads_dim,
            device=device,
            dtype=torch.float32,
        )
        z = torch.randn(
            batch_size, query_len, key_len, z_dim, device=device, dtype=torch.float32
        )
        mask = torch.rand(batch_size, key_len, device=device) < 0.5
        w_proj_z = torch.randn(num_heads, z_dim, device=device, dtype=torch.float32)
        w_proj_g = torch.randn(
            hidden_dim, hidden_dim, device=device, dtype=torch.float32
        )
        w_proj_o = torch.randn(
            hidden_dim, hidden_dim, device=device, dtype=torch.float32
        )
        w_ln_z = torch.randn(z_dim, device=device, dtype=torch.float32)
        b_ln_z = torch.randn(z_dim, device=device, dtype=torch.float32)
        # Perform operation

        output, proj_z = cuet.attention_pair_bias(
            s=s,
            q=q,
            k=k,
            v=v,
            z=z,
            mask=mask,
            num_heads=num_heads,
            w_proj_z=w_proj_z,
            w_proj_g=w_proj_g,
            w_proj_o=w_proj_o,
            w_ln_z=w_ln_z,
            b_ln_z=b_ln_z,
        )
        assert output.shape == torch.Size([batch_size, seq_len, hidden_dim])


def test_trimul_precision_mxfp8_enum_member():
    # BIO-714: MXFP8 must be discoverable on the public TriMulPrecision enum, whether it
    # is imported from the cuequivariance_ops_torch backend or resolved from the in-module
    # fallback enum (no-ops-installed case). Both must carry the same sentinel values.
    assert hasattr(cuet.TriMulPrecision, "MXFP8")
    assert hasattr(cuet.TriMulPrecision, "BFLOAT16")
    assert int(cuet.TriMulPrecision.MXFP8.value) == -1000
    assert int(cuet.TriMulPrecision.BFLOAT16.value) == -1001


def test_triangle_multiplicative_update_mxfp8_inference():
    # BIO-714: exercise the opt-in, inference-only MXFP8 path via the enum member.
    # seq_len is > the eager bf16-fallback threshold (150) and divisible by 32 so the
    # optimized path engages; on a Blackwell GPU (cc >= 10.0) this hits the MXFP8 CUTLASS
    # kernel, and on other GPUs it transparently falls back to bf16 (same output shape).
    if not torch.cuda.is_available():
        return
    try:
        import cuequivariance_ops_torch  # noqa: F401
    except ImportError:
        return  # backend ops not installed; the MXFP8 path is unavailable
    device = torch.device("cuda")
    batch_size, seq_len, hidden_dim = 1, 160, 32
    expected = torch.Size([batch_size, seq_len, seq_len, hidden_dim])
    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, seq_len, hidden_dim, device=device)
        mask = torch.ones(batch_size, seq_len, seq_len, device=device)
        # Both the enum member and the legacy string spelling select the same path.
        for precision in (cuet.TriMulPrecision.MXFP8, "mxfp8"):
            output = cuet.triangle_multiplicative_update(
                x=x,
                direction="outgoing",
                mask=mask,
                precision=precision,
            )
            assert output.shape == expected
