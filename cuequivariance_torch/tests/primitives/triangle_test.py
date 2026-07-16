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

import pytest
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


def test_triangle_attention_kv_lengths():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size, seq_len, num_heads, hidden_dim = 1, 16, 2, 32
        q = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            seq_len,
            hidden_dim,
            device=device,
            dtype=torch.float16,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        bias = torch.randn(
            batch_size,
            1,
            num_heads,
            seq_len,
            seq_len,
            device=device,
            dtype=torch.float32,
        )
        seq_lengths = torch.tensor([12], device=device, dtype=torch.int32)
        positions = torch.arange(seq_len, device=device)
        row_valid = positions.view(1, seq_len) < seq_lengths.view(batch_size, 1)
        col_valid = positions.view(1, seq_len) < seq_lengths.view(batch_size, 1)
        mask = row_valid.view(batch_size, seq_len, 1, 1, 1) & col_valid.view(
            batch_size, 1, 1, 1, seq_len
        )
        kv_lengths = cuet.mask_to_kv_lengths(mask)

        assert kv_lengths.shape == torch.Size([batch_size, seq_len, 1, 1, 1])
        assert kv_lengths.dtype == torch.int32
        output = cuet.triangle_attention(
            q=q,
            k=k,
            v=v,
            bias=bias,
            scale=1 / math.sqrt(hidden_dim),
            kv_lengths=kv_lengths,
        )
        assert output.shape == torch.Size(
            [batch_size, seq_len, num_heads, seq_len, hidden_dim]
        )
        zero_rows = kv_lengths.view(batch_size, seq_len) == 0
        torch.testing.assert_close(
            output[zero_rows], torch.zeros_like(output[zero_rows])
        )


def test_triangle_attention_mask_and_kv_lengths_mutually_exclusive():
    # kv_lengths selects the SM100f length fast path; a dense mask uses the
    # fallback. Passing both is contradictory and must raise. The guard lives in
    # the public wrapper, before any backend dispatch, so this needs no GPU.
    n = 8
    q = torch.zeros(1, n, 1, n, 8)
    k = torch.zeros(1, n, 1, n, 8)
    v = torch.zeros(1, n, 1, n, 8)
    bias = torch.zeros(1, 1, 1, n, n)
    mask = torch.ones(1, n, 1, 1, n, dtype=torch.bool)
    kv_lengths = torch.full((1, n, 1, 1, 1), n, dtype=torch.int32)
    with pytest.raises(ValueError, match="not both"):
        cuet.triangle_attention(q, k, v, bias, mask=mask, kv_lengths=kv_lengths)


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
        z_dim = 16
        inner = num_heads * heads_dim
        dt = torch.float32

        single_repr = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, dtype=dt
        )
        pair_repr = torch.randn(
            batch_size, seq_len, seq_len, z_dim, device=device, dtype=dt
        )
        mask = torch.rand(batch_size, seq_len, device=device) < 0.5
        w_ln_a = torch.randn(hidden_dim, device=device, dtype=dt)
        b_ln_a = torch.randn(hidden_dim, device=device, dtype=dt)
        w_proj_q = torch.randn(inner, hidden_dim, device=device, dtype=dt)
        b_proj_q = torch.randn(inner, device=device, dtype=dt)
        w_proj_k = torch.randn(inner, hidden_dim, device=device, dtype=dt)
        w_proj_v = torch.randn(inner, hidden_dim, device=device, dtype=dt)
        w_proj_g = torch.randn(inner, hidden_dim, device=device, dtype=dt)
        w_proj_o = torch.randn(hidden_dim, inner, device=device, dtype=dt)
        w_proj_z = torch.randn(num_heads, z_dim, device=device, dtype=dt)
        w_ln_z = torch.randn(z_dim, device=device, dtype=dt)
        b_ln_z = torch.randn(z_dim, device=device, dtype=dt)
        # Perform operation
        output, z_proj = cuet.attention_pair_bias(
            single_repr=single_repr,
            pair_repr=pair_repr,
            mask=mask,
            num_heads=num_heads,
            w_ln_a=w_ln_a,
            b_ln_a=b_ln_a,
            w_proj_q=w_proj_q,
            b_proj_q=b_proj_q,
            w_proj_k=w_proj_k,
            w_proj_v=w_proj_v,
            w_proj_g=w_proj_g,
            w_proj_o=w_proj_o,
            w_proj_z=w_proj_z,
            w_ln_z=w_ln_z,
            b_ln_z=b_ln_z,
        )
        assert output.shape == torch.Size([batch_size, seq_len, hidden_dim])


def test_attention_pair_bias_generalized_projection():
    # The optional generalized (Proteina/Complexa) projection params must be
    # forwarded to the backend and actually change the output relative to the
    # strict OpenFold3/Boltz call that omits them.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size, seq_len, num_heads, heads_dim, hidden_dim = 1, 32, 2, 32, 64
        z_dim = 16
        inner = num_heads * heads_dim
        dt = torch.float32

        single_repr = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, dtype=dt
        )
        pair_repr = torch.randn(
            batch_size, seq_len, seq_len, z_dim, device=device, dtype=dt
        )
        mask = torch.rand(batch_size, seq_len, device=device) < 0.5
        common = dict(
            single_repr=single_repr,
            pair_repr=pair_repr,
            mask=mask,
            num_heads=num_heads,
            w_ln_a=torch.randn(hidden_dim, device=device, dtype=dt),
            b_ln_a=torch.randn(hidden_dim, device=device, dtype=dt),
            w_proj_q=torch.randn(inner, hidden_dim, device=device, dtype=dt),
            b_proj_q=torch.randn(inner, device=device, dtype=dt),
            w_proj_k=torch.randn(inner, hidden_dim, device=device, dtype=dt),
            w_proj_v=torch.randn(inner, hidden_dim, device=device, dtype=dt),
            w_proj_g=torch.randn(inner, hidden_dim, device=device, dtype=dt),
            w_proj_o=torch.randn(hidden_dim, inner, device=device, dtype=dt),
            w_proj_z=torch.randn(num_heads, z_dim, device=device, dtype=dt),
            w_ln_z=torch.randn(z_dim, device=device, dtype=dt),
            b_ln_z=torch.randn(z_dim, device=device, dtype=dt),
        )

        strict, _ = cuet.attention_pair_bias(**common)
        generalized, _ = cuet.attention_pair_bias(
            **common,
            b_proj_k=torch.randn(inner, device=device, dtype=dt),
            b_proj_v=torch.randn(inner, device=device, dtype=dt),
            w_ln_q=torch.randn(inner, device=device, dtype=dt),
            b_ln_q=torch.randn(inner, device=device, dtype=dt),
            w_ln_k=torch.randn(inner, device=device, dtype=dt),
            b_ln_k=torch.randn(inner, device=device, dtype=dt),
            b_proj_g=torch.randn(inner, device=device, dtype=dt),
        )
        assert generalized.shape == torch.Size([batch_size, seq_len, hidden_dim])
        assert not torch.allclose(strict, generalized), (
            "generalized projection params were not forwarded to the backend"
        )
