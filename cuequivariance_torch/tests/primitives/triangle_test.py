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
import sys
import types

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
    # The optional generalized projection params must be
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


def test_attention_pair_bias_forwards_generalized_projection_and_defaults(monkeypatch):
    """The public backend export receives generalized and cached-pair options."""
    backend_module = types.ModuleType("cuequivariance_ops_torch")
    calls = []
    output = torch.tensor([123.0])
    cached_output = torch.tensor([456.0])

    def backend_spy(
        single_repr,
        pair_repr,
        mask,
        num_heads,
        w_ln_a,
        b_ln_a,
        w_proj_q,
        b_proj_q,
        w_proj_k,
        w_proj_v,
        w_proj_g,
        w_proj_o,
        w_proj_z,
        **kwargs,
    ):
        calls.append(
            {
                "single_repr": single_repr,
                "pair_repr": pair_repr,
                "mask": mask,
                "num_heads": num_heads,
                "w_proj_z": w_proj_z,
                "kwargs": kwargs,
            }
        )
        return (cached_output if kwargs["is_cached_z_proj"] else output), None

    backend_module.attention_pair_bias = backend_spy
    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_ops_torch",
        backend_module,
    )
    monkeypatch.delitem(
        sys.modules,
        "cuequivariance_ops_torch.attention_pair_bias",
        raising=False,
    )

    tensor = torch.tensor([1.0])
    common = dict(
        single_repr=tensor,
        mask=None,
        num_heads=2,
        w_ln_a=tensor,
        b_ln_a=None,
        w_proj_q=tensor,
        b_proj_q=None,
        w_proj_k=tensor,
        w_proj_v=tensor,
        w_proj_g=tensor,
        w_proj_o=tensor,
    )
    generalized = {
        "b_proj_k": torch.tensor([3.0]),
        "b_proj_v": torch.tensor([4.0]),
        "w_ln_q": torch.tensor([5.0]),
        "b_ln_q": torch.tensor([6.0]),
        "w_ln_k": torch.tensor([7.0]),
        "b_ln_k": torch.tensor([8.0]),
        "b_proj_g": torch.tensor([9.0]),
    }

    actual, _ = cuet.attention_pair_bias(
        **common,
        pair_repr=torch.tensor([2.0]),
        w_proj_z=torch.tensor([10.0]),
        **generalized,
    )
    assert actual is output
    assert calls[-1]["w_proj_z"].item() == 10.0
    for name, value in generalized.items():
        assert calls[-1]["kwargs"][name] is value

    rms_generalized = {
        **generalized,
        "b_ln_q": None,
        "b_ln_k": None,
    }
    actual, _ = cuet.attention_pair_bias(
        **common,
        pair_repr=torch.tensor([11.0]),
        w_proj_z=torch.tensor([12.0]),
        norm_kind="rms_norm",
        **rms_generalized,
    )
    assert actual is output
    assert calls[-1]["kwargs"]["norm_kind"] == "rms_norm"

    actual, _ = cuet.attention_pair_bias(
        **common,
        pair_repr=torch.tensor([13.0]),
        is_cached_z_proj=True,
    )
    assert actual is cached_output
    assert calls[-1]["w_proj_z"] is None
    assert calls[-1]["kwargs"]["is_cached_z_proj"] is True
    assert "norm_kind" not in calls[-1]["kwargs"]
    for name in generalized:
        assert calls[-1]["kwargs"][name] is None

    with pytest.raises(ValueError, match="w_proj_z is required"):
        cuet.attention_pair_bias(
            **common,
            pair_repr=torch.tensor([14.0]),
        )
