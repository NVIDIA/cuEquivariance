# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Attention with pair bias for JAX."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

try:
    from cuequivariance_ops_jax import (
        attention_pair_bias as _attention_pair_bias_backend,
    )

    HAS_CUE_OPS_JAX_APB = True
except ImportError:
    _attention_pair_bias_backend = None
    HAS_CUE_OPS_JAX_APB = False


def _layer_norm(
    x: jax.Array,
    weight: Optional[jax.Array],
    bias: Optional[jax.Array],
    eps: float,
) -> jax.Array:
    output_dtype = x.dtype
    stats_dtype = (
        jnp.float32
        if output_dtype in (jnp.float16, jnp.bfloat16)
        else output_dtype
    )
    x_stats = x.astype(stats_dtype)
    centered = x_stats - jnp.mean(x_stats, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    normalized = centered * jax.lax.rsqrt(variance + eps)
    if weight is not None:
        normalized = normalized * weight.astype(stats_dtype)
    if bias is not None:
        normalized = normalized + bias.astype(stats_dtype)
    return normalized.astype(output_dtype)


def _linear(
    x: jax.Array,
    weight: jax.Array,
    bias: Optional[jax.Array],
    precision: jax.lax.Precision | None,
) -> jax.Array:
    dtype = x.dtype
    out = jnp.einsum(
        "...d,od->...o",
        x,
        weight.astype(dtype),
        precision=precision,
    )
    if bias is not None:
        out = out + bias.astype(dtype)
    return out


def _validate_inputs(
    single_repr: jax.Array,
    pair_repr: jax.Array,
    mask: Optional[jax.Array],
    num_heads: int,
    w_ln_a: jax.Array,
    w_proj_q: jax.Array,
    w_proj_k: jax.Array,
    w_proj_v: jax.Array,
    w_proj_g: jax.Array,
    w_proj_o: jax.Array,
    w_proj_z: Optional[jax.Array],
    *,
    is_cached_z_proj: bool,
) -> tuple[int, int, int, int, int]:
    if single_repr.ndim != 3:
        raise ValueError(
            f"single_repr must have shape (B * M, N, D), got {single_repr.shape}"
        )
    if pair_repr.ndim != 4:
        raise ValueError(
            "pair_repr must have shape (B, N, N, D_z), or (B, H, N, N) "
            f"when cached, got {pair_repr.shape}"
        )
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    BM, N, D = single_repr.shape
    B = pair_repr.shape[0]
    if B < 1:
        raise ValueError("pair_repr batch dimension must be >= 1")
    if BM % B != 0:
        raise ValueError(
            f"single_repr batch {BM} must be divisible by pair_repr batch {B}"
        )
    multiplicity = BM // B

    projected_channels = w_proj_q.shape[0]
    if projected_channels % num_heads != 0:
        raise ValueError(
            f"projected channels {projected_channels} must be divisible by "
            f"num_heads {num_heads}"
        )
    head_dim = projected_channels // num_heads
    for name, value in (
        ("w_proj_q", w_proj_q),
        ("w_proj_k", w_proj_k),
        ("w_proj_v", w_proj_v),
        ("w_proj_g", w_proj_g),
    ):
        if value.shape != (projected_channels, D):
            raise ValueError(
                f"{name} must have shape {(projected_channels, D)}, "
                f"got {value.shape}"
            )
    if w_proj_o.shape != (D, projected_channels):
        raise ValueError(
            f"w_proj_o must have shape {(D, projected_channels)}, "
            f"got {w_proj_o.shape}"
        )
    if w_ln_a.shape != (D,):
        raise ValueError(f"w_ln_a must have shape {(D,)}, got {w_ln_a.shape}")

    if is_cached_z_proj:
        if pair_repr.shape != (B, num_heads, N, N):
            raise ValueError(
                "cached pair_repr must have shape "
                f"{(B, num_heads, N, N)}, got {pair_repr.shape}"
            )
    else:
        if pair_repr.shape[1:3] != (N, N):
            raise ValueError(
                f"raw pair_repr must have square sequence axes {(N, N)}, "
                f"got {pair_repr.shape[1:3]}"
            )
        if w_proj_z is None:
            raise ValueError("w_proj_z is required when is_cached_z_proj is False")
        if w_proj_z.shape != (num_heads, pair_repr.shape[-1]):
            raise ValueError(
                f"w_proj_z must have shape {(num_heads, pair_repr.shape[-1])}, "
                f"got {w_proj_z.shape}"
            )

    if mask is not None and mask.shape not in ((B, N), (BM, N)):
        raise ValueError(
            f"mask must have shape {(B, N)} or {(BM, N)}, got {mask.shape}"
        )
    return B, BM, N, head_dim, multiplicity


def _attention_pair_bias_reference(
    single_repr: jax.Array,
    pair_repr: jax.Array,
    mask: Optional[jax.Array],
    num_heads: int,
    w_ln_a: jax.Array,
    b_ln_a: Optional[jax.Array],
    w_proj_q: jax.Array,
    b_proj_q: Optional[jax.Array],
    w_proj_k: jax.Array,
    w_proj_v: jax.Array,
    w_proj_g: jax.Array,
    w_proj_o: jax.Array,
    w_proj_z: Optional[jax.Array] = None,
    b_proj_o: Optional[jax.Array] = None,
    b_proj_z: Optional[jax.Array] = None,
    w_ln_z: Optional[jax.Array] = None,
    b_ln_z: Optional[jax.Array] = None,
    inf: float = 1e6,
    eps: float = 1e-5,
    attn_scale: Optional[float] = None,
    return_z_proj: bool = False,
    is_cached_z_proj: bool = False,
    *,
    b_proj_k: Optional[jax.Array] = None,
    b_proj_v: Optional[jax.Array] = None,
    w_ln_q: Optional[jax.Array] = None,
    b_ln_q: Optional[jax.Array] = None,
    w_ln_k: Optional[jax.Array] = None,
    b_ln_k: Optional[jax.Array] = None,
    b_proj_g: Optional[jax.Array] = None,
    precision: jax.lax.Precision | None = None,
) -> tuple[jax.Array, Optional[jax.Array]]:
    """Pure-JAX reference implementation."""
    B, BM, N, head_dim, multiplicity = _validate_inputs(
        single_repr,
        pair_repr,
        mask,
        num_heads,
        w_ln_a,
        w_proj_q,
        w_proj_k,
        w_proj_v,
        w_proj_g,
        w_proj_o,
        w_proj_z,
        is_cached_z_proj=is_cached_z_proj,
    )
    input_dtype = single_repr.dtype
    normalized = _layer_norm(single_repr, w_ln_a, b_ln_a, eps)
    q = _linear(normalized, w_proj_q, b_proj_q, precision)
    k = _linear(normalized, w_proj_k, b_proj_k, precision)
    v = _linear(normalized, w_proj_v, b_proj_v, precision)
    if w_ln_q is not None or b_ln_q is not None:
        q = _layer_norm(q, w_ln_q, b_ln_q, eps)
    if w_ln_k is not None or b_ln_k is not None:
        k = _layer_norm(k, w_ln_k, b_ln_k, eps)
    q, k, v = (
        value.reshape(BM, N, num_heads, head_dim).transpose(0, 2, 1, 3)
        for value in (q, k, v)
    )

    if is_cached_z_proj:
        z_proj = pair_repr
    else:
        z_norm = _layer_norm(pair_repr, w_ln_z, b_ln_z, eps)
        z_proj = _linear(z_norm, w_proj_z, b_proj_z, precision)
        z_proj = jnp.transpose(z_proj, (0, 3, 1, 2))
    z_proj = z_proj.astype(input_dtype)
    z_expanded = jnp.repeat(z_proj, multiplicity, axis=0)

    if mask is None:
        expanded_mask = jnp.ones((BM, N), dtype=jnp.float32)
    else:
        expanded_mask = mask.astype(jnp.float32)
        if mask.shape[0] == B:
            expanded_mask = jnp.repeat(expanded_mask, multiplicity, axis=0)
    pair_bias = (
        z_expanded.astype(jnp.float32)
        + (1.0 - expanded_mask[:, None, None, :]) * (-inf)
    ).astype(input_dtype)

    scale = head_dim**-0.5 if attn_scale is None else attn_scale
    scores = jnp.einsum(
        "bhid,bhjd->bhij", q, k, precision=precision
    ).astype(jnp.float32)
    scores = scores * scale + pair_bias.astype(jnp.float32)
    attention = jax.nn.softmax(scores, axis=-1).astype(input_dtype)
    out = jnp.einsum(
        "bhij,bhjd->bihd", attention, v, precision=precision
    ).reshape(BM, N, num_heads * head_dim)

    gate = jax.nn.sigmoid(
        _linear(normalized, w_proj_g, b_proj_g, precision)
    )
    output = _linear(gate * out, w_proj_o, b_proj_o, precision).astype(
        input_dtype
    )
    return output, z_proj if return_z_proj else None


def attention_pair_bias(
    single_repr: jax.Array,
    pair_repr: jax.Array,
    mask: Optional[jax.Array],
    num_heads: int,
    w_ln_a: jax.Array,
    b_ln_a: Optional[jax.Array],
    w_proj_q: jax.Array,
    b_proj_q: Optional[jax.Array],
    w_proj_k: jax.Array,
    w_proj_v: jax.Array,
    w_proj_g: jax.Array,
    w_proj_o: jax.Array,
    w_proj_z: Optional[jax.Array] = None,
    b_proj_o: Optional[jax.Array] = None,
    b_proj_z: Optional[jax.Array] = None,
    w_ln_z: Optional[jax.Array] = None,
    b_ln_z: Optional[jax.Array] = None,
    inf: float = 1e6,
    eps: float = 1e-5,
    attn_scale: Optional[float] = None,
    return_z_proj: bool = False,
    is_cached_z_proj: bool = False,
    *,
    b_proj_k: Optional[jax.Array] = None,
    b_proj_v: Optional[jax.Array] = None,
    w_ln_q: Optional[jax.Array] = None,
    b_ln_q: Optional[jax.Array] = None,
    w_ln_k: Optional[jax.Array] = None,
    b_ln_k: Optional[jax.Array] = None,
    b_proj_g: Optional[jax.Array] = None,
    precision: jax.lax.Precision | None = None,
) -> tuple[jax.Array, Optional[jax.Array]]:
    """Compute attention with a pairwise bias.

    This matches the Torch APB signature and supports raw pair representations
    ``[B, N, N, D_z]`` as well as cached projections ``[B, H, N, N]``.  The
    operation remains available as pure JAX when ``cuequivariance_ops_jax`` is
    not installed.  When it is installed, eligible CUDA cells reuse its
    triangle-attention backend while unsupported shapes retain reference
    semantics. The public APB contract requires at least one valid key in each
    mask row; fully masked rows are unsupported.

    ``single_repr`` has shape ``[B * M, N, D]``.  ``mask`` may have shape
    ``[B, N]`` or ``[B * M, N]`` and is an additive finite mask, so gradients
    with respect to floating masks are preserved.
    """
    implementation = (
        _attention_pair_bias_backend
        if HAS_CUE_OPS_JAX_APB
        else _attention_pair_bias_reference
    )
    return implementation(
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
        b_proj_o,
        b_proj_z,
        w_ln_z,
        b_ln_z,
        inf,
        eps,
        attn_scale,
        return_z_proj,
        is_cached_z_proj,
        b_proj_k=b_proj_k,
        b_proj_v=b_proj_v,
        w_ln_q=w_ln_q,
        b_ln_q=b_ln_q,
        w_ln_k=w_ln_k,
        b_ln_k=b_ln_k,
        b_proj_g=b_proj_g,
        precision=precision,
    )


__all__ = ["attention_pair_bias"]
