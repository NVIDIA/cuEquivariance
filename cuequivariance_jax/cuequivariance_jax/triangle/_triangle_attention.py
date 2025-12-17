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
from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.interpreters import batching, mlir, xla

from cuequivariance_jax.triangle._naive_batching import naive_batching_rule

try:
    from cuequivariance_ops_jax import (
        triangle_attention_cuda_bwd,
        triangle_attention_cuda_fwd,
    )

    HAS_CUE_OPS_JAX = True
except ImportError:
    HAS_CUE_OPS_JAX = False


def triangle_attention(
    q: jax.Array,  # [B, N, H, S_qo, D]
    k: jax.Array,  # [B, N, H, S_kv, D]
    v: jax.Array,  # [B, N, H, S_kv, D]
    bias: jax.Array,  # [B, 1, H, S_qo, S_kv]
    mask: jax.Array,  # [B, N, 1, 1, S_kv] boolean
    scale: float,
    precision: jax.lax.Precision | None = None,
):
    r"""triangle attention

    Args:
        q: Query tensor of shape [B, N, H, S_qo, D].
        k: Key tensor of shape [B, N, H, S_kv, D].
        v: Value tensor of shape [B, N, H, S_kv, D].
        bias: Bias tensor of shape [B, 1, H, S_qo, S_kv].
        mask: Mask tensor of shape [B, N, 1, 1, S_kv] (boolean, True means valid).
        scale: Scaling factor for the dot product.
        precision: Precision for the computation (default is None).

    Returns:
        A tuple containing the attention output, log-sum-exp, and maximum value.

    .. math::

        \text{Attention}_a(Q, K, V, M, T) = \sum_b \mathrm{softmax}_b\left( M_b \cdot (Q_a K_b + T_{ab}) + (1 - M_b) \cdot (-10^9) \right) V_b

    where :math:`Q`, :math:`K`, and :math:`V` are the query, key, and value tensors,
    :math:`M` is the mask bias, and :math:`T` is the triangle bias.

    .. note::
        This operation uses a custom CUDA kernel for performance. When using this function
        on multiple devices, manual sharding is required to achieve proper performance.
        Without explicit sharding, performance will be significantly degraded. See
        `JAX shard_map documentation <https://docs.jax.dev/en/latest/notebooks/shard_map.html>`_
        for details on manual parallelism.
    """
    return triangle_attention_custom_vjp(
        q, k, v, bias, mask, scale=scale, precision=precision
    )


@partial(jax.jit, static_argnames=("scale", "precision"))
def triangle_attention_jax_fwd(
    q: jax.Array,  # [B, N, H, S_qo, D]
    k: jax.Array,  # [B, N, H, S_kv, D]
    v: jax.Array,  # [B, N, H, S_kv, D]
    bias: jax.Array,  # [B, 1, H, S_qo, S_kv]
    mask: jax.Array,  # [B, N, 1, 1, S_kv] boolean
    scale: float,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:  # [B, N, H, S_qo, D]
    r"""JAX reference implementation for triangle attention."""
    dtype = q.dtype
    assert k.dtype == dtype and v.dtype == dtype

    q = scale * q
    a = jnp.einsum("...ai,...bi->...ab", q, k, precision=precision)
    a = a + bias
    a = jnp.where(mask, a, -1e9)

    a = a.astype(jnp.float32)  # [B, N, H, S_qo, S_kv]
    amax = jnp.max(a, axis=-1, keepdims=True)
    lse = jax.scipy.special.logsumexp(a - amax, axis=-1, keepdims=True)
    a = jnp.exp(a - amax - lse)

    a = a.astype(dtype)
    a = jnp.einsum(
        "...ab, ...bi -> ...ai", a, v, precision=precision
    )  # [B, N, H, S_qo, D]

    return a, lse, amax


fwd_p = jax.extend.core.Primitive("triangle_attention_fwd")
fwd_p.multiple_results = True

bwd_p = jax.extend.core.Primitive("triangle_attention_bwd")
bwd_p.multiple_results = True


def triangle_attention_fwd_abstract_eval(
    q: jax.core.ShapedArray,  # [B, N, H, S_qo, D]
    k: jax.core.ShapedArray,  # [B, N, H, S_kv, D]
    v: jax.core.ShapedArray,  # [B, N, H, S_kv, D]
    bias: jax.core.ShapedArray,  # [B, 1, H, S_qo, S_kv]
    mask: jax.core.ShapedArray,  # [B, N, 1, 1, S_kv] boolean
    **unused_kwargs,
) -> tuple[jax.core.ShapedArray, jax.core.ShapedArray, jax.core.ShapedArray]:
    B, N, H, S_qo, D = q.shape
    a_shape = jax.core.ShapedArray((B, N, H, S_qo, D), q.dtype)
    lse_shape = jax.core.ShapedArray((B, N, H, S_qo, 1), jnp.float32)
    amax_shape = jax.core.ShapedArray((B, N, H, S_qo, 1), jnp.float32)
    return a_shape, lse_shape, amax_shape


def triangle_attention_bwd_abstract_eval(
    da: jax.core.ShapedArray,  # [B, N, H, S_qo, D]
    a: jax.core.ShapedArray,  # [B, N, H, S_qo, D]
    lse: jax.core.ShapedArray,  # [B, N, H, S_qo, 1]
    q: jax.core.ShapedArray,  # [B, N, H, S_qo, D]
    k: jax.core.ShapedArray,  # [B, N, H, S_kv, D]
    v: jax.core.ShapedArray,  # [B, N, H, S_kv, D]
    bias: jax.core.ShapedArray,  # [B, 1, H, S_qo, S_kv]
    mask: jax.core.ShapedArray,  # [B, N, 1, 1, S_kv] boolean
    **unused_kwargs,
) -> tuple[
    jax.core.ShapedArray,
    jax.core.ShapedArray,
    jax.core.ShapedArray,
    jax.core.ShapedArray,
]:
    dq_shape = jax.core.ShapedArray(q.shape, q.dtype)
    dk_shape = jax.core.ShapedArray(k.shape, k.dtype)
    dv_shape = jax.core.ShapedArray(v.shape, v.dtype)
    dbias_shape = jax.core.ShapedArray(bias.shape, bias.dtype)
    return dq_shape, dk_shape, dv_shape, dbias_shape


def triangle_attention_fwd_impl(
    platform: str | None,
    q: jax.Array,  # [B, N, H, S_qo, D]
    k: jax.Array,  # [B, N, H, S_kv, D]
    v: jax.Array,  # [B, N, H, S_kv, D]
    bias: jax.Array,  # [B, 1, H, S_qo, S_kv]
    mask: jax.Array,  # [B, N, 1, 1, S_kv] boolean
    *,
    scale: float,
    precision: jax.lax.Precision | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if platform == "cuda":
        assert HAS_CUE_OPS_JAX, (
            "Please install cuequivariance_ops_jax for CUDA support."
        )
        return triangle_attention_cuda_fwd(q, k, v, mask, bias, scale, precision)
    else:
        return triangle_attention_jax_fwd(q, k, v, bias, mask, scale, precision)


def triangle_attention_bwd_impl(
    platform: str | None,
    da: jax.Array,  # [B, N, H, S_qo, D]
    a: jax.Array,  # [B, N, H, S_qo, D]
    lse: jax.Array,  # [B, N, H, S_qo, 1]
    q: jax.Array,  # [B, N, H, S_qo, D]
    k: jax.Array,  # [B, N, H, S_kv, D]
    v: jax.Array,  # [B, N, H, S_kv, D]
    bias: jax.Array,  # [B, 1, H, S_qo, S_kv]
    mask: jax.Array,  # [B, N, 1, 1, S_kv] boolean
    *,
    scale: float,
    precision: jax.lax.Precision | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if platform == "cuda":
        assert HAS_CUE_OPS_JAX, (
            "Please install cuequivariance_ops_jax for CUDA support."
        )
        dq, dk, dv, dbias = triangle_attention_cuda_bwd(
            da, a, lse, q, k, v, mask, bias, scale, precision
        )
        return dq, dk, dv, dbias.astype(bias.dtype)
    else:
        # Use JAX autodiff for backward pass
        def forward_fn(q, k, v, bias):
            a, lse, amax = triangle_attention_jax_fwd(
                q, k, v, bias, mask, scale, precision
            )
            return a

        _, vjp_fn = jax.vjp(forward_fn, q, k, v, bias)
        dq, dk, dv, dbias = vjp_fn(da)
        return dq, dk, dv, dbias


fwd_p.def_abstract_eval(triangle_attention_fwd_abstract_eval)
fwd_p.def_impl(partial(xla.apply_primitive, fwd_p))
for platform in ["cuda", None]:
    mlir.register_lowering(
        fwd_p,
        mlir.lower_fun(
            partial(triangle_attention_fwd_impl, platform), fwd_p.multiple_results
        ),
        platform,
    )

bwd_p.def_abstract_eval(triangle_attention_bwd_abstract_eval)
bwd_p.def_impl(partial(xla.apply_primitive, bwd_p))
for platform in ["cuda", None]:
    mlir.register_lowering(
        bwd_p,
        mlir.lower_fun(
            partial(triangle_attention_bwd_impl, platform), bwd_p.multiple_results
        ),
        platform,
    )

batching.primitive_batchers[fwd_p] = partial(
    naive_batching_rule, fwd_p, (0, 0, 0, 0, 0), (0, 0, 0)
)
batching.primitive_batchers[bwd_p] = partial(
    naive_batching_rule, bwd_p, (0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0)
)

# Custom VJP:


@partial(custom_vjp, nondiff_argnums=(5, 6))
def triangle_attention_custom_vjp(
    q: jax.Array,  # [B, N, H, S_qo, D]
    k: jax.Array,  # [B, N, H, S_kv, D]
    v: jax.Array,  # [B, N, H, S_kv, D]
    bias: jax.Array,  # [B, 1, H, S_qo, S_kv]
    mask: jax.Array,  # [B, N, 1, 1, S_kv] boolean
    scale: float,
    precision: jax.lax.Precision | None = None,
):
    return fwd_p.bind(q, k, v, bias, mask, scale=scale, precision=precision)


def triangle_attention_custom_vjp_fwd(q, k, v, bias, mask, scale, precision=None):
    a, lse, amax = fwd_p.bind(q, k, v, bias, mask, scale=scale, precision=precision)
    residuals = (a, lse, q, k, v, bias, mask)
    return (a, lse, amax), residuals


def triangle_attention_custom_vjp_bwd(scale, precision, residuals, cotangents):
    a, lse, q, k, v, bias, mask = residuals
    da, dlse, damax = cotangents

    dq, dk, dv, dbias = bwd_p.bind(
        da, a, lse, q, k, v, bias, mask, scale=scale, precision=precision
    )
    return (dq, dk, dv, dbias, None)


triangle_attention_custom_vjp.defvjp(
    triangle_attention_custom_vjp_fwd, triangle_attention_custom_vjp_bwd
)
