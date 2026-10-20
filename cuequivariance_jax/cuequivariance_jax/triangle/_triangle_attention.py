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

try:
    from cuequivariance_ops_jax import TRIANGLE_ATTENTION_ACCEPTS_NO_MASK
except ImportError:  # absent, or a release that predates mask=None support
    TRIANGLE_ATTENTION_ACCEPTS_NO_MASK = False


def triangle_attention(
    q: jax.Array,  # [B, N, H, S_qo, D]
    k: jax.Array,  # [B, N, H, S_kv, D]
    v: jax.Array,  # [B, N, H, S_kv, D]
    bias: jax.Array,  # [B, 1, H, S_qo, S_kv]
    mask: jax.Array | None,  # [B, N, 1, 1, S_kv] boolean
    scale: float,
    precision: jax.lax.Precision | None = None,
    *,
    fast: bool = False,
):
    r"""triangle attention

    Args:
        q: Query tensor of shape [B, N, H, S_qo, D].
        k: Key tensor of shape [B, N, H, S_kv, D].
        v: Value tensor of shape [B, N, H, S_kv, D].
        bias: Bias tensor of shape [B, 1, H, S_qo, S_kv]. Cast to float32 for standard
            kernels; on Blackwell GPUs (sm100f), cast to match q/k/v dtype (bf16/fp16).
        mask: Mask tensor of shape [B, N, 1, 1, S_kv] (boolean, True means valid), or
            ``None`` for no masking. ``None`` is equivalent to an all-True mask without
            creating one, unless the installed cuequivariance_ops_jax predates
            ``None`` support.
        scale: Scaling factor for the dot product.
        precision: Precision for the computation (default is None).
        fast: Static precision choice for supported Rubin forward calls, with
            BF16 inputs and head dimension 16, 32, 64, or 128 at the default scale.
            ``False`` uses FP32 exponentials; ``True`` uses BF16 exponentials.
            Other routes ignore the arithmetic choice. ``True`` is inference-only
            and raises ``ValueError`` under differentiation; use ``False`` for
            training. FP64 forward validation uses output ``atol=rtol=0.03`` for
            Fast and ``0.01`` for Accurate on the actual kernel inputs, including
            bias rounded to BF16. The Fast LSE diagnostic compares ``lse / ln(2)``
            with ``atol=rtol=0.01``. Requires matching 0.14 backend packages when
            ``True``.

    Returns:
        A three-element list containing the attention output, log-sum-exp, and
        maximum value. On supported Rubin GPUs, an empty Boolean-mask row
        returns zero output and ``lse == max == -1e9``. LSE uses natural
        logarithms in both precision modes.

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

    .. note::
        On Blackwell GPUs (cc 10.0 or 10.3, cu13 builds), the sm100f kernel supports
        hidden_dim<=256 for forward and hidden_dim<=128 for backward passes (bf16/fp16
        only; hidden_dim must be divisible by 8). The sm100f forward kernel requires
        S_kv to be a multiple of 8, please pad if necessary.
        The kernel provides optimal performance for a "padding mask" consisting in
        (all True, followed by all False) in the last dimension.
    """
    if not isinstance(fast, bool):
        raise ValueError("triangle_attention: fast must be a static bool")
    if mask is None and HAS_CUE_OPS_JAX and not TRIANGLE_ATTENTION_ACCEPTS_NO_MASK:
        mask = jnp.ones((*k.shape[:2], 1, 1, k.shape[3]), dtype=jnp.bool_)
    return triangle_attention_custom_vjp(
        q, k, v, bias, mask, scale=scale, precision=precision, fast=fast
    )


@partial(jax.jit, static_argnames=("scale", "precision"))
def triangle_attention_jax_fwd(
    q: jax.Array,  # [B, N, H, S_qo, D]
    k: jax.Array,  # [B, N, H, S_kv, D]
    v: jax.Array,  # [B, N, H, S_kv, D]
    bias: jax.Array,  # [B, 1, H, S_qo, S_kv]
    mask: jax.Array | None,  # [B, N, 1, 1, S_kv] boolean
    scale: float,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:  # [B, N, H, S_qo, D]
    r"""JAX reference implementation for triangle attention."""
    dtype = q.dtype
    assert k.dtype == dtype and v.dtype == dtype

    q = scale * q
    a = jnp.einsum("...ai,...bi->...ab", q, k, precision=precision)
    a = a + bias
    if mask is not None:
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
    *mask: jax.core.ShapedArray,  # [B, N, 1, 1, S_kv] boolean, absent for no mask
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
    *mask: jax.core.ShapedArray,  # [B, N, 1, 1, S_kv] boolean, absent for no mask
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
    *mask_operands: jax.Array,  # [B, N, 1, 1, S_kv] boolean, absent for no mask
    scale: float,
    precision: jax.lax.Precision | None = None,
    fast: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mask = mask_operands[0] if mask_operands else None
    if platform == "cuda":
        assert HAS_CUE_OPS_JAX, (
            "Please install cuequivariance_ops_jax for CUDA support."
        )
        if fast:
            return triangle_attention_cuda_fwd(
                q, k, v, mask, bias, scale, precision, fast=True
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
    *mask_operands: jax.Array,  # [B, N, 1, 1, S_kv] boolean, absent for no mask
    scale: float,
    precision: jax.lax.Precision | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mask = mask_operands[0] if mask_operands else None
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


def _batching_rule(primitive, num_outputs, batched_inputs, vmapped_axes, **kwargs):
    # Every operand, including the optional mask, is batched along axis 0.
    return naive_batching_rule(
        primitive,
        (0,) * len(batched_inputs),
        (0,) * num_outputs,
        batched_inputs,
        vmapped_axes,
        **kwargs,
    )


batching.primitive_batchers[fwd_p] = partial(_batching_rule, fwd_p, 3)
batching.primitive_batchers[bwd_p] = partial(_batching_rule, bwd_p, 4)

# Custom VJP:


def _mask_operands(mask: jax.Array | None) -> tuple[jax.Array, ...]:
    """The primitives take the mask as an optional trailing operand."""
    return () if mask is None else (mask,)


@partial(custom_vjp, nondiff_argnums=(5, 6, 7))
def triangle_attention_custom_vjp(
    q: jax.Array,  # [B, N, H, S_qo, D]
    k: jax.Array,  # [B, N, H, S_kv, D]
    v: jax.Array,  # [B, N, H, S_kv, D]
    bias: jax.Array,  # [B, 1, H, S_qo, S_kv]
    mask: jax.Array | None,  # [B, N, 1, 1, S_kv] boolean
    scale: float,
    precision: jax.lax.Precision | None = None,
    fast: bool = False,
):
    return fwd_p.bind(
        q,
        k,
        v,
        bias,
        *_mask_operands(mask),
        scale=scale,
        precision=precision,
        fast=fast,
    )


def triangle_attention_custom_vjp_fwd(
    q, k, v, bias, mask, scale, precision=None, fast=False
):
    if fast:
        raise ValueError(
            "triangle_attention: fast=True is inference-only; use fast=False "
            "for differentiation"
        )
    a, lse, amax = fwd_p.bind(
        q,
        k,
        v,
        bias,
        *_mask_operands(mask),
        scale=scale,
        precision=precision,
        fast=fast,
    )
    residuals = (a, lse, q, k, v, bias, mask)
    # Match the primal output's pytree structure: `triangle_attention_custom_vjp`
    # returns `fwd_p.bind(...)`, and a multiple_results primitive's bind returns a
    # list. Returning a tuple here trips custom_vjp's fwd/primal structure check
    # on JAX versions that enforce exact tuple-vs-list equality (TypeError).
    return [a, lse, amax], residuals


def triangle_attention_custom_vjp_bwd(scale, precision, fast, residuals, cotangents):
    a, lse, q, k, v, bias, mask = residuals
    da, dlse, damax = cotangents

    dq, dk, dv, dbias = bwd_p.bind(
        da,
        a,
        lse,
        q,
        k,
        v,
        bias,
        *_mask_operands(mask),
        scale=scale,
        precision=precision,
    )
    return (dq, dk, dv, dbias, None)


triangle_attention_custom_vjp.defvjp(
    triangle_attention_custom_vjp_fwd, triangle_attention_custom_vjp_bwd
)
