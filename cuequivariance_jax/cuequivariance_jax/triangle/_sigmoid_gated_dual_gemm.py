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
from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.interpreters import batching, mlir, xla

from cuequivariance_jax.benchmarking import measure_clock_ticks

from ._naive_batching import naive_batching_rule
from ._utils import Precision

try:
    import jax_triton as jt
    import triton

    HAS_JAX_TRITON = True
except ImportError:
    HAS_JAX_TRITON = False


# Unified JAX primitives
sigmoid_gated_dual_gemm_fwd_p = jax.extend.core.Primitive("sigmoid_gated_dual_gemm_fwd")
sigmoid_gated_dual_gemm_fwd_p.multiple_results = True
sigmoid_gated_dual_gemm_bwd_p = jax.extend.core.Primitive("sigmoid_gated_dual_gemm_bwd")
sigmoid_gated_dual_gemm_bwd_p.multiple_results = True

# Architecture Overview:
#
# This implementation uses a uniform approach to handle optional parameters (bias tensors and mask)
# while maintaining compatibility with JAX primitives and optimal performance with Triton kernels.

# Key Design Principles:
# 1. JAX primitives (.bind() calls) don't support None/optional arguments - they require concrete arrays
# 2. We convert None values to appropriate defaults (zeros for bias, ones for mask) early in _prepare_inputs()
# 3. We track the original None status with boolean flags (has_b1, has_b2, has_mask)
# 4. These flags are passed through the entire call chain to control actual computation
# 5. Triton kernels receive the flags to skip unnecessary operations, maintaining performance
#
# Benefits:
# - Clean separation: None handling is centralized in _prepare_inputs()
# - JAX compatibility: Primitives always receive valid arrays
# - Performance: Triton kernels can skip operations based on flags
# - Type safety: All functions have explicit type annotations
# - Maintainability: Uniform pattern across all implementation variants


def _abstract_eval_fwd(
    x1: jax.core.ShapedArray,  # (M, K)
    x2: jax.core.ShapedArray,  # (M, K)
    w1: jax.core.ShapedArray,  # (N, K)
    w2: jax.core.ShapedArray,  # (N, K)
    b1: jax.core.ShapedArray,  # (N,)
    b2: jax.core.ShapedArray,  # (N,)
    mask: jax.core.ShapedArray,  # (M,)
    *,
    transpose_out: bool,
    **unused_kwargs,
):
    """Abstract evaluation for forward pass."""
    M, N = x1.shape[0], w1.shape[0]
    out_shape = (N, M) if transpose_out else (M, N)
    return (
        jax.core.ShapedArray(out_shape, x1.dtype),  # (M, N) or (N, M) if transpose_out
    )


def _abstract_eval_bwd(
    grad_out: jax.core.ShapedArray,  # (M, N) or (N, M) if transpose_out
    x1: jax.core.ShapedArray,  # (M, K)
    x2: jax.core.ShapedArray,  # (M, K)
    w1: jax.core.ShapedArray,  # (N, K)
    w2: jax.core.ShapedArray,  # (N, K)
    b1: jax.core.ShapedArray,  # (N,)
    b2: jax.core.ShapedArray,  # (N,)
    mask: jax.core.ShapedArray,  # (M,)
    **unused_kwargs,
):
    """Abstract evaluation for backward pass.

    Returns intermediate gradients (matching Triton kernel output):
    - grad_xw1: (M, N) - gradient w.r.t. first projection
    - grad_xw2: (M, N) - gradient w.r.t. second projection
    - grad_mask: (M,) - gradient w.r.t. mask (summed)
    """
    M, N = x1.shape[0], w1.shape[0]
    return (
        jax.core.ShapedArray((M, N), grad_out.dtype),  # grad_xw1: (M, N)
        jax.core.ShapedArray((M, N), grad_out.dtype),  # grad_xw2: (M, N)
        jax.core.ShapedArray((M,), mask.dtype),  # grad_mask: (M,)
    )


def _reference_forward(
    x1: jax.Array,  # (M, K)
    x2: jax.Array,  # (M, K)
    w1: jax.Array,  # (N, K)
    w2: jax.Array,  # (N, K)
    b1: jax.Array,  # (N,)
    b2: jax.Array,  # (N,)
    mask: jax.Array,  # (M,)
    *,
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
):
    """Pure JAX reference implementation."""
    precision = precision._to_jax()
    if two_inputs:
        acc_1 = jnp.dot(x1, w1.T, precision=precision)
        acc_2 = jnp.dot(x2, w2.T, precision=precision)
    else:
        acc_1 = jnp.dot(x1, w1.T, precision=precision)
        acc_2 = jnp.dot(x1, w2.T, precision=precision)

    # Add bias (bias arrays are always provided, but has_b1/has_b2 determine if they should be applied)
    if has_b1:
        acc_1 = acc_1 + b1[None, :]
    if has_b2:
        acc_2 = acc_2 + b2[None, :]

    output = jax.nn.sigmoid(acc_1) * acc_2

    # Apply mask (mask array is always provided, but has_mask determines if it should be applied)
    if has_mask:
        output = output * mask[:, None]

    output = output.astype(x1.dtype)

    return output.T if transpose_out else output  # (M, N) or (N, M) if transpose_out


def _reference_backward(
    grad_out: jax.Array,  # (M, N) or (N, M) if transpose_out
    x1: jax.Array,  # (M, K)
    x2: jax.Array,  # (M, K)
    w1: jax.Array,  # (N, K)
    w2: jax.Array,  # (N, K)
    b1: jax.Array,  # (N,)
    b2: jax.Array,  # (N,)
    mask: jax.Array,  # (M,)
    *,
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
):
    """Pure JAX reference implementation of backward pass.

    Replicates the Triton backward kernel logic, returning intermediate gradients
    that match the kernel's output format.

    Returns:
        tuple: (grad_xw1, grad_xw2, grad_mask) where:
            - grad_xw1: shape (M, N) - gradient w.r.t. first projection
            - grad_xw2: shape (M, N) - gradient w.r.t. second projection
            - grad_mask: shape (M,) - gradient w.r.t. mask (summed)
    """

    precision = precision._to_jax()

    # Recompute forward pass to get intermediate values
    if two_inputs:
        acc_1 = jnp.dot(x1, w1.T, precision=precision)
        acc_2 = jnp.dot(x2, w2.T, precision=precision)
    else:
        acc_1 = jnp.dot(x1, w1.T, precision=precision)
        acc_2 = jnp.dot(x1, w2.T, precision=precision)

    # Add bias (bias arrays are always provided, but has_b1/has_b2 determine if they should be applied)
    if has_b1:
        acc_1 = acc_1 + b1[None, :]
    if has_b2:
        acc_2 = acc_2 + b2[None, :]

    # Compute sigmoid
    acc_sig = jax.nn.sigmoid(acc_1)  # (M, N)

    # Handle transposed gradient output
    if transpose_out:
        grad_out = grad_out.T  # Convert (N, M) -> (M, N)

    # Apply mask to gradient if present
    if has_mask:
        # Compute gradient w.r.t. mask: grad_mask = grad_out * (acc_sig * acc_2)
        grad_mask = jnp.sum(grad_out * (acc_sig * acc_2), axis=1)  # (M,)
        # Apply mask to gradient
        grad_out = grad_out * mask[:, None]  # (M, N)
    else:
        grad_mask = jnp.zeros(x1.shape[0], dtype=x1.dtype)  # (M,)

    # Compute intermediate gradients (matching Triton kernel output)
    grad_xw2 = grad_out * acc_sig  # (M, N)
    grad_xw1 = grad_out * acc_2 * acc_sig * (1.0 - acc_sig)  # (M, N)

    return grad_xw1, grad_xw2, grad_mask  # (M, N), (M, N), (M,)


def fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper(
    x1: jax.Array,
    x2: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    b1: jax.Array,
    b2: jax.Array,
    mask: jax.Array,
    *,
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
    TILE_M: int = 64,
    TILE_N: int = 32,
    TILE_K: int = 32,
    num_stages: int = 4,
    num_warps: int = 4,
):
    """Triton implementation of forward pass."""
    if not HAS_JAX_TRITON:
        raise ImportError("jax_triton is required for GPU implementation")

    from cuequivariance_ops.triton import fused_sigmoid_gated_dual_gemm_forward_kernel

    dtype = x1.dtype
    assert dtype != jnp.float64
    # All inputs are guaranteed to be arrays at this point
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    w1 = w1.astype(dtype)
    w2 = w2.astype(dtype)
    b1 = b1.astype(dtype)
    b2 = b2.astype(dtype)
    mask = mask.astype(dtype)

    M, K, N = x1.shape[0], x1.shape[1], w1.shape[0]
    assert N % TILE_N == 0 and K % TILE_K == 0

    NEEDS_INT64 = (M * K >= 2**31 - 1) or (M * N >= 2**31 - 1)

    out_shape = (N, M) if transpose_out else (M, N)
    dummy = jnp.zeros((), dtype=dtype)

    return jt.triton_call(
        x1,
        x2 if two_inputs else dummy,
        w1,
        w2,
        b1 if has_b1 else dummy,
        b2 if has_b2 else dummy,
        mask if has_mask else dummy,
        M,
        N,
        K,
        kernel=fused_sigmoid_gated_dual_gemm_forward_kernel,
        out_shape=[jax.ShapeDtypeStruct(shape=out_shape, dtype=x1.dtype)],
        grid=(triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N), 1),
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        PRECISION=precision,
        APPLY_MASK=has_mask,
        TRANSPOSE_OUT=transpose_out,
        TWO_INPUTS=two_inputs,
        HAS_B1=has_b1,
        HAS_B2=has_b2,
        NEEDS_INT64=NEEDS_INT64,
        num_stages=num_stages,
        num_warps=num_warps,
    )


def fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel_wrapper(
    grad_out: jax.Array,
    x1: jax.Array,
    x2: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    b1: jax.Array,
    b2: jax.Array,
    mask: jax.Array,
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
    TILE_M: int = 64,
    TILE_N: int = 32,
    TILE_K: int = 32,
    num_stages: int = 4,
    num_warps: int = 4,
):
    """Triton implementation of backward pass."""
    if not HAS_JAX_TRITON:
        raise ImportError("jax_triton is required for GPU implementation")

    from cuequivariance_ops.triton import (
        fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel,
    )

    dtype = x1.dtype
    assert dtype != jnp.float64
    # All inputs are guaranteed to be arrays at this point
    grad_out = grad_out.astype(dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    w1 = w1.astype(dtype)
    w2 = w2.astype(dtype)
    b1 = b1.astype(dtype)
    b2 = b2.astype(dtype)
    mask = mask.astype(dtype)

    M, K, N = x1.shape[0], x1.shape[1], w1.shape[0]
    assert N % TILE_N == 0 and K % TILE_K == 0

    NEEDS_INT64 = (M * K >= 2**31 - 1) or (M * N >= 2**31 - 1)

    num_tiles_n = triton.cdiv(N, TILE_N)
    out_shapes = [
        jax.ShapeDtypeStruct(shape=(M, N), dtype=x1.dtype),  # grad_xw1
        jax.ShapeDtypeStruct(shape=(M, N), dtype=x1.dtype),  # grad_xw2
        jax.ShapeDtypeStruct(shape=(num_tiles_n, M), dtype=mask.dtype),
    ]

    dummy = jnp.zeros((), dtype=dtype)
    grad_xw1, grad_xw2, grad_mask = jt.triton_call(
        grad_out,
        x1,
        x2 if two_inputs else dummy,
        w1,
        w2,
        b1 if has_b1 else dummy,
        b2 if has_b2 else dummy,
        mask if has_mask else dummy,
        M,
        N,
        K,
        kernel=fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel,
        out_shape=out_shapes,
        grid=(triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N), 1),
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        APPLY_MASK=has_mask,
        TRANSPOSE_OUT=transpose_out,
        PRECISION=precision,
        TWO_INPUTS=two_inputs,
        NEEDS_INT64=NEEDS_INT64,
        num_stages=num_stages,
        num_warps=num_warps,
        HAS_B1=has_b1,
        HAS_B2=has_b2,
    )
    return grad_xw1, grad_xw2, jnp.sum(grad_mask, axis=0)


def run_decoy(f, input_dict):
    with jax.ensure_compile_time_eval():
        f(
            **{
                k: jnp.zeros_like(v) if isinstance(v, jax.Array) else v
                for k, v in input_dict.items()
            }
        )


def run_bench(f, input_dict):
    with jax.ensure_compile_time_eval():
        arrays = {
            k: jax.random.normal(jax.random.key(i), v.shape, dtype=v.dtype)
            for i, (k, v) in enumerate(input_dict.items())
            if isinstance(v, jax.Array)
        }
        options = {k: v for k, v in input_dict.items() if not isinstance(v, jax.Array)}
        rate, time = measure_clock_ticks(lambda **kw: f(**kw, **options), **arrays)
        return rate * time


def _generate_inputs(
    M: int,
    N: int,
    K: int,
    dtype_input: jnp.dtype,
    two_inputs: bool,
    precision: Precision,
    include_grad: bool,
):
    """Generate inputs for kernel autotuning."""
    key = jax.random.key(42)
    keys = jax.random.split(key, 8)

    inputs = {
        "x1": jax.random.normal(keys[0], (M, K), dtype=dtype_input),
        "x2": jax.random.normal(keys[1], (M, K), dtype=dtype_input),
        "w1": jax.random.normal(keys[2], (N, K), dtype=dtype_input),
        "w2": jax.random.normal(keys[3], (N, K), dtype=dtype_input),
        "b1": jax.random.normal(keys[4], (N,), dtype=dtype_input),
        "b2": jax.random.normal(keys[5], (N,), dtype=dtype_input),
        "mask": jax.random.normal(keys[6], (M,), dtype=dtype_input),
        "two_inputs": two_inputs,
        "transpose_out": False,
        "precision": precision,
        "has_b1": False,
        "has_b2": False,
        "has_mask": True,
    }
    if include_grad:
        inputs["grad_out"] = jax.random.normal(keys[7], (M, N), dtype=dtype_input)
    return inputs


def _input_to_key(
    x1: jax.Array,
    x2: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    b1: jax.Array,
    b2: jax.Array,
    mask: jax.Array,
    two_inputs: bool,
    precision: Precision,
    grad_out: jax.Array | None = None,
    **unused_kwargs,
):
    """Generate cache key from inputs."""
    M, K, N = x1.shape[0], x1.shape[1], w1.shape[0]

    # round mantissa
    def fn(x):
        a = math.floor(math.log2(x))
        x = x / 2**a
        n = 64
        x = round(x * n) / n
        return int(x * 2**a)

    assert (fn(1000), fn(1006), fn(8000), fn(8033)) == (1000, 1008, 8000, 8064)
    key_m, key_k, key_n = fn(M), fn(K), fn(N)

    # Normalize dtypes
    arrays = [x1, x2, w1, w2, mask]
    if grad_out is not None:
        arrays = [grad_out] + arrays
    dtypes = [
        "torch." + str(t.dtype if t.dtype != jnp.bfloat16 else jnp.dtype(jnp.float16))
        for t in arrays
    ]

    match precision:
        case Precision.TF32:
            precision_key = "tf32"
        case Precision.TF32x3:
            precision_key = "tf32x3"
        case Precision.IEEE:
            precision_key = "ieee"
        case _:
            precision_key = "default"

    return f"{key_m}_{key_n}_{key_k}_{'_'.join(dtypes)}_{two_inputs}_False_False_{precision_key}"


def _get_autotuned_kernel(is_forward: bool):
    """Get or create autotuned kernel."""
    global _autotuned_forward, _autotuned_backward
    from cuequivariance_ops.triton import autotune_aot

    input_configs = [
        {
            "M": m,
            "N": n,
            "K": 128,
            "dtype_input": dt,
            "two_inputs": ti,
            "precision": p,
        }
        for n in (128, 256)
        for ti in (True, False)
        for m in range(32, 2048 * 2048 + 1, 32)
        for dt, p in [
            (jnp.bfloat16, Precision.DEFAULT),
            (jnp.float32, Precision.TF32),
            (jnp.float32, Precision.TF32x3),
        ]
    ]
    tunable_configs = [
        {
            "TILE_M": tm,
            "TILE_N": tn,
            "TILE_K": tk,
            "num_stages": ns,
            "num_warps": nw,
        }
        for tm in (64, 128)
        for tn in (32, 64, 128)
        for tk in (16, 32, 64)
        for ns in (3, 4)
        for nw in (4, 8)
    ]

    if is_forward and _autotuned_forward is None:
        _autotuned_forward = autotune_aot(
            input_generator=lambda **k: _generate_inputs(**k, include_grad=False),
            input_to_key=_input_to_key,
            input_configs=input_configs,
            tunable_configs=tunable_configs,
            prune_configs_fn=None,
            run_decoy=run_decoy,
            run_bench=run_bench,
        )(fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper)

    if not is_forward and _autotuned_backward is None:
        _autotuned_backward = autotune_aot(
            input_generator=lambda **k: _generate_inputs(**k, include_grad=True),
            input_to_key=lambda grad_out, **k: _input_to_key(**k, grad_out=grad_out),
            input_configs=input_configs,
            tunable_configs=tunable_configs,
            prune_configs_fn=None,
            run_decoy=run_decoy,
            run_bench=run_bench,
        )(fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel_wrapper)

    return _autotuned_forward if is_forward else _autotuned_backward


# Create autotuned kernel wrappers
_autotuned_forward = None
_autotuned_backward = None


def _impl_dispatcher(platform: str | None, is_forward: bool, *args, **kwargs):
    """Implementation dispatcher."""
    fallback = kwargs.pop("fallback", False)

    if platform == "cuda" and not fallback:
        return _get_autotuned_kernel(is_forward)(*args, **kwargs)

    # Fallback to reference implementation
    if is_forward:
        return (_reference_forward(*args, **kwargs),)
    else:
        return _reference_backward(*args, **kwargs)


# Register primitives
sigmoid_gated_dual_gemm_fwd_p.def_abstract_eval(_abstract_eval_fwd)
sigmoid_gated_dual_gemm_fwd_p.def_impl(
    partial(xla.apply_primitive, sigmoid_gated_dual_gemm_fwd_p)
)
sigmoid_gated_dual_gemm_bwd_p.def_abstract_eval(_abstract_eval_bwd)
sigmoid_gated_dual_gemm_bwd_p.def_impl(
    partial(xla.apply_primitive, sigmoid_gated_dual_gemm_bwd_p)
)

# Register lowering for both platforms
for platform in ["cuda", None]:
    mlir.register_lowering(
        sigmoid_gated_dual_gemm_fwd_p,
        mlir.lower_fun(partial(_impl_dispatcher, platform, True), True),
        platform,
    )
    mlir.register_lowering(
        sigmoid_gated_dual_gemm_bwd_p,
        mlir.lower_fun(partial(_impl_dispatcher, platform, False), True),
        platform,
    )


def _sigmoid_gated_dual_gemm_fwd_batching_rule(
    batched_inputs: tuple[jax.Array, ...],
    vmapped_axes: tuple[int | None, ...],
    *,
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    fallback: bool,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    """Batching rule for sigmoid gated dual GEMM forward pass.

    Assumes batch axis is the M axis (axis 0 for x1, x2, mask).
    """
    # Input batch axes: (x1, x2, w1, w2, b1, b2, mask)
    # x1: (M, K) -> batch on axis 0
    # x2: (M, K) -> batch on axis 0
    # w1: (N, K) -> not batched (None)
    # w2: (N, K) -> not batched (None)
    # b1: (N,) -> not batched (None)
    # b2: (N,) -> not batched (None)
    # mask: (M,) -> batch on axis 0
    input_batch_axes = (0, 0, None, None, None, None, 0)

    # Output batch axes: (out,)
    # out: (M, N) -> batch on axis 0 if not transpose_out
    # out: (N, M) -> batch on axis 1 if transpose_out
    output_batch_axes = (1 if transpose_out else 0,)

    return naive_batching_rule(
        sigmoid_gated_dual_gemm_fwd_p,
        input_batch_axes,
        output_batch_axes,
        batched_inputs,
        vmapped_axes,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
        has_b1=has_b1,
        has_b2=has_b2,
        has_mask=has_mask,
    )


def _sigmoid_gated_dual_gemm_bwd_batching_rule(
    batched_inputs: tuple[jax.Array, ...],
    vmapped_axes: tuple[int | None, ...],
    *,
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    fallback: bool,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    """Batching rule for sigmoid gated dual GEMM backward pass.

    Assumes batch axis is the M axis (axis 0 for x1, x2, mask, and axis 0/1 for grad_out).
    """
    # Input batch axes: (grad_out, x1, x2, w1, w2, b1, b2, mask)
    # grad_out: (M, N) or (N, M) if transpose_out -> batch on axis 0 if not transpose_out, axis 1 if transpose_out
    # x1: (M, K) -> batch on axis 0
    # x2: (M, K) -> batch on axis 0
    # w1: (N, K) -> not batched (None)
    # w2: (N, K) -> not batched (None)
    # b1: (N,) -> not batched (None)
    # b2: (N,) -> not batched (None)
    # mask: (M,) -> batch on axis 0
    input_batch_axes = (1 if transpose_out else 0, 0, 0, None, None, None, None, 0)

    # Output batch axes: (grad_xw1, grad_xw2, grad_mask)
    # grad_xw1: (M, N) -> batch on axis 0
    # grad_xw2: (M, N) -> batch on axis 0
    # grad_mask: (M,) -> batch on axis 0
    output_batch_axes = (0, 0, 0)

    return naive_batching_rule(
        sigmoid_gated_dual_gemm_bwd_p,
        input_batch_axes,
        output_batch_axes,
        batched_inputs,
        vmapped_axes,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
        has_b1=has_b1,
        has_b2=has_b2,
        has_mask=has_mask,
    )


# Register batching rules
batching.primitive_batchers[sigmoid_gated_dual_gemm_fwd_p] = (
    _sigmoid_gated_dual_gemm_fwd_batching_rule
)
batching.primitive_batchers[sigmoid_gated_dual_gemm_bwd_p] = (
    _sigmoid_gated_dual_gemm_bwd_batching_rule
)


@partial(custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _sigmoid_gated_dual_gemm_core(
    x1: jax.Array,
    x2: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    b1: jax.Array,
    b2: jax.Array,
    mask: jax.Array,
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    fallback: bool,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
):
    """Core implementation with custom VJP."""
    if isinstance(precision, int):
        precision = Precision(precision)

    # All inputs are guaranteed to be arrays at this point
    (out,) = sigmoid_gated_dual_gemm_fwd_p.bind(
        x1,
        x2,
        w1,
        w2,
        b1,
        b2,
        mask,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
        has_b1=has_b1,
        has_b2=has_b2,
        has_mask=has_mask,
    )
    return out


def _fwd(
    x1: jax.Array,
    x2: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    b1: jax.Array,
    b2: jax.Array,
    mask: jax.Array,
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    fallback: bool,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
):
    # All inputs are guaranteed to be arrays at this point
    (out,) = sigmoid_gated_dual_gemm_fwd_p.bind(
        x1,
        x2,
        w1,
        w2,
        b1,
        b2,
        mask,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
        has_b1=has_b1,
        has_b2=has_b2,
        has_mask=has_mask,
    )

    return out, (x1, x2, w1, w2, b1, b2, mask)


def _bwd(
    two_inputs: bool,
    transpose_out: bool,
    precision: Precision,
    fallback: bool,
    has_b1: bool,
    has_b2: bool,
    has_mask: bool,
    residuals,
    grad_out: jax.Array,
):
    x1, x2, w1, w2, b1, b2, mask = residuals

    # Get intermediate gradients from primitive
    grad_xw1, grad_xw2, grad_mask = sigmoid_gated_dual_gemm_bwd_p.bind(
        grad_out,
        x1,
        x2,
        w1,
        w2,
        b1,
        b2,
        mask,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
        has_b1=has_b1,
        has_b2=has_b2,
        has_mask=has_mask,
    )

    # Post-process intermediate gradients to compute final gradients
    precision_jax = precision._to_jax()

    # Compute weight gradients
    grad_w1 = jnp.dot(grad_xw1.T, x1, precision=precision_jax)
    grad_x1 = jnp.dot(grad_xw1, w1, precision=precision_jax)

    if two_inputs:
        grad_w2 = jnp.dot(grad_xw2.T, x2, precision=precision_jax)
        grad_x2 = jnp.dot(grad_xw2, w2, precision=precision_jax)
    else:
        grad_w2 = jnp.dot(grad_xw2.T, x1, precision=precision_jax)
        grad_x1 += jnp.dot(grad_xw2, w2, precision=precision_jax)
        grad_x2 = jnp.zeros_like(x2)

    # Compute bias gradients
    grad_b1 = jnp.sum(grad_xw1, axis=0)
    grad_b2 = jnp.sum(grad_xw2, axis=0)

    # Handle None bias gradients based on has_b1 and has_b2
    if not has_b1:
        grad_b1 = None
    if not has_b2:
        grad_b2 = None
    if not has_mask:
        grad_mask = None

    # Always return 7 gradients to match the core function signature
    return (grad_x1, grad_x2, grad_w1, grad_w2, grad_b1, grad_b2, grad_mask)


_sigmoid_gated_dual_gemm_core.defvjp(_fwd, _bwd)


def _prepare_inputs(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    b1: jax.Array | None = None,
    b2: jax.Array | None = None,
    mask: jax.Array | None = None,
):
    """Prepare inputs and handle reshaping."""
    x = jnp.asarray(x)
    w1 = jnp.asarray(w1)
    w2 = jnp.asarray(w2)

    # Track whether bias and mask were originally provided
    has_b1 = b1 is not None
    has_b2 = b2 is not None
    has_mask = mask is not None

    # Convert None bias to zero arrays
    N = w1.shape[0]
    if b1 is not None:
        b1 = jnp.asarray(b1)
    else:
        b1 = jnp.zeros(N, dtype=x.dtype)
    if b2 is not None:
        b2 = jnp.asarray(b2)
    else:
        b2 = jnp.zeros(N, dtype=x.dtype)

    # Convert None mask to ones array
    original_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])

    M = x.shape[0]
    if mask is not None:
        mask = jnp.asarray(mask)
        if len(original_shape) > 2 and mask.ndim > 1:
            # If x was reshaped from (..., K) to (M, K), then mask should be reshaped from (...,) to (M,)
            mask = mask.reshape(-1)
    else:
        mask = jnp.ones(M, dtype=x.dtype)

    return x, w1, w2, b1, b2, mask, original_shape, has_b1, has_b2, has_mask


def _reshape_output(
    out: jax.Array,
    original_shape: tuple[int, ...],
    w1_shape: tuple[int, ...],
    transpose_out: bool,
):
    """Reshape output back to original batch dimensions."""
    if len(original_shape) > 2:
        if transpose_out:
            out_shape = (w1_shape[0], *original_shape[:-1])
        else:
            out_shape = (*original_shape[:-1], w1_shape[0])
        out = out.reshape(out_shape)
    return out


def sigmoid_gated_dual_gemm(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    *,
    b1: jax.Array | None = None,
    b2: jax.Array | None = None,
    mask: jax.Array | None = None,
    transpose_out: bool = False,
    precision: Precision = Precision.DEFAULT,
    fallback: bool = False,
) -> jax.Array:
    """Apply fused sigmoid-gated dual GEMM operation with single input.

    Performs: sigmoid(x @ w1 + b1) * (x @ w2 + b2) with optional masking.

    Args:
        x: Input tensor of shape (M, K) or (..., K)
        w1: First weight matrix of shape (N, K)
        w2: Second weight matrix of shape (N, K)
        b1: Optional bias tensor for first projection of shape (N,)
        b2: Optional bias tensor for second projection of shape (N,)
        mask: Optional mask tensor of shape (M,) or (...,)
        transpose_out: Whether to transpose the output
        precision: Precision mode for matrix multiplication
        fallback: Whether to force fallback to reference implementation

    Returns:
        Output tensor of shape (M, N) or (..., N) if transpose_out=False,
        (N, M) or (N, ...) if transpose_out=True
    """
    x, w1, w2, b1, b2, mask, original_shape, has_b1, has_b2, has_mask = _prepare_inputs(
        x, w1, w2, b1, b2, mask
    )
    x2 = jnp.zeros_like(x)  # dummy x2 for single input mode

    out = _sigmoid_gated_dual_gemm_core(
        x,
        x2,
        w1,
        w2,
        b1,
        b2,
        mask,
        two_inputs=False,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
        has_b1=has_b1,
        has_b2=has_b2,
        has_mask=has_mask,
    )

    return _reshape_output(out, original_shape, w1.shape, transpose_out)


def sigmoid_gated_dual_gemm_dual_x(
    x1: jax.Array,
    x2: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    b1: jax.Array | None = None,
    b2: jax.Array | None = None,
    mask: jax.Array | None = None,
    transpose_out: bool = False,
    precision: Precision = Precision.DEFAULT,
    fallback: bool = False,
) -> jax.Array:
    """Apply fused sigmoid-gated dual GEMM operation with two inputs.

    Performs: sigmoid(x1 @ w1 + b1) * (x2 @ w2 + b2) with optional masking.

    Args:
        x1: First input tensor of shape (M, K) or (..., K)
        x2: Second input tensor of shape (M, K) or (..., K)
        w1: First weight matrix of shape (N, K)
        w2: Second weight matrix of shape (N, K)
        b1: Optional bias tensor for first projection of shape (N,)
        b2: Optional bias tensor for second projection of shape (N,)
        mask: Optional mask tensor of shape (M,) or (...,)
        transpose_out: Whether to transpose the output
        precision: Precision mode for matrix multiplication
        fallback: Whether to force fallback to reference implementation

    Returns:
        Output tensor of shape (M, N) or (..., N) if transpose_out=False,
        (N, M) or (N, ...) if transpose_out=True
    """
    x1, w1, w2, b1, b2, mask, original_shape, has_b1, has_b2, has_mask = (
        _prepare_inputs(x1, w1, w2, b1, b2, mask)
    )
    x2 = jnp.asarray(x2)
    if x2.ndim > 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    out = _sigmoid_gated_dual_gemm_core(
        x1,
        x2,
        w1,
        w2,
        b1,
        b2,
        mask,
        two_inputs=True,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
        has_b1=has_b1,
        has_b2=has_b2,
        has_mask=has_mask,
    )

    return _reshape_output(out, original_shape, w1.shape, transpose_out)


# Export reference function for tests
def sigmoid_gated_dual_gemm_reference(
    x1: jax.Array,
    x2: jax.Array | None,
    w1: jax.Array,
    w2: jax.Array,
    b1: jax.Array | None,
    b2: jax.Array | None,
    mask: jax.Array | None,
    *,
    transpose_out: bool,
    precision: Precision,
) -> jax.Array:
    """Reference implementation for testing - matches original function signature."""
    # For test calls, b1 and b2 can be None, so we determine has_b1 and has_b2
    has_b1 = b1 is not None
    has_b2 = b2 is not None
    has_mask = mask is not None
    two_inputs = x2 is not None

    # Convert None values to appropriate defaults for the reference implementation
    M = x1.shape[0]
    N = w1.shape[0]
    if b1 is None:
        b1 = jnp.zeros(N, dtype=x1.dtype)
    if b2 is None:
        b2 = jnp.zeros(N, dtype=x1.dtype)
    if mask is None:
        mask = jnp.ones(M, dtype=x1.dtype)
    if x2 is None:
        x2 = jnp.zeros_like(x1)

    return _reference_forward(
        x1,
        x2,
        w1,
        w2,
        b1,
        b2,
        mask,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        has_b1=has_b1,
        has_b2=has_b2,
        has_mask=has_mask,
    )
