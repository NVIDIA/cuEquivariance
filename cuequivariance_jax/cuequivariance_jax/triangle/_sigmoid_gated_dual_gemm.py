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

import enum
import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.interpreters import mlir, xla

from cuequivariance_jax.benchmarking import measure_clock_ticks

try:
    import jax_triton as jt
    import triton

    HAS_JAX_TRITON = True
except ImportError:
    HAS_JAX_TRITON = False


class BenchmarkMode(enum.Enum):
    FLUSH_CACHE = 0
    FLUSH_CACHE_PEAK_PROXY = 1
    ROT_BUFFER = 2
    ROT_BUFFER_PEAK_PROXY = 3


# Precision modes matching cuequivariance_ops
class Precision(enum.IntEnum):
    DEFAULT = 0
    TF32 = 1
    TF32x3 = 2
    IEEE = 3

    def _to_jax(self):
        """Convert Precision enum to JAX precision."""
        if self == Precision.DEFAULT:
            return jax.lax.Precision.DEFAULT
        elif self == Precision.TF32:
            return jax.lax.Precision.HIGH
        elif self == Precision.TF32x3:
            return jax.lax.Precision.HIGHEST
        elif self == Precision.IEEE:
            return jax.lax.Precision.HIGHEST


# Unified JAX primitives
sigmoid_gated_dual_gemm_fwd_p = jax.extend.core.Primitive("sigmoid_gated_dual_gemm_fwd")
sigmoid_gated_dual_gemm_bwd_p = jax.extend.core.Primitive("sigmoid_gated_dual_gemm_bwd")
sigmoid_gated_dual_gemm_bwd_p.multiple_results = True


def _abstract_eval_fwd(
    x1, x2, w1, w2, mask, *, two_inputs, transpose_out, precision, fallback
):
    """Abstract evaluation for forward pass."""
    M, N = x1.shape[0], w1.shape[0]
    out_shape = (N, M) if transpose_out else (M, N)
    return jax.core.ShapedArray(out_shape, x1.dtype)


def _abstract_eval_bwd(
    grad_out, x1, x2, w1, w2, mask, *, two_inputs, transpose_out, precision, fallback
):
    """Abstract evaluation for backward pass."""
    return (
        jax.core.ShapedArray(x1.shape, x1.dtype),  # grad_x1
        jax.core.ShapedArray(x2.shape, x2.dtype),  # grad_x2
        jax.core.ShapedArray(w1.shape, w1.dtype),  # grad_w1
        jax.core.ShapedArray(w2.shape, w2.dtype),  # grad_w2
        jax.core.ShapedArray(mask.shape, mask.dtype),  # grad_mask
    )


def _reference_forward(x1, x2, w1, w2, mask, two_inputs, transpose_out, precision):
    """Pure JAX reference implementation."""
    # x1: (M, K)
    # x2: (M, K)
    # w1: (N, K)
    # w2: (N, K)
    # mask: (M,) or None
    # returns: (M, N) or (N, M) if transpose_out=True
    precision = precision._to_jax()
    if two_inputs:
        acc_1 = jnp.dot(x1, w1.T, precision=precision)
        acc_2 = jnp.dot(x2, w2.T, precision=precision)
    else:
        acc_1 = jnp.dot(x1, w1.T, precision=precision)
        acc_2 = jnp.dot(x1, w2.T, precision=precision)

    output = jax.nn.sigmoid(acc_1) * acc_2

    if mask is not None:
        output = output * mask[:, None]

    output = output.astype(x1.dtype)

    return output.T if transpose_out else output


def _triton_forward(
    x1,
    x2,
    w1,
    w2,
    mask,
    *,
    two_inputs,
    transpose_out,
    precision,
    TILE_M=64,
    TILE_N=32,
    TILE_K=32,
    num_stages=4,
    num_warps=4,
):
    """Triton implementation of forward pass."""
    if not HAS_JAX_TRITON:
        raise ImportError("jax_triton is required for GPU implementation")

    from cuequivariance_ops.triton import fused_sigmoid_gated_dual_gemm_forward_kernel

    dtype = x1.dtype
    assert dtype != jnp.float64
    x1 = x1.astype(dtype)
    if x2 is not None:
        x2 = x2.astype(dtype)
    w1 = w1.astype(dtype)
    w2 = w2.astype(dtype)
    if mask is not None:
        mask = mask.astype(dtype)

    M, K, N = x1.shape[0], x1.shape[1], w1.shape[0]
    assert N % TILE_N == 0 and K % TILE_K == 0

    out_shape = (N, M) if transpose_out else (M, N)
    out_shapes = [jax.ShapeDtypeStruct(shape=out_shape, dtype=x1.dtype)]

    results = jt.triton_call(
        x1,
        x2,
        w1,
        w2,
        mask,
        M,
        N,
        K,
        kernel=fused_sigmoid_gated_dual_gemm_forward_kernel,
        out_shape=out_shapes,
        grid=(triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N), 1),
        num_stages=num_stages,
        num_warps=num_warps,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        PRECISION=precision,
        APPLY_MASK=mask is not None,
        TRANSPOSE_OUT=transpose_out,
        TWO_INPUTS=two_inputs,
    )

    return results[0]


def _triton_backward(
    grad_out,
    x1,
    x2,
    w1,
    w2,
    mask,
    two_inputs,
    transpose_out,
    precision,
    TILE_M=64,
    TILE_N=32,
    TILE_K=32,
    num_stages=4,
    num_warps=4,
):
    """Triton implementation of backward pass."""
    if not HAS_JAX_TRITON:
        raise ImportError("jax_triton is required for GPU implementation")

    from cuequivariance_ops.triton import (
        fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel,
    )

    dtype = x1.dtype
    assert dtype != jnp.float64
    grad_out = grad_out.astype(dtype)
    x1 = x1.astype(dtype)
    if x2 is not None:
        x2 = x2.astype(dtype)
    w1 = w1.astype(dtype)
    w2 = w2.astype(dtype)
    if mask is not None:
        mask = mask.astype(dtype)

    M, K, N = x1.shape[0], x1.shape[1], w1.shape[0]
    assert N % TILE_N == 0 and K % TILE_K == 0

    out_shapes = [
        jax.ShapeDtypeStruct(shape=(M, N), dtype=x1.dtype),  # grad_xw1
        jax.ShapeDtypeStruct(shape=(M, N), dtype=x1.dtype),  # grad_xw2
    ]
    if mask is not None:
        num_tiles_n = triton.cdiv(N, TILE_N)
        out_shapes.append(
            jax.ShapeDtypeStruct(shape=(num_tiles_n, M), dtype=mask.dtype)
        )

    results = jt.triton_call(
        grad_out,
        x1,
        x2,
        w1,
        w2,
        mask,
        M,
        N,
        K,
        kernel=fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel,
        out_shape=out_shapes,
        grid=(triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N), 1),
        num_stages=num_stages,
        num_warps=num_warps,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        PRECISION=precision,
        APPLY_MASK=mask is not None,
        TRANSPOSE_OUT=transpose_out,
        TWO_INPUTS=two_inputs,
    )

    grad_xw1, grad_xw2 = results[0], results[1]
    grad_mask = results[2] if len(results) > 2 else None

    precision = precision._to_jax()
    grad_w1 = jnp.dot(grad_xw1.T, x1, precision=precision)
    grad_x1 = jnp.dot(grad_xw1, w1, precision=precision)
    if two_inputs:
        grad_w2 = jnp.dot(grad_xw2.T, x2, precision=precision)
        grad_x2 = jnp.dot(grad_xw2, w2, precision=precision)
    else:
        grad_w2 = jnp.dot(grad_xw2.T, x1, precision=precision)
        grad_x1 += jnp.dot(grad_xw2, w2, precision=precision)
        grad_x2 = jnp.zeros_like(x2)

    if grad_mask is not None:
        grad_mask = jnp.sum(grad_mask, axis=0)
    else:
        grad_mask = jnp.zeros(x1.shape[0], dtype=x1.dtype)

    return grad_x1, grad_x2, grad_w1, grad_w2, grad_mask


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
        kwargs = {
            k: jax.random.normal(jax.random.key(i), v.shape, dtype=v.dtype)
            if isinstance(v, jax.Array)
            else v
            for i, (k, v) in enumerate(input_dict.items())
        }
        return measure_clock_ticks(f, **kwargs)


def _generate_inputs(
    M,
    N,
    K,
    dtype_input,
    two_inputs,
    precision,
    include_grad=False,
):
    """Generate inputs for kernel autotuning."""
    key = jax.random.key(42)
    keys = jax.random.split(key, 7 if include_grad else 6)

    inputs = {
        "x1": jax.random.normal(keys[0], (M, K), dtype=dtype_input),
        "x2": jax.random.normal(keys[1], (M, K), dtype=dtype_input)
        if two_inputs
        else None,
        "w1": jax.random.normal(keys[2], (N, K), dtype=dtype_input),
        "w2": jax.random.normal(keys[3], (N, K), dtype=dtype_input),
        "mask": jax.random.normal(keys[4], (M,), dtype=dtype_input),
        "two_inputs": two_inputs,
        "transpose_out": False,
        "precision": precision,
        "fallback": False,
    }

    if include_grad:
        inputs["grad_out"] = jax.random.normal(keys[5], (M, N), dtype=dtype_input)

    return inputs


def _input_to_key(x1, x2, w1, w2, mask, two_inputs, precision, **unused_kwargs):
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
    dtypes = [
        str(t.dtype if t.dtype != jnp.bfloat16 else jnp.dtype(jnp.float16))
        for t in [x1, x2, w1, w2, mask]
        if t is not None
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

    return f"{key_m}_{key_k}_{key_n}_{'_'.join(dtypes)}_{two_inputs}_{precision_key}"


def _get_autotuned_kernel(is_forward=True):
    """Get or create autotuned kernel."""
    global _autotuned_forward, _autotuned_backward
    from cuequivariance_ops.triton import autotune_aot

    if is_forward and _autotuned_forward is None:
        _autotuned_forward = autotune_aot(
            input_generator=lambda **k: _generate_inputs(**k),
            input_to_key=_input_to_key,
            input_configs=[
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
                for m in range(32, 2048, 32)
                for dt, p in [
                    (jnp.bfloat16, Precision.DEFAULT),
                    (jnp.float32, Precision.TF32),
                    (jnp.float32, Precision.TF32x3),
                ]
            ],
            tunable_configs=[
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
            ],
            prune_configs_fn=None,
            run_decoy=run_decoy,
            run_bench=run_bench,
        )(_triton_forward)

    if not is_forward and _autotuned_backward is None:
        _autotuned_backward = autotune_aot(
            input_generator=lambda **k: _generate_inputs(**k, include_grad=True),
            input_to_key=lambda grad_out, **k: _input_to_key(**k),
            input_configs=[
                {"M": m, "N": n, "K": 128, "dtype_input": dt, "two_inputs": ti}
                for n in (128, 256)
                for ti in (True, False)
                for m in range(32, 2048, 32)
                for dt in (jnp.bfloat16, jnp.float32)
            ],
            tunable_configs=[
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
            ],
            prune_configs_fn=None,
            run_decoy=run_decoy,
            run_bench=run_bench,
        )(_triton_backward)

    return _autotuned_forward if is_forward else _autotuned_backward


# Create autotuned kernel wrappers
_autotuned_forward = None
_autotuned_backward = None


def _impl_dispatcher(platform, is_forward, *args, **kwargs):
    """Implementation dispatcher."""
    fallback = kwargs.pop("fallback", False)

    if platform == "cuda" and not fallback:
        if is_forward:
            return _get_autotuned_kernel(True)(*args, **kwargs)
        else:
            return _get_autotuned_kernel(False)(*args, **kwargs)

    # Fallback to reference implementation
    if is_forward:
        return _reference_forward(*args, **kwargs)
    else:
        # Use JAX autodiff for backward pass
        grad_out = args[0]
        x1, x2, w1, w2, mask = args[1:]

        def forward_fn(x1, x2, w1, w2, mask):
            return _reference_forward(x1, x2, w1, w2, mask, **kwargs)

        _, vjp_fn = jax.vjp(forward_fn, x1, x2, w1, w2, mask)
        return vjp_fn(grad_out)


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
        mlir.lower_fun(partial(_impl_dispatcher, platform, True), False),
        platform,
    )
    mlir.register_lowering(
        sigmoid_gated_dual_gemm_bwd_p,
        mlir.lower_fun(partial(_impl_dispatcher, platform, False), True),
        platform,
    )


@partial(
    custom_vjp,
    nondiff_argnames=("two_inputs", "transpose_out", "precision", "fallback"),
)
def _sigmoid_gated_dual_gemm_core(
    x1, x2, w1, w2, mask, two_inputs, transpose_out, precision, fallback=False
):
    """Core implementation with custom VJP."""
    if isinstance(precision, int):
        precision = Precision(precision)

    if mask is None:
        mask = jnp.ones(x1.shape[0], dtype=x1.dtype)

    return sigmoid_gated_dual_gemm_fwd_p.bind(
        x1,
        x2,
        w1,
        w2,
        mask,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
    )


def _fwd(x1, x2, w1, w2, mask, two_inputs, transpose_out, precision, fallback):
    original_mask = mask
    if mask is None:
        mask = jnp.ones(x1.shape[0], dtype=x1.dtype)

    result = sigmoid_gated_dual_gemm_fwd_p.bind(
        x1,
        x2,
        w1,
        w2,
        mask,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
    )

    return result, (x1, x2, w1, w2, original_mask)


def _bwd(two_inputs, transpose_out, precision, fallback, residuals, grad_out):
    x1, x2, w1, w2, original_mask = residuals
    mask = (
        original_mask
        if original_mask is not None
        else jnp.ones(x1.shape[0], dtype=x1.dtype)
    )

    grads = sigmoid_gated_dual_gemm_bwd_p.bind(
        grad_out,
        x1,
        x2,
        w1,
        w2,
        mask,
        two_inputs=two_inputs,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
    )

    # grads is always (grad_x1, grad_x2, grad_w1, grad_w2, grad_mask)
    grad_x1, grad_x2, grad_w1, grad_w2, grad_mask = grads

    if original_mask is None:
        # Replace mask gradient with zeros
        grad_mask = jnp.zeros_like(mask)

    # Always return 5 gradients to match the core function signature
    return (grad_x1, grad_x2, grad_w1, grad_w2, grad_mask)


_sigmoid_gated_dual_gemm_core.defvjp(_fwd, _bwd)


def _prepare_inputs(x, w1, w2, mask=None):
    """Prepare inputs and handle reshaping."""
    x = jnp.asarray(x)
    w1 = jnp.asarray(w1)
    w2 = jnp.asarray(w2)

    original_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])

    # Reshape mask to match the flattened x tensor if needed
    if mask is not None:
        mask = jnp.asarray(mask)
        if len(original_shape) > 2 and mask.ndim > 1:
            # If x was reshaped from (..., K) to (M, K), then mask should be reshaped from (...,) to (M,)
            mask = mask.reshape(-1)

    return x, w1, w2, mask, original_shape


def _reshape_output(out, original_shape, w1_shape, transpose_out):
    """Reshape output back to original batch dimensions."""
    if len(original_shape) > 2:
        if transpose_out:
            out_shape = (w1_shape[0], *original_shape[:-1])
        else:
            out_shape = (*original_shape[:-1], w1_shape[0])
        out = out.reshape(out_shape)
    return out


def sigmoid_gated_dual_gemm(
    x,
    w1,
    w2,
    mask: Optional[jnp.ndarray] = None,
    transpose_out: bool = False,
    precision: Precision = Precision.DEFAULT,
    fallback: bool = False,
):
    """Apply fused sigmoid-gated dual GEMM operation with single input.

    Performs: sigmoid(x @ w1) * (x @ w2) with optional masking.

    Args:
        x: Input tensor of shape (M, K) or (..., K)
        w1: First weight matrix of shape (N, K)
        w2: Second weight matrix of shape (N, K)
        mask: Optional mask tensor of shape (M,) or (...,)
        transpose_out: Whether to transpose the output
        precision: Precision mode for matrix multiplication
        fallback: Whether to force fallback to reference implementation

    Returns:
        Output tensor of shape (M, N) or (..., N) if transpose_out=False,
        (N, M) or (N, ...) if transpose_out=True
    """
    x, w1, w2, mask, original_shape = _prepare_inputs(x, w1, w2, mask)
    x2 = jnp.zeros_like(x)  # dummy x2 for single input mode

    out = _sigmoid_gated_dual_gemm_core(
        x,
        x2,
        w1,
        w2,
        mask,
        two_inputs=False,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
    )

    return _reshape_output(out, original_shape, w1.shape, transpose_out)


def sigmoid_gated_dual_gemm_dual_x(
    x1,
    x2,
    w1,
    w2,
    mask: Optional[jnp.ndarray] = None,
    transpose_out: bool = False,
    precision: Precision = Precision.DEFAULT,
    fallback: bool = False,
):
    """Apply fused sigmoid-gated dual GEMM operation with two inputs.

    Performs: sigmoid(x1 @ w1) * (x2 @ w2) with optional masking.

    Args:
        x1: First input tensor of shape (M, K) or (..., K)
        x2: Second input tensor of shape (M, K) or (..., K)
        w1: First weight matrix of shape (N, K)
        w2: Second weight matrix of shape (N, K)
        mask: Optional mask tensor of shape (M,) or (...,)
        transpose_out: Whether to transpose the output
        precision: Precision mode for matrix multiplication
        fallback: Whether to force fallback to reference implementation

    Returns:
        Output tensor of shape (M, N) or (..., N) if transpose_out=False,
        (N, M) or (N, ...) if transpose_out=True
    """
    x1, w1, w2, mask, original_shape = _prepare_inputs(x1, w1, w2, mask)
    x2 = jnp.asarray(x2)
    if x2.ndim > 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    out = _sigmoid_gated_dual_gemm_core(
        x1,
        x2,
        w1,
        w2,
        mask,
        two_inputs=True,
        transpose_out=transpose_out,
        precision=precision,
        fallback=fallback,
    )

    return _reshape_output(out, original_shape, w1.shape, transpose_out)


# Export reference function for tests
def _sigmoid_gated_dual_gemm_reference(
    x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
):
    """Reference implementation for testing - matches original function signature."""
    return _reference_forward(
        x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
    )
