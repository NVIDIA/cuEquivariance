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
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
from jax import custom_vjp
from jax.interpreters import mlir, xla


# Precision modes matching cuequivariance_ops
class Precision(enum.IntEnum):
    DEFAULT = 0
    TF32 = 1
    TF32x3 = 2
    IEEE = 3


# Unified JAX primitives
sigmoid_gated_dual_gemm_fwd_p = jax.extend.core.Primitive("sigmoid_gated_dual_gemm_fwd")
sigmoid_gated_dual_gemm_bwd_p = jax.extend.core.Primitive("sigmoid_gated_dual_gemm_bwd")
sigmoid_gated_dual_gemm_bwd_p.multiple_results = True


def _abstract_eval_fwd(x1, x2, w1, w2, mask, *, two_inputs, transpose_out, precision):
    """Abstract evaluation for forward pass."""
    M, N = x1.shape[0], w1.shape[0]
    out_shape = (N, M) if transpose_out else (M, N)
    return jax.core.ShapedArray(out_shape, x1.dtype)


def _abstract_eval_bwd(
    grad_out, x1, x2, w1, w2, mask, *, two_inputs, transpose_out, precision
):
    """Abstract evaluation for backward pass."""
    outputs = [
        jax.core.ShapedArray(x1.shape, x1.dtype),  # grad_x1
        jax.core.ShapedArray(w1.shape, w1.dtype),  # grad_w1
        jax.core.ShapedArray(w2.shape, w2.dtype),  # grad_w2
    ]
    if two_inputs:
        outputs.insert(1, jax.core.ShapedArray(x2.shape, x2.dtype))  # grad_x2
    outputs.append(jax.core.ShapedArray(mask.shape, mask.dtype))  # grad_mask
    return tuple(outputs)


def _reference_forward(x1, x2, w1, w2, mask, two_inputs, transpose_out, precision):
    """Pure JAX reference implementation."""
    if two_inputs:
        acc_1 = jnp.dot(x1, w1.T, precision=jax.lax.Precision.HIGHEST)
        acc_2 = jnp.dot(x2, w2.T, precision=jax.lax.Precision.HIGHEST)
    else:
        acc_1 = jnp.dot(x1, w1.T, precision=jax.lax.Precision.HIGHEST)
        acc_2 = jnp.dot(x1, w2.T, precision=jax.lax.Precision.HIGHEST)

    acc_sig = jax.nn.sigmoid(acc_1)
    output = acc_sig * acc_2

    if mask is not None:
        output = output * mask[:, None]

    return output.T if transpose_out else output


def _triton_forward(x1, x2, w1, w2, mask, two_inputs, transpose_out, precision):
    """Triton implementation of forward pass."""
    from cuequivariance_ops.triton import fused_sigmoid_gated_dual_gemm_forward_kernel

    M, K, N = x1.shape[0], x1.shape[1], w1.shape[0]
    TILE_M, TILE_N, TILE_K = 64, 32, 32

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
    grad_out, x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
):
    """Triton implementation of backward pass."""
    from cuequivariance_ops.triton import (
        fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel,
    )

    M, K, N = x1.shape[0], x1.shape[1], w1.shape[0]
    TILE_M, TILE_N, TILE_K = 64, 32, 32

    out_shapes = [
        jax.ShapeDtypeStruct(shape=(M, N), dtype=x1.dtype),  # grad_xw1
        jax.ShapeDtypeStruct(shape=(M, N), dtype=x1.dtype),  # grad_xw2
    ]
    if mask is not None:
        out_shapes.append(jax.ShapeDtypeStruct(shape=(N, M), dtype=mask.dtype))

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

    if two_inputs:
        grad_w1, grad_w2 = jnp.dot(grad_xw1.T, x1), jnp.dot(grad_xw2.T, x2)
        grad_x1, grad_x2 = jnp.dot(grad_xw1, w1), jnp.dot(grad_xw2, w2)
        result = [grad_x1, grad_x2, grad_w1, grad_w2]
    else:
        grad_w1, grad_w2 = jnp.dot(grad_xw1.T, x1), jnp.dot(grad_xw2.T, x1)
        grad_x1 = jnp.dot(grad_xw1, w1) + jnp.dot(grad_xw2, w2)
        result = [grad_x1, grad_w1, grad_w2]

    if grad_mask is not None:
        grad_mask = jnp.sum(grad_mask, axis=0)
    else:
        grad_mask = jnp.zeros(x1.shape[0], dtype=x1.dtype)
    result.append(grad_mask)

    return tuple(result)


def _impl_dispatcher(platform, is_forward, *args, **kwargs):
    """Implementation dispatcher."""

    if platform == "cuda":
        try:
            import cuequivariance_ops.triton  # noqa: F401

            if is_forward:
                return _triton_forward(*args, **kwargs)
            else:
                return _triton_backward(*args, **kwargs)
        except ImportError:
            pass

    # Fallback to reference implementation
    if is_forward:
        return _reference_forward(*args, **kwargs)
    else:
        # Use JAX autodiff for backward pass
        grad_out = args[0]
        forward_args = args[1:]

        def forward_fn(*fwd_args):
            return _reference_forward(*fwd_args, **kwargs)

        _, vjp_fn = jax.vjp(forward_fn, *forward_args)
        grad_outputs = vjp_fn(grad_out)

        return tuple(grad_outputs)


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


@partial(custom_vjp, nondiff_argnames=("two_inputs", "transpose_out", "precision"))
def _sigmoid_gated_dual_gemm_core(
    x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
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
    )


def _fwd(x1, x2, w1, w2, mask, two_inputs, transpose_out, precision):
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
    )

    return result, (x1, x2, w1, w2, original_mask)


def _bwd(two_inputs, transpose_out, precision, residuals, grad_out):
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
    )

    if original_mask is None:
        # Replace mask gradient with zeros
        grad_mask = jnp.zeros_like(mask)
        if two_inputs:
            grad_x1, grad_x2, grad_w1, grad_w2, _ = grads
            return (grad_x1, grad_x2, grad_w1, grad_w2, grad_mask)
        else:
            grad_x1, grad_w1, grad_w2, _ = grads
            # For single input mode, still need to return gradient for dummy x2
            grad_x2 = jnp.zeros_like(x2)
            return (grad_x1, grad_x2, grad_w1, grad_w2, grad_mask)

    # Always return 5 gradients to match the core function signature
    if two_inputs:
        return grads
    else:
        # For single input mode, insert zero gradient for dummy x2
        grad_x1, grad_w1, grad_w2, grad_mask = grads
        grad_x2 = jnp.zeros_like(x2)
        return (grad_x1, grad_x2, grad_w1, grad_w2, grad_mask)


_sigmoid_gated_dual_gemm_core.defvjp(_fwd, _bwd)


def _prepare_inputs(x, w1, w2):
    """Prepare inputs and handle reshaping."""
    x = jnp.asarray(x)
    w1 = jnp.asarray(w1)
    w2 = jnp.asarray(w2)

    original_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])

    return x, w1, w2, original_shape


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

    Returns:
        Output tensor of shape (M, N) or (..., N) if transpose_out=False,
        (N, M) or (N, ...) if transpose_out=True
    """
    x, w1, w2, original_shape = _prepare_inputs(x, w1, w2)
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

    Returns:
        Output tensor of shape (M, N) or (..., N) if transpose_out=False,
        (N, M) or (N, ...) if transpose_out=True
    """
    x1, w1, w2, original_shape = _prepare_inputs(x1, w1, w2)
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
    )

    return _reshape_output(out, original_shape, w1.shape, transpose_out)


# Export reference function for tests
def sigmoid_gated_dual_gemm_reference_forward(
    x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
):
    """Reference implementation for testing - matches original function signature."""
    return _reference_forward(
        x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
    )
