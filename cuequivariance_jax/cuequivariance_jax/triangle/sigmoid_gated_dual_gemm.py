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


# JAX primitives for single input mode
sigmoid_gated_dual_gemm_single_fwd_p = jax.extend.core.Primitive(
    "sigmoid_gated_dual_gemm_single_fwd"
)
sigmoid_gated_dual_gemm_single_bwd_p = jax.extend.core.Primitive(
    "sigmoid_gated_dual_gemm_single_bwd"
)
sigmoid_gated_dual_gemm_single_bwd_p.multiple_results = True

# JAX primitives for dual input mode
sigmoid_gated_dual_gemm_dual_fwd_p = jax.extend.core.Primitive(
    "sigmoid_gated_dual_gemm_dual_fwd"
)
sigmoid_gated_dual_gemm_dual_bwd_p = jax.extend.core.Primitive(
    "sigmoid_gated_dual_gemm_dual_bwd"
)
sigmoid_gated_dual_gemm_dual_bwd_p.multiple_results = True


def sigmoid_gated_dual_gemm_single_fwd_abstract_eval(
    x1, w1, w2, mask, *, transpose_out, precision
):
    """Abstract evaluation for single input forward pass."""
    M = x1.shape[0]
    N = w1.shape[0]

    if transpose_out:
        out_shape = (N, M)
    else:
        out_shape = (M, N)

    return jax.core.ShapedArray(out_shape, x1.dtype)


def sigmoid_gated_dual_gemm_single_bwd_abstract_eval(
    grad_out, x1, w1, w2, mask, *, transpose_out, precision
):
    """Abstract evaluation for single input backward pass."""
    outputs = [
        jax.core.ShapedArray(x1.shape, x1.dtype),  # grad_x1
        jax.core.ShapedArray(w1.shape, w1.dtype),  # grad_w1
        jax.core.ShapedArray(w2.shape, w2.dtype),  # grad_w2
    ]

    # Always include mask gradient since we create dummy mask in forward
    outputs.append(jax.core.ShapedArray(mask.shape, mask.dtype))  # grad_mask

    return tuple(outputs)


def sigmoid_gated_dual_gemm_dual_fwd_abstract_eval(
    x1, x2, w1, w2, mask, *, transpose_out, precision
):
    """Abstract evaluation for dual input forward pass."""
    M = x1.shape[0]
    N = w1.shape[0]

    if transpose_out:
        out_shape = (N, M)
    else:
        out_shape = (M, N)

    return jax.core.ShapedArray(out_shape, x1.dtype)


def sigmoid_gated_dual_gemm_dual_bwd_abstract_eval(
    grad_out, x1, x2, w1, w2, mask, *, transpose_out, precision
):
    """Abstract evaluation for dual input backward pass."""
    outputs = [
        jax.core.ShapedArray(x1.shape, x1.dtype),  # grad_x1
        jax.core.ShapedArray(x2.shape, x2.dtype),  # grad_x2
        jax.core.ShapedArray(w1.shape, w1.dtype),  # grad_w1
        jax.core.ShapedArray(w2.shape, w2.dtype),  # grad_w2
    ]

    # Always include mask gradient since we create dummy mask in forward
    outputs.append(jax.core.ShapedArray(mask.shape, mask.dtype))  # grad_mask

    return tuple(outputs)


def sigmoid_gated_dual_gemm_reference_forward(
    x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
):
    """Pure JAX reference implementation."""
    if two_inputs:
        # Two input mode: x1 @ w1 and x2 @ w2
        acc_1 = jnp.dot(x1, w1.T, precision=jax.lax.Precision.HIGHEST)  # (M, N)
        acc_2 = jnp.dot(x2, w2.T, precision=jax.lax.Precision.HIGHEST)  # (M, N)
    else:
        # Single input mode: x1 @ w1 and x1 @ w2
        acc_1 = jnp.dot(x1, w1.T, precision=jax.lax.Precision.HIGHEST)  # (M, N)
        acc_2 = jnp.dot(x1, w2.T, precision=jax.lax.Precision.HIGHEST)  # (M, N)

    # Apply sigmoid gating
    acc_sig = jax.nn.sigmoid(acc_1)
    output = acc_sig * acc_2

    # Apply mask if provided
    if mask is not None:
        # Mask should be applied row-wise (along M dimension)
        output = output * mask[:, None]

    # Transpose output if requested
    if transpose_out:
        output = output.T

    return output


def _sigmoid_gated_dual_gemm_forward_impl(
    x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
):
    """Triton implementation of forward pass."""
    from cuequivariance_ops.triton import fused_sigmoid_gated_dual_gemm_forward_kernel

    M = x1.shape[0]
    K = x1.shape[1]
    N = w1.shape[0]

    # Default tile sizes
    TILE_M = 64
    TILE_N = 32
    TILE_K = 32

    assert N % TILE_N == 0, f"N ({N}) must be divisible by TILE_N ({TILE_N})"
    assert K % TILE_K == 0, f"K ({K}) must be divisible by TILE_K ({TILE_K})"

    if transpose_out:
        out_shape = (N, M)
    else:
        out_shape = (M, N)

    # Prepare outputs
    out_shapes = [jax.ShapeDtypeStruct(shape=out_shape, dtype=x1.dtype)]

    # Call triton kernel
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

    return results[0] if len(results) == 1 else results


def _sigmoid_gated_dual_gemm_backward_impl(
    grad_out, x1, x2, w1, w2, mask, two_inputs, transpose_out, precision
):
    """Triton implementation of backward pass."""
    from cuequivariance_ops.triton import (
        fused_sigmoid_gated_dual_gemm_backward_pregemm_kernel,
    )

    M = x1.shape[0]
    K = x1.shape[1]
    N = w1.shape[0]

    # Default tile sizes
    TILE_M = 64
    TILE_N = 32
    TILE_K = 32

    assert N % TILE_N == 0, f"N ({N}) must be divisible by TILE_N ({TILE_N})"
    assert K % TILE_K == 0, f"K ({K}) must be divisible by TILE_K ({TILE_K})"
    # Prepare output shapes
    out_shapes = [
        jax.ShapeDtypeStruct(shape=(M, N), dtype=x1.dtype),  # grad_xw1
        jax.ShapeDtypeStruct(shape=(M, N), dtype=x1.dtype),  # grad_xw2
    ]
    if mask is not None:
        out_shapes.append(
            jax.ShapeDtypeStruct(shape=(N, M), dtype=mask.dtype)
        )  # grad_mask

    # Call triton kernel
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

    # Compute final gradients
    if two_inputs:
        grad_w1 = jnp.dot(grad_xw1.T, x1)
        grad_w2 = jnp.dot(grad_xw2.T, x2)
        grad_x1 = jnp.dot(grad_xw1, w1)
        grad_x2 = jnp.dot(grad_xw2, w2)

        result = [grad_x1, grad_x2, grad_w1, grad_w2]
        if grad_mask is not None:
            grad_mask = jnp.sum(grad_mask, axis=0)
        else:
            grad_mask = jnp.zeros(x1.shape[0], dtype=x1.dtype)
        result.append(grad_mask)
        return tuple(result)
    else:
        grad_w1 = jnp.dot(grad_xw1.T, x1)
        grad_w2 = jnp.dot(grad_xw2.T, x1)
        grad_x1 = jnp.dot(grad_xw1, w1) + jnp.dot(grad_xw2, w2)

        result = [grad_x1, grad_w1, grad_w2]
        if grad_mask is not None:
            grad_mask = jnp.sum(grad_mask, axis=0)
        else:
            grad_mask = jnp.zeros(x1.shape[0], dtype=x1.dtype)
        result.append(grad_mask)
        return tuple(result)


def sigmoid_gated_dual_gemm_single_impl(platform, is_forward, *args, **kwargs):
    """Implementation dispatcher for single input mode."""
    if platform == "cuda":
        try:
            import cuequivariance_ops.triton  # noqa: F401
        except ImportError:
            pass
        else:
            if is_forward:
                x1, w1, w2, mask = args
                x2 = jnp.zeros_like(x1)  # dummy x2 for single input mode

                return _sigmoid_gated_dual_gemm_forward_impl(
                    x1, x2, w1, w2, mask, two_inputs=False, **kwargs
                )
            else:
                grad_out, x1, w1, w2, mask = args
                x2 = jnp.zeros_like(x1)  # dummy x2 for single input mode

                return _sigmoid_gated_dual_gemm_backward_impl(
                    grad_out, x1, x2, w1, w2, mask, two_inputs=False, **kwargs
                )

    if is_forward:
        x1, w1, w2, mask = args
        transpose_out = kwargs.get("transpose_out", False)
        precision = kwargs.get("precision", Precision.DEFAULT)
        return sigmoid_gated_dual_gemm_reference_forward(
            x1, None, w1, w2, mask, False, transpose_out, precision
        )
    else:
        # JAX autodiff for backward pass
        grad_out, x1, w1, w2, mask = args
        transpose_out = kwargs.get("transpose_out", False)
        precision = kwargs.get("precision", Precision.DEFAULT)

        def forward_fn(x1, w1, w2, mask):
            return sigmoid_gated_dual_gemm_reference_forward(
                x1, None, w1, w2, mask, False, transpose_out, precision
            )

            # Use JAX's autodiff for backward pass

        _, vjp_fn = jax.vjp(forward_fn, x1, w1, w2, mask)
        grad_outputs = vjp_fn(grad_out)

        # Always return mask gradient since we create dummy mask in forward
        result = [grad_outputs[0], grad_outputs[1], grad_outputs[2], grad_outputs[3]]
        return tuple(result)


def sigmoid_gated_dual_gemm_dual_impl(platform, is_forward, *args, **kwargs):
    """Implementation dispatcher for dual input mode."""
    if platform == "cuda":
        try:
            import cuequivariance_ops.triton  # noqa: F401
        except ImportError:
            pass
        else:
            if is_forward:
                x1, x2, w1, w2, mask = args

                return _sigmoid_gated_dual_gemm_forward_impl(
                    x1, x2, w1, w2, mask, two_inputs=True, **kwargs
                )
            else:
                grad_out, x1, x2, w1, w2, mask = args

                return _sigmoid_gated_dual_gemm_backward_impl(
                    grad_out, x1, x2, w1, w2, mask, two_inputs=True, **kwargs
                )

    if is_forward:
        x1, x2, w1, w2, mask = args
        transpose_out = kwargs.get("transpose_out", False)
        precision = kwargs.get("precision", Precision.DEFAULT)
        return sigmoid_gated_dual_gemm_reference_forward(
            x1, x2, w1, w2, mask, True, transpose_out, precision
        )
    else:
        # JAX autodiff for backward pass
        grad_out, x1, x2, w1, w2, mask = args
        transpose_out = kwargs.get("transpose_out", False)
        precision = kwargs.get("precision", Precision.DEFAULT)

        def forward_fn(x1, x2, w1, w2, mask):
            return sigmoid_gated_dual_gemm_reference_forward(
                x1, x2, w1, w2, mask, True, transpose_out, precision
            )

            # Use JAX's autodiff for backward pass

        _, vjp_fn = jax.vjp(forward_fn, x1, x2, w1, w2, mask)
        grad_outputs = vjp_fn(grad_out)

        # Always return mask gradient since we create dummy mask in forward
        result = [
            grad_outputs[0],
            grad_outputs[1],
            grad_outputs[2],
            grad_outputs[3],
            grad_outputs[4],
        ]
        return tuple(result)


# Register single input primitives
sigmoid_gated_dual_gemm_single_fwd_p.def_abstract_eval(
    sigmoid_gated_dual_gemm_single_fwd_abstract_eval
)
sigmoid_gated_dual_gemm_single_fwd_p.def_impl(
    partial(xla.apply_primitive, sigmoid_gated_dual_gemm_single_fwd_p)
)
sigmoid_gated_dual_gemm_single_bwd_p.def_abstract_eval(
    sigmoid_gated_dual_gemm_single_bwd_abstract_eval
)
sigmoid_gated_dual_gemm_single_bwd_p.def_impl(
    partial(xla.apply_primitive, sigmoid_gated_dual_gemm_single_bwd_p)
)

# Register dual input primitives
sigmoid_gated_dual_gemm_dual_fwd_p.def_abstract_eval(
    sigmoid_gated_dual_gemm_dual_fwd_abstract_eval
)
sigmoid_gated_dual_gemm_dual_fwd_p.def_impl(
    partial(xla.apply_primitive, sigmoid_gated_dual_gemm_dual_fwd_p)
)
sigmoid_gated_dual_gemm_dual_bwd_p.def_abstract_eval(
    sigmoid_gated_dual_gemm_dual_bwd_abstract_eval
)
sigmoid_gated_dual_gemm_dual_bwd_p.def_impl(
    partial(xla.apply_primitive, sigmoid_gated_dual_gemm_dual_bwd_p)
)

# Register lowering for single input mode
for platform in ["cuda", None]:
    mlir.register_lowering(
        sigmoid_gated_dual_gemm_single_fwd_p,
        mlir.lower_fun(
            partial(sigmoid_gated_dual_gemm_single_impl, platform, True),
            False,
        ),
        platform,
    )
    mlir.register_lowering(
        sigmoid_gated_dual_gemm_single_bwd_p,
        mlir.lower_fun(
            partial(sigmoid_gated_dual_gemm_single_impl, platform, False),
            sigmoid_gated_dual_gemm_single_bwd_p.multiple_results,
        ),
        platform,
    )
    mlir.register_lowering(
        sigmoid_gated_dual_gemm_dual_fwd_p,
        mlir.lower_fun(
            partial(sigmoid_gated_dual_gemm_dual_impl, platform, True),
            False,
        ),
        platform,
    )
    mlir.register_lowering(
        sigmoid_gated_dual_gemm_dual_bwd_p,
        mlir.lower_fun(
            partial(sigmoid_gated_dual_gemm_dual_impl, platform, False),
            sigmoid_gated_dual_gemm_dual_bwd_p.multiple_results,
        ),
        platform,
    )


@partial(custom_vjp, nondiff_argnames=("transpose_out", "precision"))
def _sigmoid_gated_dual_gemm_single(
    x1,
    w1,
    w2,
    mask,
    transpose_out=False,
    precision=Precision.DEFAULT,
):
    """JAX implementation of sigmoid-gated dual GEMM with single input and custom VJP."""
    if isinstance(precision, int):
        precision = Precision(precision)

    # Handle None values for JAX primitives
    if mask is None:
        mask = jnp.ones(x1.shape[0], dtype=x1.dtype)  # dummy mask

    result = sigmoid_gated_dual_gemm_single_fwd_p.bind(
        x1,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        precision=precision,
    )

    return result


def _sigmoid_gated_dual_gemm_single_fwd(x1, w1, w2, mask, transpose_out, precision):
    # Store original mask value
    original_mask = mask

    # Handle None values for JAX primitives
    if mask is None:
        mask = jnp.ones(x1.shape[0], dtype=x1.dtype)  # dummy mask

    result = sigmoid_gated_dual_gemm_single_fwd_p.bind(
        x1,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        precision=precision,
    )

    return result, (x1, w1, w2, original_mask)


def _sigmoid_gated_dual_gemm_single_bwd(transpose_out, precision, residuals, grad_out):
    x1, w1, w2, mask = residuals

    # Handle None values for JAX primitives
    if mask is None:
        mask = jnp.ones(x1.shape[0], dtype=x1.dtype)  # dummy mask

    return sigmoid_gated_dual_gemm_single_bwd_p.bind(
        grad_out,
        x1,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        precision=precision,
    )


_sigmoid_gated_dual_gemm_single.defvjp(
    _sigmoid_gated_dual_gemm_single_fwd, _sigmoid_gated_dual_gemm_single_bwd
)


@partial(custom_vjp, nondiff_argnames=("transpose_out", "precision"))
def _sigmoid_gated_dual_gemm_dual(
    x1,
    x2,
    w1,
    w2,
    mask,
    transpose_out=False,
    precision=Precision.DEFAULT,
):
    """JAX implementation of sigmoid-gated dual GEMM with dual input and custom VJP."""
    if isinstance(precision, int):
        precision = Precision(precision)

    # Handle None values for JAX primitives
    if mask is None:
        mask = jnp.ones(x1.shape[0], dtype=x1.dtype)  # dummy mask

    result = sigmoid_gated_dual_gemm_dual_fwd_p.bind(
        x1,
        x2,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        precision=precision,
    )

    return result


def _sigmoid_gated_dual_gemm_dual_fwd(x1, x2, w1, w2, mask, transpose_out, precision):
    # Handle None values for JAX primitives
    if mask is None:
        mask = jnp.ones(x1.shape[0], dtype=x1.dtype)  # dummy mask

    result = sigmoid_gated_dual_gemm_dual_fwd_p.bind(
        x1,
        x2,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        precision=precision,
    )

    return result, (x1, x2, w1, w2, mask)


def _sigmoid_gated_dual_gemm_dual_bwd(transpose_out, precision, residuals, grad_out):
    x1, x2, w1, w2, mask = residuals

    # Handle None values for JAX primitives
    if mask is None:
        mask = jnp.ones(x1.shape[0], dtype=x1.dtype)  # dummy mask

    return sigmoid_gated_dual_gemm_dual_bwd_p.bind(
        grad_out,
        x1,
        x2,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        precision=precision,
    )


_sigmoid_gated_dual_gemm_dual.defvjp(
    _sigmoid_gated_dual_gemm_dual_fwd, _sigmoid_gated_dual_gemm_dual_bwd
)


def sigmoid_gated_dual_gemm(
    x,
    w1,
    w2,
    mask: Optional[jnp.ndarray] = None,
    transpose_out: bool = False,
    precision: Precision = Precision.DEFAULT,
):
    """Apply fused sigmoid-gated dual GEMM operation with single input.

    This function performs a dual matrix multiplication with sigmoid gating:
    1. First matrix multiplication: x @ w1
    2. Second matrix multiplication: x @ w2
    3. Apply sigmoid to the first result
    4. Element-wise multiplication of sigmoid output with second result
    5. Optional masking of the final output

    Args:
        x: Input tensor of shape (M, K)
        w1: First weight matrix of shape (N, K) for the main projection
        w2: Second weight matrix of shape (N, K) for the gating projection
        mask: Optional mask tensor of shape (M,) for element-wise multiplication
        transpose_out: Whether to transpose the output
        precision: Precision mode for matrix multiplication

    Returns:
        Output tensor of shape (M, N) if transpose_out=False, (N, M) if transpose_out=True

    Examples:
        >>> x = jnp.ones((4, 128))  # (M, K)
        >>> w1 = jnp.ones((64, 128))  # (N, K)
        >>> w2 = jnp.ones((64, 128))  # (N, K)
        >>> out = sigmoid_gated_dual_gemm(x, w1, w2)
        >>> out.shape  # (M, N)
        (4, 64)
    """
    # Ensure inputs are contiguous
    x = jnp.asarray(x)
    w1 = jnp.asarray(w1)
    w2 = jnp.asarray(w2)

    # Reshape x to 2D if needed
    original_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])

    # Call internal implementation
    out = _sigmoid_gated_dual_gemm_single(
        x,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        precision=precision,
    )

    # Reshape output back to original batch dimensions
    if len(original_shape) > 2:
        if transpose_out:
            out_shape = (w1.shape[0], *original_shape[:-1])
        else:
            out_shape = (*original_shape[:-1], w1.shape[0])
        out = out.reshape(out_shape)

    return out


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

    This function performs a dual matrix multiplication with sigmoid gating:
    1. First matrix multiplication: x1 @ w1
    2. Second matrix multiplication: x2 @ w2
    3. Apply sigmoid to the first result
    4. Element-wise multiplication of sigmoid output with second result
    5. Optional masking of the final output

    Args:
        x1: First input tensor of shape (M, K)
        x2: Second input tensor of shape (M, K)
        w1: First weight matrix of shape (N, K) for the main projection
        w2: Second weight matrix of shape (N, K) for the gating projection
        mask: Optional mask tensor of shape (M,) for element-wise multiplication
        transpose_out: Whether to transpose the output
        precision: Precision mode for matrix multiplication

    Returns:
        Output tensor of shape (M, N) if transpose_out=False, (N, M) if transpose_out=True

    Examples:
        >>> x1 = jnp.ones((4, 128))  # (M, K)
        >>> x2 = jnp.ones((4, 128))  # (M, K)
        >>> w1 = jnp.ones((64, 128))  # (N, K)
        >>> w2 = jnp.ones((64, 128))  # (N, K)
        >>> out = sigmoid_gated_dual_gemm_dual_x(x1, x2, w1, w2)
        >>> out.shape  # (M, N)
        (4, 64)
    """
    # Ensure inputs are contiguous
    x1 = jnp.asarray(x1)
    x2 = jnp.asarray(x2)
    w1 = jnp.asarray(w1)
    w2 = jnp.asarray(w2)

    # Reshape inputs to 2D if needed
    original_shape = x1.shape
    if x1.ndim > 2:
        x1 = x1.reshape(-1, x1.shape[-1])
        x2 = x2.reshape(-1, x2.shape[-1])

    # Call internal implementation
    out = _sigmoid_gated_dual_gemm_dual(
        x1,
        x2,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        precision=precision,
    )

    # Reshape output back to original batch dimensions
    if len(original_shape) > 2:
        if transpose_out:
            out_shape = (w1.shape[0], *original_shape[:-1])
        else:
            out_shape = (*original_shape[:-1], w1.shape[0])
        out = out.reshape(out_shape)

    return out
