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

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.interpreters import batching, mlir, xla

from cuequivariance_jax.triangle._naive_batching import naive_batching_rule

try:
    import jax_triton as jt
    import triton

    HAS_JAX_TRITON = True
except ImportError:
    HAS_JAX_TRITON = False


# copy from cuequivariance_ops to avoid requiring cuequivariance_ops to be installed
class Layout(enum.IntEnum):
    BND_BND = 0
    BDN_BND = 1
    BND_BDN = 2
    DBN_BND = 3
    BND_DBN = 4


# JAX primitives
layer_norm_fwd_p = jax.extend.core.Primitive("layer_norm_transpose_fwd")
layer_norm_fwd_p.multiple_results = True
layer_norm_bwd_p = jax.extend.core.Primitive("layer_norm_transpose_bwd")
layer_norm_bwd_p.multiple_results = True

# Layout configuration mapping
LAYOUT_CONFIG = {
    Layout.BND_BND: {
        "dims": lambda x: x.shape,
        "tiles": (64, 64),
        "in_batch_axis": 0,
        "out_batch_axis": 0,
    },
    Layout.BDN_BND: {
        "dims": lambda x: (x.shape[0], x.shape[2], x.shape[1]),
        "tiles": (64, 64),
        "in_batch_axis": 0,
        "out_batch_axis": 0,
    },
    Layout.BND_BDN: {
        "dims": lambda x: x.shape,
        "tiles": (64, 64),
        "in_batch_axis": 0,
        "out_batch_axis": 0,
    },
    Layout.DBN_BND: {
        "dims": lambda x: (x.shape[1], x.shape[2], x.shape[0]),
        "tiles": (64, 64),
        "in_batch_axis": 1,
        "out_batch_axis": 0,
    },
    Layout.BND_DBN: {
        "dims": lambda x: x.shape,
        "tiles": (64, 64),
        "in_batch_axis": 0,
        "out_batch_axis": 1,
    },
}

OUTPUT_SHAPES = {
    Layout.BND_BND: lambda B, N, D: (B, N, D),
    Layout.BDN_BND: lambda B, N, D: (B, N, D),
    Layout.BND_BDN: lambda B, N, D: (B, D, N),
    Layout.DBN_BND: lambda B, N, D: (B, N, D),
    Layout.BND_DBN: lambda B, N, D: (D, B, N),
}


def get_dims_and_config(x, layout):
    """Get B, N, D dimensions and tile configuration for given layout.

    Args:
        x: Input tensor with shape depending on layout:
            - Layout.BND_BND: (B, N, D)
            - Layout.BDN_BND: (B, D, N) -> dims extracted as (B, N, D)
            - Layout.BND_BDN: (B, N, D)
            - Layout.DBN_BND: (D, B, N) -> dims extracted as (B, N, D)
            - Layout.BND_DBN: (B, N, D)
        layout: Layout enum specifying tensor layout

    Returns:
        Tuple containing:
            - B: Batch dimension
            - N: Sequence/spatial dimension
            - D: Feature dimension
            - tiles: Tuple of tile sizes (TILE_N, TILE_D) for kernel optimization
    """
    B, N, D = LAYOUT_CONFIG[layout]["dims"](x)
    tiles = LAYOUT_CONFIG[layout]["tiles"]
    return B, N, D, tiles


def get_backward_tile_n(dtype, base_tile_n=64):
    """Get TILE_N for backward pass based on data type.

    Args:
        dtype: Input tensor dtype
        base_tile_n: Base TILE_N value (default 64)

    Returns:
        TILE_N value: 32 for float32, base_tile_n for others
    """
    if dtype == jnp.float32:
        return 32
    return base_tile_n


def layer_norm_fwd_abstract_eval(x, w, b, *, layout, **unused_kwargs):
    # x: shape determined by input layout
    # w: shape (D,)
    # b: shape (D,)
    B, N, D, _ = get_dims_and_config(x, layout)
    out_shape = OUTPUT_SHAPES[layout](B, N, D)
    return (
        jax.core.ShapedArray(out_shape, x.dtype),  # out
        jax.core.ShapedArray((B, N), x.dtype),  # mean
        jax.core.ShapedArray((B, N), x.dtype),  # rstd
    )


def layer_norm_bwd_abstract_eval(
    grad_out, x, w, b, mean, rstd, *, layout, **unused_kwargs
):
    # grad_out: shape determined by output layout
    # x: shape determined by input layout
    # w: shape (D,)
    # b: shape (D,)
    # mean: shape (B, N)
    # rstd: shape (B, N)
    B, N, D, _ = get_dims_and_config(x, layout)
    return (
        jax.core.ShapedArray(x.shape, x.dtype),  # shape determined by input layout
        jax.core.ShapedArray((B, D), w.dtype),  # (B, D)
        jax.core.ShapedArray((B, D), b.dtype),  # (B, D)
    )


def layer_norm_transpose_reference_forward(x, w, b, eps, elementwise_affine, layout):
    """Pure JAX reference implementation of layer normalization with layout transformation.

    Args:
        x: Input tensor with layout-dependent shape:
            - Layout.BND_BND: (B, N, D)
            - Layout.BDN_BND: (B, D, N)
            - Layout.DBN_BND: (D, B, N)
            - Layout.BND_BDN: (B, N, D)
            - Layout.BND_DBN: (B, N, D)
        w: Weight tensor for scaling, shape (D,) (or (B, D))
        b: Bias tensor for shifting, shape (D,) (or (B, D))
        eps: Small constant for numerical stability
        elementwise_affine: Whether to apply elementwise affine transformation
        layout: Layout enum specifying input/output transformation

    Returns:
        Tuple containing:
            - out: Normalized output tensor with layout-dependent shape:
                - Layout.BND_BND: (B, N, D)
                - Layout.BDN_BND: (B, N, D)
                - Layout.DBN_BND: (B, N, D)
                - Layout.BND_BDN: (B, D, N)
                - Layout.BND_DBN: (D, B, N)
            - mean: Per-sample means, shape (B, N)
            - rstd: Per-sample reciprocal standard deviations, shape (B, N)
    """
    B, N, D, _ = get_dims_and_config(x, layout)

    # Transform input to BND format
    if layout == Layout.BDN_BND:
        x = x.transpose(0, 2, 1)
    elif layout == Layout.DBN_BND:
        x = x.transpose(1, 2, 0)

    assert x.shape == (B, N, D), f"x.shape: {x.shape}"
    assert w.shape == (D,) or w.shape == (B, D), f"w.shape: {w.shape}"
    assert b.shape == (D,) or b.shape == (B, D), f"b.shape: {b.shape}"

    # Compute mean and normalize
    mean = jnp.mean(x, axis=2, keepdims=False)
    x_centered = x - mean[:, :, None]
    var = jnp.mean(x_centered * x_centered, axis=2, keepdims=False)
    rstd = 1.0 / jnp.sqrt(var + eps)
    x_hat = x_centered * rstd[:, :, None]

    # Apply affine transformation
    if elementwise_affine:
        w = w[:, None, :] if w.ndim == 2 else w[None, None, :]
        b = b[:, None, :] if b.ndim == 2 else b[None, None, :]

        out = x_hat * w + b
    else:
        out = x_hat

    # Transform to output layout
    if layout == Layout.BND_BDN:
        out = out.transpose(0, 2, 1)
    elif layout == Layout.BND_DBN:
        out = out.transpose(2, 0, 1)

    out = out.astype(x.dtype)
    mean = mean.astype(x.dtype)
    rstd = rstd.astype(x.dtype)

    return out, mean, rstd


def _layer_norm_forward_impl(x, w, b, eps, elementwise_affine, layout):
    """Triton implementation of forward pass."""
    if not HAS_JAX_TRITON:
        raise ImportError("jax_triton is required for GPU implementation")

    from cuequivariance_ops.triton import layer_norm_transpose_forward_kernel

    B, N, D, (TILE_N, TILE_D) = get_dims_and_config(x, layout)
    out_shape = OUTPUT_SHAPES[layout](B, N, D)

    assert w.shape == (D,)
    assert b.shape == (D,)

    NEEDS_INT64 = B * N * D >= 2**31 - 1

    out, mean, rstd = jt.triton_call(
        x,
        w,
        b,
        kernel=layer_norm_transpose_forward_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(shape=out_shape, dtype=x.dtype),
            jax.ShapeDtypeStruct(shape=(B, N), dtype=x.dtype),
            jax.ShapeDtypeStruct(shape=(B, N), dtype=x.dtype),
        ],
        grid=(triton.cdiv(N, TILE_N), B, 1),
        B=B,
        N=N,
        D=D,
        EPS=eps,
        TILE_N=TILE_N,
        TILE_D=TILE_D,
        ELEMENTWISE_AFFINE=elementwise_affine,
        LAYOUT=layout.value,
        NEEDS_INT64=NEEDS_INT64,
        num_warps=8,
        num_stages=2,
    )
    return out, mean, rstd


def _layer_norm_backward_impl(
    grad_out, x, w, b, mean, rstd, eps, elementwise_affine, layout
):
    """Triton implementation of backward pass."""
    if not HAS_JAX_TRITON:
        raise ImportError("jax_triton is required for GPU implementation")

    from cuequivariance_ops.triton import layer_norm_transpose_backward_kernel

    B, N, D, (base_tile_n, TILE_D) = get_dims_and_config(x, layout)
    assert w.shape == (D,)
    assert b.shape == (D,)

    # Use dtype-dependent TILE_N for backward pass
    TILE_N = get_backward_tile_n(x.dtype, base_tile_n)
    num_tiles = triton.cdiv(N, TILE_N)

    NEEDS_INT64 = B * N * D >= 2**31 - 1

    grad_x, grad_w_tiles, grad_b_tiles = jt.triton_call(
        grad_out,
        x,
        w,
        mean,
        rstd,
        kernel=layer_norm_transpose_backward_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
            jax.ShapeDtypeStruct(shape=(B, num_tiles, D), dtype=w.dtype),
            jax.ShapeDtypeStruct(shape=(B, num_tiles, D), dtype=w.dtype),
        ],
        grid=(num_tiles, B, 1),
        B=B,
        N=N,
        D=D,
        TILE_N=TILE_N,
        TILE_D=TILE_D,
        ELEMENTWISE_AFFINE=elementwise_affine,
        LAYOUT=layout.value,
        NEEDS_INT64=NEEDS_INT64,
        num_warps=8,
        num_stages=2,
    )

    grad_w = jnp.sum(grad_w_tiles, axis=1)
    grad_b = jnp.sum(grad_b_tiles, axis=1)
    assert grad_w.shape == (B, D)
    assert grad_b.shape == (B, D)

    # When elementwise_affine=False, gradients w.r.t. w and b should be zero
    if not elementwise_affine:
        grad_w = jnp.zeros_like(w, shape=(B, D))
        grad_b = jnp.zeros_like(b, shape=(B, D))

    return grad_x, grad_w, grad_b


def layer_norm_impl(platform, is_forward, *args, **kwargs):
    """Unified implementation dispatcher."""
    fallback = kwargs.pop("fallback", False)

    if platform == "cuda" and not fallback:
        return (
            _layer_norm_forward_impl(*args, **kwargs)
            if is_forward
            else _layer_norm_backward_impl(*args, **kwargs)
        )

    if is_forward:
        return layer_norm_transpose_reference_forward(*args, **kwargs)
    else:
        # JAX autodiff for backward pass
        grad_out, x, w, b, mean, rstd = args
        eps = kwargs["eps"]
        elementwise_affine = kwargs["elementwise_affine"]
        layout = kwargs["layout"]

        def forward_fn(x, w, b):
            return layer_norm_transpose_reference_forward(
                x, w, b, eps, elementwise_affine, layout
            )[0]

        # We need to broadcase (D,) -> (B, D) to output a gradient with a batch size to be compatible with _layer_norm_backward_impl format
        # _layer_norm_backward_impl outputs a w and b gradient with shape (B, D) to support the vmap rule
        B, N, D, _ = get_dims_and_config(x, layout)
        w = jnp.broadcast_to(w, (B, D))
        b = jnp.broadcast_to(b, (B, D))

        _, vjp_fn = jax.vjp(forward_fn, x, w, b)
        return vjp_fn(grad_out)


# Register primitives
layer_norm_fwd_p.def_abstract_eval(layer_norm_fwd_abstract_eval)
layer_norm_fwd_p.def_impl(partial(xla.apply_primitive, layer_norm_fwd_p))
layer_norm_bwd_p.def_abstract_eval(layer_norm_bwd_abstract_eval)
layer_norm_bwd_p.def_impl(partial(xla.apply_primitive, layer_norm_bwd_p))

for platform in ["cuda", None]:
    mlir.register_lowering(
        layer_norm_fwd_p,
        mlir.lower_fun(
            partial(layer_norm_impl, platform, True), layer_norm_fwd_p.multiple_results
        ),
        platform,
    )
    mlir.register_lowering(
        layer_norm_bwd_p,
        mlir.lower_fun(
            partial(layer_norm_impl, platform, False), layer_norm_bwd_p.multiple_results
        ),
        platform,
    )


def _layer_norm_fwd_batching_rule(
    batched_inputs: tuple[jax.Array, ...],
    vmapped_axes: tuple[int | None, ...],
    *,
    eps,
    elementwise_affine,
    layout,
    fallback,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    """Batching rule for layer norm forward pass."""
    axis_in = LAYOUT_CONFIG[layout]["in_batch_axis"]
    axis_out = LAYOUT_CONFIG[layout]["out_batch_axis"]
    return naive_batching_rule(
        layer_norm_fwd_p,
        (axis_in, None, None),  # (x, w, b)
        (axis_out, 0, 0),  # (out, mean, rstd)
        batched_inputs,
        vmapped_axes,
        eps=eps,
        elementwise_affine=elementwise_affine,
        layout=layout,
        fallback=fallback,
    )


def _layer_norm_bwd_batching_rule(
    batched_inputs: tuple[jax.Array, ...],
    vmapped_axes: tuple[int | None, ...],
    *,
    eps,
    elementwise_affine,
    layout,
    fallback,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    """Batching rule for layer norm backward pass."""
    axis_in = LAYOUT_CONFIG[layout]["in_batch_axis"]
    axis_out = LAYOUT_CONFIG[layout]["out_batch_axis"]
    return naive_batching_rule(
        layer_norm_bwd_p,
        (axis_out, axis_in, None, None, 0, 0),  # (grad_out, x, w, b, mean, rstd)
        (axis_in, 0, 0),  # (grad_x, grad_w, grad_b)
        batched_inputs,
        vmapped_axes,
        eps=eps,
        elementwise_affine=elementwise_affine,
        layout=layout,
        fallback=fallback,
    )


batching.primitive_batchers[layer_norm_fwd_p] = _layer_norm_fwd_batching_rule
batching.primitive_batchers[layer_norm_bwd_p] = _layer_norm_bwd_batching_rule


@partial(custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _layer_norm(
    x, w, b, eps=1e-5, elementwise_affine=True, layout=Layout.BND_BND, fallback=False
):
    """JAX implementation of layer norm with custom VJP."""
    if isinstance(layout, int):
        layout = Layout(layout)
    out, mean, rstd = layer_norm_fwd_p.bind(
        x,
        w,
        b,
        eps=eps,
        elementwise_affine=elementwise_affine,
        layout=layout,
        fallback=fallback,
    )
    return out


def _layer_norm_fwd(x, w, b, eps, elementwise_affine, layout, fallback):
    out, mean, rstd = layer_norm_fwd_p.bind(
        x,
        w,
        b,
        eps=eps,
        elementwise_affine=elementwise_affine,
        layout=layout,
        fallback=fallback,
    )
    return out, (x, w, b, mean, rstd)


def _layer_norm_bwd(eps, elementwise_affine, layout, fallback, residuals, grad_out):
    x, w, b, mean, rstd = residuals
    grad_x, grad_w, grad_b = layer_norm_bwd_p.bind(
        grad_out,
        x,
        w,
        b,
        mean,
        rstd,
        eps=eps,
        elementwise_affine=elementwise_affine,
        layout=layout,
        fallback=fallback,
    )
    grad_w = jnp.sum(grad_w, axis=0)
    grad_b = jnp.sum(grad_b, axis=0)
    assert grad_x.shape == x.shape
    assert grad_w.shape == w.shape
    assert grad_b.shape == b.shape
    return (grad_x, grad_w, grad_b)


_layer_norm.defvjp(_layer_norm_fwd, _layer_norm_bwd)


def layer_norm_transpose(
    x, w, b, eps=1e-5, elementwise_affine=True, layout="nd->nd", fallback=False
):
    """Apply fused layer normalization with support for various input/output layouts.

    This function performs layer normalization along the last dimension while supporting
    different input and output tensor layouts through reshaping and transposition.

    Args:
        x: Input tensor with shape depending on layout:
            - "nd->nd": (N, D)
            - "nd->dn": (N, D)
            - "bnd->bnd": (B, N, D)
            - "bdn->bnd": (B, D, N)
            - "bnd->bdn": (B, N, D)
            - "dbn->bnd": (D, B, N)
            - "bnd->dbn": (B, N, D)
            - "bijd->bijd": (B, I, J, D)
            - "bijd->bdij": (B, I, J, D)
            - "bdij->bijd": (B, D, I, J)
            - "dbij->bijd": (D, B, I, J)
            - "bijd->dbij": (B, I, J, D)
        w: Weight tensor for scaling, shape (D,) where D is the feature dimension
        b: Bias tensor for shifting, shape (D,) where D is the feature dimension
        eps: Small constant for numerical stability (default: 1e-5)
        elementwise_affine: Whether to apply elementwise affine transformation (default: True)
        layout: Input/output layout specification string (default: "nd->nd")
        fallback: Whether to force fallback to reference implementation (default: False)

    Returns:
        Normalized tensor with shape determined by the output layout:
            - "nd->nd": (N, D)
            - "nd->dn": (D, N)
            - "bnd->bnd": (B, N, D)
            - "bdn->bnd": (B, N, D)
            - "bnd->bdn": (B, D, N)
            - "dbn->bnd": (B, N, D)
            - "bnd->dbn": (D, B, N)
            - "bijd->bijd": (B, I, J, D)
            - "bijd->bdij": (B, D, I, J)
            - "bdij->bijd": (B, I, J, D)
            - "dbij->bijd": (B, I, J, D)
            - "bijd->dbij": (D, B, I, J)

    Notes:
        - Normalization is always performed along the feature dimension D
        - For 4D tensors like "bijd->bijd", the I*J dimensions are flattened to N for normalization
        - When fallback=False, uses optimized Triton kernels on GPU; otherwise uses JAX reference

    Examples:
        >>> x = jnp.ones((4, 16, 64))  # (B, N, D)
        >>> w = jnp.ones((64,))
        >>> b = jnp.zeros((64,))
        >>> out = layer_norm_transpose(x, w, b, layout="bnd->bnd")
        >>> out.shape  # (B, N, D)
        (4, 16, 64)
    """
    # Layout mapping with input parsing and output reshaping
    layout_map = {
        "nd->nd": (
            lambda x: (1, *x.shape),
            Layout.BND_BND,
            lambda out, x: out.reshape(x.shape),
        ),
        "nd->dn": (
            lambda x: (1, *x.shape),
            Layout.BND_BDN,
            lambda out, x: out.reshape(x.shape[::-1]),
        ),
        "bnd->bnd": (lambda x: x.shape, Layout.BND_BND, lambda out, x: out),
        "bdn->bnd": (lambda x: x.shape, Layout.BDN_BND, lambda out, x: out),
        "bnd->bdn": (lambda x: x.shape, Layout.BND_BDN, lambda out, x: out),
        "dbn->bnd": (lambda x: x.shape, Layout.DBN_BND, lambda out, x: out),
        "bnd->dbn": (lambda x: x.shape, Layout.BND_DBN, lambda out, x: out),
        "bijd->bijd": (
            lambda x: (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]),
            Layout.BND_BND,
            lambda out, x: out.reshape(x.shape),
        ),
        "bijd->bdij": (
            lambda x: (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]),
            Layout.BND_BDN,
            lambda out, x: out.reshape(x.shape[0], x.shape[3], x.shape[1], x.shape[2]),
        ),
        "bdij->bijd": (
            lambda x: (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]),
            Layout.BDN_BND,
            lambda out, x: out.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1]),
        ),
        "dbij->bijd": (
            lambda x: (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]),
            Layout.DBN_BND,
            lambda out, x: out.reshape(x.shape[1], x.shape[2], x.shape[3], x.shape[0]),
        ),
        "bijd->dbij": (
            lambda x: (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]),
            Layout.BND_DBN,
            lambda out, x: out.reshape(x.shape[3], x.shape[0], x.shape[1], x.shape[2]),
        ),
    }

    if layout not in layout_map:
        raise ValueError(
            f"layout {layout} not supported. supported layouts are: {list(layout_map.keys())}"
        )

    shape_fn, layout_enum, reshape_fn = layout_map[layout]
    x_reshaped = x.reshape(shape_fn(x))
    out = _layer_norm(x_reshaped, w, b, eps, elementwise_affine, layout_enum, fallback)
    return reshape_fn(out, x)
