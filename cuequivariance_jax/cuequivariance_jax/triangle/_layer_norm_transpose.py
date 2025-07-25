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
from jax.interpreters import mlir, xla

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
    Layout.BND_BND: {"dims": lambda x: x.shape, "tiles": (64, 64)},
    Layout.BDN_BND: {
        "dims": lambda x: (x.shape[0], x.shape[2], x.shape[1]),
        "tiles": (64, 64),
    },
    Layout.BND_BDN: {"dims": lambda x: x.shape, "tiles": (64, 64)},
    Layout.DBN_BND: {
        "dims": lambda x: (x.shape[1], x.shape[2], x.shape[0]),
        "tiles": (64, 64),
    },
    Layout.BND_DBN: {"dims": lambda x: x.shape, "tiles": (64, 64)},
}

OUTPUT_SHAPES = {
    Layout.BND_BND: lambda B, N, D: (B, N, D),
    Layout.BDN_BND: lambda B, N, D: (B, N, D),
    Layout.BND_BDN: lambda B, N, D: (B, D, N),
    Layout.DBN_BND: lambda B, N, D: (B, N, D),
    Layout.BND_DBN: lambda B, N, D: (D, B, N),
}


def get_dims_and_config(x, layout):
    """Get B, N, D dimensions and tile configuration for given layout."""
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


def layer_norm_fwd_abstract_eval(x, w, b, *, eps, elementwise_affine, layout, fallback):
    B, N, D, _ = get_dims_and_config(x, layout)
    out_shape = OUTPUT_SHAPES[layout](B, N, D)
    return (
        jax.core.ShapedArray(out_shape, x.dtype),
        jax.core.ShapedArray((B, N), x.dtype),
        jax.core.ShapedArray((B, N), x.dtype),
    )


def layer_norm_bwd_abstract_eval(
    grad_out, x, w, b, mean, rstd, *, eps, elementwise_affine, layout, fallback
):
    return (
        jax.core.ShapedArray(x.shape, x.dtype),
        jax.core.ShapedArray(w.shape, w.dtype),
        jax.core.ShapedArray(b.shape, b.dtype),
    )


def layer_norm_transpose_reference_forward(x, w, b, eps, elementwise_affine, layout):
    """Pure JAX reference implementation."""
    # Transform input to BND format
    if layout == Layout.BDN_BND:
        x = x.transpose(0, 2, 1)
    elif layout == Layout.DBN_BND:
        x = x.transpose(1, 2, 0)

    # Compute mean and normalize
    mean = jnp.mean(x, axis=2, keepdims=False)
    x_centered = x - mean[:, :, None]
    var = jnp.mean(x_centered * x_centered, axis=2, keepdims=False)
    rstd = 1.0 / jnp.sqrt(var + eps)
    x_hat = x_centered * rstd[:, :, None]

    # Apply affine transformation
    if elementwise_affine:
        out = x_hat * w[None, None, :] + b[None, None, :]
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
    # Use dtype-dependent TILE_N for backward pass
    TILE_N = get_backward_tile_n(x.dtype, base_tile_n)
    num_tiles = triton.cdiv(N, TILE_N)

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
        num_warps=8,
        num_stages=2,
    )

    grad_w = jnp.sum(grad_w_tiles, axis=(0, 1))
    grad_b = jnp.sum(grad_b_tiles, axis=(0, 1))

    # When elementwise_affine=False, gradients w.r.t. w and b should be zero
    if not elementwise_affine:
        grad_w = jnp.zeros_like(w)
        grad_b = jnp.zeros_like(b)

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


@partial(
    custom_vjp, nondiff_argnames=("eps", "elementwise_affine", "layout", "fallback")
)
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
    return layer_norm_bwd_p.bind(
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


_layer_norm.defvjp(_layer_norm_fwd, _layer_norm_bwd)


def layer_norm_transpose(
    x, w, b, eps=1e-5, elementwise_affine=True, layout="nd->nd", fallback=False
):
    """Apply fused layer normalization with support for various input layouts.

    Args:
        x: Input tensor
        w: Weight tensor for scaling (shape: (D,))
        b: Bias tensor for shifting (shape: (D,))
        eps: Small constant for numerical stability
        elementwise_affine: Whether to apply elementwise affine transformation
        layout: Input/output layout specification
        fallback: Whether to force fallback to reference implementation

    Returns:
        Normalized tensor with shape determined by the output layout

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
