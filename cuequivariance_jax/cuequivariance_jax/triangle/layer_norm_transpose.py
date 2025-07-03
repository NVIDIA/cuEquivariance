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
import jax_triton as jt
import triton
from cuequivariance_ops.triton import (
    Layout,
    layer_norm_transpose_backward_kernel,
    layer_norm_transpose_forward_kernel,
)
from jax import custom_vjp
from jax.interpreters import mlir, xla

# Create JAX primitives
layer_norm_fwd_p = jax.extend.core.Primitive("layer_norm_transpose_fwd")
layer_norm_fwd_p.multiple_results = True

layer_norm_bwd_p = jax.extend.core.Primitive("layer_norm_transpose_bwd")
layer_norm_bwd_p.multiple_results = True


def get_dims_from_input(x, layout: Layout):
    """Extract B, N, D dimensions from input tensor based on layout."""
    if layout == Layout.BND_BND:
        B, N, D = x.shape
    elif layout == Layout.BDN_BND:
        B, D, N = x.shape
    elif layout == Layout.BND_BDN:
        B, N, D = x.shape
    elif layout == Layout.DBN_BND:
        D, B, N = x.shape
    elif layout == Layout.BND_DBN:
        B, N, D = x.shape
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return B, N, D


def get_output_shape(B: int, N: int, D: int, layout: Layout) -> tuple[int, ...]:
    """Get output shape based on B, N, D and layout."""
    if layout == Layout.BND_BND:
        return (B, N, D)
    elif layout == Layout.BDN_BND:
        return (B, N, D)
    elif layout == Layout.BND_BDN:
        return (B, D, N)
    elif layout == Layout.DBN_BND:
        return (B, N, D)
    elif layout == Layout.BND_DBN:
        return (D, B, N)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def layer_norm_fwd_abstract_eval(
    x: jax.core.ShapedArray,
    w: jax.core.ShapedArray,
    b: jax.core.ShapedArray,
    *,
    eps: float,
    elementwise_affine: bool,
    layout: Layout,
) -> tuple[jax.core.ShapedArray, jax.core.ShapedArray, jax.core.ShapedArray]:
    B, N, D = get_dims_from_input(x, layout)
    out_shape = get_output_shape(B, N, D, layout)
    out_shape_array = jax.core.ShapedArray(out_shape, x.dtype)
    mean_shape_array = jax.core.ShapedArray((B, N), x.dtype)
    rstd_shape_array = jax.core.ShapedArray((B, N), x.dtype)
    return out_shape_array, mean_shape_array, rstd_shape_array


def layer_norm_bwd_abstract_eval(
    grad_out: jax.core.ShapedArray,
    x: jax.core.ShapedArray,
    w: jax.core.ShapedArray,
    b: jax.core.ShapedArray,
    mean: jax.core.ShapedArray,
    rstd: jax.core.ShapedArray,
    *,
    eps: float,
    elementwise_affine: bool,
    layout: Layout,
) -> tuple[jax.core.ShapedArray, jax.core.ShapedArray, jax.core.ShapedArray]:
    grad_x_shape = jax.core.ShapedArray(x.shape, x.dtype)
    grad_w_shape = jax.core.ShapedArray(w.shape, w.dtype)
    grad_b_shape = jax.core.ShapedArray(b.shape, b.dtype)
    return grad_x_shape, grad_w_shape, grad_b_shape


def layer_norm_fwd_impl(
    platform: str | None,
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    *,
    eps: float,
    elementwise_affine: bool,
    layout: Layout,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if platform == "cuda":
        return _layer_norm_forward_impl(x, w, b, eps, elementwise_affine, layout)
    else:
        return layer_norm_transpose_reference_forward(
            x, w, b, eps, elementwise_affine, layout
        )


def layer_norm_bwd_impl(
    platform: str | None,
    grad_out: jax.Array,
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    mean: jax.Array,
    rstd: jax.Array,
    *,
    eps: float,
    elementwise_affine: bool,
    layout: Layout,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if platform == "cuda":
        return _layer_norm_backward_impl(
            grad_out, x, w, b, mean, rstd, eps, elementwise_affine, layout
        )
    else:
        # Use JAX autodiff for backward pass
        def forward_fn(x, w, b):
            out, mean, rstd = layer_norm_transpose_reference_forward(
                x, w, b, eps, elementwise_affine, layout
            )
            return out

        _, vjp_fn = jax.vjp(forward_fn, x, w, b)
        grad_x, grad_w, grad_b = vjp_fn(grad_out)
        return grad_x, grad_w, grad_b


# Register abstract evaluation functions
layer_norm_fwd_p.def_abstract_eval(layer_norm_fwd_abstract_eval)
layer_norm_fwd_p.def_impl(partial(xla.apply_primitive, layer_norm_fwd_p))
for platform in ["cuda", None]:
    mlir.register_lowering(
        layer_norm_fwd_p,
        mlir.lower_fun(
            partial(layer_norm_fwd_impl, platform), layer_norm_fwd_p.multiple_results
        ),
        platform,
    )

layer_norm_bwd_p.def_abstract_eval(layer_norm_bwd_abstract_eval)
layer_norm_bwd_p.def_impl(partial(xla.apply_primitive, layer_norm_bwd_p))
for platform in ["cuda", None]:
    mlir.register_lowering(
        layer_norm_bwd_p,
        mlir.lower_fun(
            partial(layer_norm_bwd_impl, platform), layer_norm_bwd_p.multiple_results
        ),
        platform,
    )


def layer_norm_transpose_reference_forward(x, w, b, eps, elementwise_affine, layout):
    """Pure JAX reference implementation of layer_norm_transpose_forward_kernel."""
    assert x.ndim == 3

    B, N, D = get_dims_from_input(x, layout)

    # Ensure input is in BND format for computation
    if layout == Layout.BDN_BND:
        x = jnp.transpose(x, (0, 2, 1))  # BDN -> BND
    elif layout == Layout.DBN_BND:
        x = jnp.transpose(x, (1, 2, 0))  # DBN -> BND
    elif layout == Layout.BND_DBN:
        x = jnp.transpose(x, (2, 0, 1))  # BND -> DBN
    elif layout == Layout.BND_BDN:
        # Already in BND format, no change needed
        pass

    # Now x is always in BND format for computation
    # Compute mean along the D dimension for each (B, N) position
    mean = jnp.mean(x, axis=2, keepdims=False)  # Shape: (B, N)

    # Compute variance along the D dimension
    x_centered = x - mean[:, :, None]  # Broadcast mean to (B, N, D)
    var = jnp.mean(x_centered * x_centered, axis=2, keepdims=False)  # Shape: (B, N)

    # Compute reciprocal standard deviation
    rstd = 1.0 / jnp.sqrt(var + eps)  # Shape: (B, N)

    # Normalize the input
    x_hat = x_centered * rstd[:, :, None]  # Shape: (B, N, D)

    # Apply elementwise affine transformation if enabled
    if elementwise_affine:
        # w and b have shape (D,), broadcast to (B, N, D)
        out = x_hat * w[None, None, :] + b[None, None, :]
    else:
        out = x_hat

    # Apply output layout transformation
    if layout == Layout.BND_BND:
        # Output shape: (B, N, D) - no change needed
        pass
    elif layout == Layout.BDN_BND:
        # Input was BDN, output should be BND - no change needed
        pass
    elif layout == Layout.BND_BDN:
        # Output shape: (B, D, N)
        out = out.transpose(0, 2, 1)  # BND -> BDN
    elif layout == Layout.DBN_BND:
        # Input was DBN, output should be BND - no change needed
        pass
    elif layout == Layout.BND_DBN:
        # Output shape: (D, B, N)
        out = out.transpose(2, 0, 1)  # BND -> DBN

    return out, mean, rstd


def _layer_norm_forward_impl(x, w, b, eps, elementwise_affine, layout):
    """Implementation of the forward pass using JAX-Triton."""
    assert x.ndim == 3
    B, N, D = get_dims_from_input(x, layout)

    # Determine output shape based on layout
    if layout == Layout.BND_BND:
        out_shape = (B, N, D)
        TILE_N = 16
        TILE_D = 64
    elif layout == Layout.BDN_BND:
        out_shape = (B, N, D)
        TILE_N = 32
        TILE_D = 32
    elif layout == Layout.BND_BDN:
        out_shape = (B, D, N)
        TILE_N = 32
        TILE_D = 32
    elif layout == Layout.DBN_BND:
        out_shape = (B, N, D)
        TILE_N = 32
        TILE_D = 32
    elif layout == Layout.BND_DBN:
        out_shape = (D, B, N)
        TILE_N = 32
        TILE_D = 32
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    # Create output tensors
    out_dtype_struct = jax.ShapeDtypeStruct(shape=out_shape, dtype=x.dtype)
    mean_dtype_struct = jax.ShapeDtypeStruct(shape=(B, N), dtype=x.dtype)
    rstd_dtype_struct = jax.ShapeDtypeStruct(shape=(B, N), dtype=x.dtype)

    # Grid configuration
    assert D % TILE_D == 0
    grid = (triton.cdiv(N, TILE_N), B, 1)

    # Call the forward kernel
    out, mean, rstd = jt.triton_call(
        x,
        w,
        b,
        kernel=layer_norm_transpose_forward_kernel,
        out_shape=[out_dtype_struct, mean_dtype_struct, rstd_dtype_struct],
        grid=grid,
        B=B,
        N=N,
        D=D,
        EPS=eps,
        TILE_N=TILE_N,
        TILE_D=TILE_D,
        ELEMENTWISE_AFFINE=elementwise_affine,
        LAYOUT=layout.value,
    )

    return out, mean, rstd


def _layer_norm_backward_impl(
    grad_out, x, w, b, mean, rstd, eps, elementwise_affine, layout
):
    """Implementation of the backward pass using JAX-Triton."""
    assert x.ndim == 3
    B, N, D = get_dims_from_input(x, layout)

    # Determine tile sizes based on layout
    if layout == Layout.BND_BND:
        TILE_N = 16
        TILE_D = 64
    else:
        TILE_N = 32
        TILE_D = 32

    # Create gradient tensors
    grad_x_shape = x.shape
    grad_x_dtype_struct = jax.ShapeDtypeStruct(shape=grad_x_shape, dtype=x.dtype)

    num_tiles = triton.cdiv(N, TILE_N)
    grad_w_dtype_struct = jax.ShapeDtypeStruct(shape=(B, num_tiles, D), dtype=w.dtype)
    grad_b_dtype_struct = jax.ShapeDtypeStruct(shape=(B, num_tiles, D), dtype=w.dtype)

    # Grid configuration
    assert D % TILE_D == 0
    grid = (num_tiles, B, 1)

    # Call the backward kernel
    grad_x, grad_w_tiles, grad_b_tiles = jt.triton_call(
        grad_out,
        x,
        w,
        mean,
        rstd,
        kernel=layer_norm_transpose_backward_kernel,
        out_shape=[grad_x_dtype_struct, grad_w_dtype_struct, grad_b_dtype_struct],
        grid=grid,
        B=B,
        N=N,
        D=D,
        TILE_N=TILE_N,
        TILE_D=TILE_D,
        ELEMENTWISE_AFFINE=elementwise_affine,
        LAYOUT=layout.value,
    )

    # Sum gradients across tiles
    grad_w = jnp.sum(grad_w_tiles, axis=(0, 1))
    grad_b = jnp.sum(grad_b_tiles, axis=(0, 1))

    return grad_x, grad_w, grad_b


@partial(custom_vjp, nondiff_argnames=("eps", "elementwise_affine", "layout"))
def _layer_norm(x, w, b, eps=1e-5, elementwise_affine=True, layout=Layout.BND_BND):
    """JAX implementation of layer norm with custom VJP using primitives."""
    # Convert layout enum value to actual Layout enum if needed
    if isinstance(layout, int):
        layout = Layout(layout)

    # Use the primitive instead of direct implementation
    out, mean, rstd = layer_norm_fwd_p.bind(
        x, w, b, eps=eps, elementwise_affine=elementwise_affine, layout=layout
    )
    return out


def _layer_norm_fwd(x, w, b, eps, elementwise_affine, layout):
    """Forward pass for custom VJP."""
    out, mean, rstd = layer_norm_fwd_p.bind(
        x, w, b, eps=eps, elementwise_affine=elementwise_affine, layout=layout
    )
    residuals = (x, w, b, mean, rstd)
    return out, residuals


def _layer_norm_bwd(eps, elementwise_affine, layout, residuals, grad_out):
    """Backward pass for custom VJP."""
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
    )

    return grad_x, grad_w, grad_b


_layer_norm.defvjp(_layer_norm_fwd, _layer_norm_bwd)


def layer_norm_transpose(
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    layout: str = "nd->nd",
):
    """Apply fused layer normalization with support for various input layouts.

    This function performs layer normalization on the input tensor with optional
    elementwise affine transformation. It supports various input layouts and can
    transform between different tensor shapes.

    The normalization process consists of two steps:
    1. Normalize the input by subtracting mean and dividing by standard deviation
    2. Apply an affine transformation: output = weight * normalized_input + bias

    Args:
        x (jax.Array): Input tensor. Shape depends on the layout.
        w (jax.Array): Weight tensor for scaling the normalized values. Shape should be (D,).
            These weights allow the network to learn the optimal scale for each feature.
            Only used if elementwise_affine=True.
        b (jax.Array): Bias tensor for shifting the normalized values. Shape should be (D,).
            These biases allow the network to learn the optimal offset for each feature.
            Only used if elementwise_affine=True.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
        elementwise_affine (bool, optional): Whether to apply elementwise affine transformation.
            If False, weight and bias are not used (equivalent to weight=1, bias=0).
            Defaults to True.
        layout (str, optional): Input/output layout specification. Defaults to "nd->nd".
            Supported layouts:
            - "nd->nd": (N, D) -> (N, D)
            - "nd->dn": (N, D) -> (D, N)
            - "bnd->bnd": (B, N, D) -> (B, N, D)
            - "bdn->bnd": (B, D, N) -> (B, N, D)
            - "bnd->bdn": (B, N, D) -> (B, D, N)
            - "dbn->bnd": (D, B, N) -> (B, N, D)
            - "bnd->dbn": (B, N, D) -> (D, B, N)
            - "bijd->bijd": (B, I, J, D) -> (B, I, J, D)
            - "bijd->bdij": (B, I, J, D) -> (B, D, I, J)
            - "bdij->bijd": (B, D, I, J) -> (B, I, J, D)
            - "dbij->bijd": (D, B, I, J) -> (B, I, J, D)
            - "bijd->dbij": (B, I, J, D) -> (D, B, I, J)

    Returns:
        jax.Array: Normalized tensor with shape determined by the output layout.

    Raises:
        ValueError: If the specified layout is not supported.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((1, 128, 128))
        >>> w = jnp.ones(128)
        >>> b = jnp.zeros(128)
        >>> out = layer_norm_transpose(x, w, b, layout="bnd->bnd")
    """
    supported_layouts = (
        "nd->nd",
        "nd->dn",
        "dn->nd",
        "bnd->bnd",
        "bnd->bdn",
        "bdn->bnd",
        "dbn->bnd",
        "bnd->dbn",
        "bijd->bijd",
        "bijd->bdij",
        "bdij->bijd",
        "dbij->bijd",
        "bijd->dbij",
    )

    if layout == "nd->nd":
        N, D = x.shape
        B = 1
        x = x.reshape(1, N, D)
        out_shape = (N, D)
        layout = Layout.BND_BND

    elif layout == "nd->dn":
        N, D = x.shape
        B = 1
        x = x.reshape(1, N, D)
        out_shape = (D, N)
        layout = Layout.BND_BDN

    elif layout == "bnd->bnd":
        B, N, D = x.shape
        out_shape = (B, N, D)
        layout = Layout.BND_BND

    elif layout == "bdn->bnd":
        B, D, N = x.shape
        out_shape = (B, N, D)
        layout = Layout.BDN_BND

    elif layout == "bnd->bdn":
        B, N, D = x.shape
        out_shape = (B, D, N)
        layout = Layout.BND_BDN

    elif layout == "dbn->bnd":
        D, B, N = x.shape
        out_shape = (B, N, D)
        layout = Layout.DBN_BND

    elif layout == "bnd->dbn":
        B, N, D = x.shape
        out_shape = (D, B, N)
        layout = Layout.BND_DBN

    elif layout == "bijd->bijd":
        B, II, J, D = x.shape
        out_shape = (B, II, J, D)
        x = x.reshape(B, II * J, D)
        layout = Layout.BND_BND

    elif layout == "bijd->bdij":
        B, II, J, D = x.shape
        out_shape = (B, D, II, J)
        x = x.reshape(B, II * J, D)
        layout = Layout.BND_BDN

    elif layout == "bdij->bijd":
        B, D, II, J = x.shape
        out_shape = (B, II, J, D)
        x = x.reshape(B, D, II * J)
        layout = Layout.BDN_BND

    elif layout == "dbij->bijd":
        D, B, II, J = x.shape
        out_shape = (B, II, J, D)
        x = x.reshape(D, B, II * J)
        layout = Layout.DBN_BND

    elif layout == "bijd->dbij":
        B, II, J, D = x.shape
        out_shape = (D, B, II, J)
        x = x.reshape(B, II * J, D)
        layout = Layout.BND_DBN

    else:
        raise ValueError(
            f"layout {layout} not supported. supported layouts are: {supported_layouts}"
        )

    out = _layer_norm(x, w, b, eps, elementwise_affine, layout)
    return out.reshape(out_shape)
