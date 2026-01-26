# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utilities for working with dict[Irrep, Array] representation.

This module provides an alternative approach to RepArray for handling
equivariant data. Instead of using a single contiguous array with metadata,
features are stored as a dictionary mapping irreducible representations (Irreps)
to their corresponding arrays.

The dict[Irrep, Array] representation has several advantages:
- More flexible memory layout (arrays don't need to be contiguous)
- Works naturally with jax.tree operations
- Compatible with split_operand_by_irrep for segmented_polynomial_uniform_1d

Typical array shapes follow the convention: (..., multiplicity, irrep_dim)
where the last two dimensions are (mul, ir.dim).

Example:
    >>> import cuequivariance as cue
    >>> irreps = cue.Irreps(cue.O3, "128x0e + 64x1o")
    >>> batch = 32
    >>> # Create dict representation
    >>> x = {cue.O3(0, 1): jnp.ones((batch, 128, 1)),
    ...      cue.O3(1, -1): jnp.ones((batch, 64, 3))}
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

import cuequivariance as cue
from cuequivariance import Irrep

from .segmented_polynomials.segmented_polynomial import segmented_polynomial

__all__ = [
    "segmented_polynomial_uniform_1d",
    "assert_mul_ir_dict",
    "mul_ir_dict",
    "flat_to_dict",
    "dict_to_flat",
    "irreps_add",
    "irreps_zeros_like",
]


def segmented_polynomial_uniform_1d(
    polynomial: cue.SegmentedPolynomial,
    inputs: Any,
    outputs: Any = None,
    input_indices: Any = None,
    output_indices: Any = None,
    *,
    math_dtype: Any = None,
    name: str | None = None,
) -> Any:
    """Execute a segmented polynomial with uniform 1D method on tree-structured inputs/outputs.

    This function wraps cuex.segmented_polynomial with method="uniform_1d", handling
    the flattening/unflattening of pytree-structured inputs and outputs. It's designed
    to work with dict[Irrep, Array] representations where each array has shape
    (..., num_segments, *segment_shape).

    Args:
        polynomial: The segmented polynomial to execute.
        inputs: Pytree of input arrays. Leaves must have shape
            (..., num_segments, *segment_shape) matching the polynomial's input descriptors.
        outputs: Pytree of output arrays or ShapeDtypeStruct, or None for default zeros.
            Must have shape (..., num_segments, *segment_shape) matching output descriptors.
        input_indices: Pytree matching inputs structure with index arrays for gather operations,
            or None for no indexing. Broadcast to match inputs structure.
        output_indices: Pytree matching outputs structure with index arrays for scatter operations,
            or None for no indexing. Broadcast to match outputs structure.
        math_dtype: Optional dtype for internal computation.
        name: Optional name for profiling/debugging.

    Returns:
        Pytree with same structure as outputs, containing computed results with shape
        (..., num_segments, *segment_shape).

    Example:
        >>> # After split_operand_by_irrep, inputs/outputs are dict[Irrep, Array]
        >>> e = descriptor.split_operand_by_irrep(1).split_operand_by_irrep(-1)  # doctest: +SKIP
        >>> p = e.polynomial  # doctest: +SKIP
        >>> y = segmented_polynomial_uniform_1d(  # doctest: +SKIP
        ...     p, [w, x], y,
        ...     input_indices=[None, senders],
        ...     output_indices=receivers,
        ... )
    """

    def is_none(x):
        return x is None

    assert len(jax.tree.leaves(inputs, is_none)) == polynomial.num_inputs
    assert len(jax.tree.leaves(outputs, is_none)) == polynomial.num_outputs

    input_indices = jax.tree.broadcast(input_indices, inputs, is_none)
    output_indices = jax.tree.broadcast(output_indices, outputs, is_none)

    def flatten_input(i: int, desc: cue.SegmentedOperand, x: Array) -> Array:
        if not desc.all_same_segment_shape():
            raise ValueError(
                f"Input operand {i}: segments must have uniform shape.\n"
                f"  Descriptor: {desc}\n"
                f"  Segment shapes: {desc.segments}"
            )
        expected_suffix = (desc.num_segments,) + desc.segment_shape
        min_ndim = 1 + desc.ndim
        if x.ndim < min_ndim:
            raise ValueError(
                f"Input operand {i}: array has too few dimensions.\n"
                f"  Expected at least {min_ndim} dims (batch... + {expected_suffix})\n"
                f"  Got shape {x.shape} with {x.ndim} dims\n"
                f"  Descriptor: num_segments={desc.num_segments}, "
                f"segment_shape={desc.segment_shape}"
            )
        actual_suffix = x.shape[-(1 + desc.ndim) :]
        if actual_suffix != expected_suffix:
            raise ValueError(
                f"Input operand {i}: shape mismatch in trailing dimensions.\n"
                f"  Expected trailing dims: {expected_suffix} "
                f"(num_segments={desc.num_segments}, segment_shape={desc.segment_shape})\n"
                f"  Got trailing dims: {actual_suffix}\n"
                f"  Full array shape: {x.shape}\n"
                f"  Descriptor: {desc}"
            )
        return jnp.reshape(x, x.shape[: -(1 + desc.ndim)] + (desc.size,))

    list_inputs = jax.tree.leaves(inputs, is_none)
    assert all(isinstance(x, Array) for x in list_inputs)
    list_inputs = [
        flatten_input(i, desc, x)
        for i, (desc, x) in enumerate(zip(polynomial.inputs, list_inputs))
    ]

    shapes = []
    dtypes = []
    for x, i in zip(list_inputs, jax.tree.leaves(input_indices, is_none)):
        if i is None:
            shapes.append(x.shape[:-1])
        else:
            shapes.append(i.shape + x.shape[1:-1])
        dtypes.append(x.dtype)

    default_shape = jnp.broadcast_shapes(*shapes)
    default_dtype = jnp.result_type(*dtypes)

    def flatten_output(
        desc: cue.SegmentedOperand, x: Array | jax.ShapeDtypeStruct | None
    ) -> Array | None:
        assert desc.all_same_segment_shape()
        if isinstance(x, jax.ShapeDtypeStruct):
            x = jnp.zeros(x.shape, x.dtype)
        if x is None:
            x = jnp.zeros(
                default_shape + (desc.num_segments,) + desc.segment_shape, default_dtype
            )
        assert x.ndim >= 1 + desc.ndim, f"desc: {desc}, x.shape: {x.shape}"
        assert (
            x.shape[-(1 + desc.ndim) :] == (desc.num_segments,) + desc.segment_shape
        ), f"desc: {desc}, x.shape: {x.shape}"
        return jnp.reshape(x, x.shape[: -(1 + desc.ndim)] + (desc.size,))

    list_outputs = jax.tree.leaves(outputs, is_none)
    list_outputs = [
        flatten_output(desc, x) for desc, x in zip(polynomial.outputs, list_outputs)
    ]

    list_indices = jax.tree.leaves(input_indices, is_none) + jax.tree.leaves(
        output_indices, is_none
    )
    list_outputs = segmented_polynomial(
        polynomial,
        list_inputs,
        list_outputs,
        list_indices,
        method="uniform_1d",
        math_dtype=math_dtype,
        name=name,
    )

    def unflatten_output(desc: cue.SegmentedOperand, x: Array) -> Array:
        return jnp.reshape(x, x.shape[:-1] + (desc.num_segments,) + desc.segment_shape)

    list_outputs = [
        unflatten_output(desc, x) for desc, x in zip(polynomial.outputs, list_outputs)
    ]
    return jax.tree.unflatten(jax.tree.structure(outputs, is_none), list_outputs)


def assert_mul_ir_dict(irreps: cue.Irreps, x: dict[Irrep, Array]) -> None:
    """Assert that a dict[Irrep, Array] matches the expected irreps structure.

    Args:
        irreps: Expected irreps specification.
        x: Dictionary mapping Irreps to arrays.

    Raises:
        AssertionError: If keys don't match or array shapes are incorrect.
            Expected shape for each array is (..., multiplicity, irrep_dim).
    """
    error_msg = (
        f"Dict {jax.tree.map(lambda v: v.shape, x)} does not match irreps {irreps}"
    )
    for (expected_mul, expected_ir), (actual_ir, actual_v) in zip(irreps, x.items()):
        assert actual_ir == expected_ir, error_msg
        assert actual_v.shape[-2:] == (expected_mul, expected_ir.dim), error_msg


def mul_ir_dict(irreps: cue.Irreps, data: Any) -> dict[Irrep, Any]:
    """Create a dict[Irrep, data] by broadcasting data to match irreps structure.

    Useful for creating output templates or broadcasting scalar values across irreps.

    Args:
        irreps: Irreps specification defining the dict keys.
        data: Data to broadcast to each irrep key.

    Returns:
        Dictionary with irrep keys, each value set to data.

    Example:
        >>> import cuequivariance as cue
        >>> irreps = cue.Irreps(cue.O3, "128x0e + 64x1o")
        >>> batch = 32
        >>> template = mul_ir_dict(irreps, jax.ShapeDtypeStruct((batch,), jnp.float32))
    """
    return jax.tree.broadcast(data, {ir: None for _, ir in irreps}, lambda v: v is None)


def flat_to_dict(
    irreps: cue.Irreps, data: Array, *, layout: str = "mul_ir"
) -> dict[Irrep, Array]:
    """Convert a flat array to dict[Irrep, Array] with shape (..., mul, ir.dim).

    Splits a contiguous array along the last axis into separate arrays per irrep,
    reshaping each to have explicit (multiplicity, irrep_dim) dimensions.

    Args:
        irreps: Irreps specification for splitting.
        data: Flat array with shape (..., irreps.dim).
        layout: Memory layout of the flat data. Either "mul_ir" (default) where
            data is ordered as (mul, ir.dim), or "ir_mul" where data is ordered
            as (ir.dim, mul).

    Returns:
        Dictionary mapping each irrep to array with shape (..., mul, ir.dim).

    Example:
        >>> import cuequivariance as cue
        >>> irreps = cue.Irreps(cue.O3, "128x0e + 64x1o")
        >>> batch = 32
        >>> flat = jnp.ones((batch, irreps.dim))
        >>> d = flat_to_dict(irreps, flat)
        >>> d[cue.O3(0, 1)].shape
        (32, 128, 1)
        >>> d[cue.O3(1, -1)].shape
        (32, 64, 3)
    """
    assert layout in ("mul_ir", "ir_mul")
    result = {}
    offset = 0
    for mul, ir in irreps:
        size = mul * ir.dim
        segment = data[..., offset : offset + size]
        if layout == "mul_ir":
            result[ir] = jnp.reshape(segment, data.shape[:-1] + (mul, ir.dim))
        else:  # ir_mul
            result[ir] = jnp.reshape(segment, data.shape[:-1] + (ir.dim, mul))
            result[ir] = jnp.swapaxes(result[ir], -2, -1)
        offset += size
    return result


def dict_to_flat(irreps: cue.Irreps, x: dict[Irrep, Array]) -> Array:
    """Convert dict[Irrep, Array] back to a flat contiguous array.

    Flattens the (multiplicity, irrep_dim) dimensions and concatenates all irreps.

    Args:
        irreps: Irreps specification defining the order.
        x: Dictionary with arrays of shape (..., mul, ir.dim).

    Returns:
        Flat array with shape (..., irreps.dim).

    Example:
        >>> import cuequivariance as cue
        >>> irreps = cue.Irreps(cue.O3, "128x0e + 64x1o")
        >>> batch = 32
        >>> d = {cue.O3(0, 1): jnp.ones((batch, 128, 1)),
        ...      cue.O3(1, -1): jnp.ones((batch, 64, 3))}
        >>> flat = dict_to_flat(irreps, d)
        >>> flat.shape
        (32, 320)
    """
    arrays = []
    for mul, ir in irreps:
        v = x[ir]
        arrays.append(jnp.reshape(v, v.shape[:-2] + (mul * ir.dim,)))
    return jnp.concatenate(arrays, axis=-1)


def irreps_add(x: dict[Irrep, Array], y: dict[Irrep, Array]) -> dict[Irrep, Array]:
    """Element-wise addition of two dict[Irrep, Array] representations.

    Args:
        x: First dictionary with arrays of shape (..., mul, ir.dim).
        y: Second dictionary with same keys and compatible shapes.

    Returns:
        Dictionary with element-wise sum of corresponding arrays.

    Raises:
        AssertionError: If the dictionaries have different keys.
    """
    assert x.keys() == y.keys()
    return {ir: x[ir] + y[ir] for ir in x.keys()}


def irreps_zeros_like(x: dict[Irrep, Array]) -> dict[Irrep, Array]:
    """Create a dict[Irrep, Array] of zeros with the same structure.

    Args:
        x: Template dictionary with arrays of shape (..., mul, ir.dim).

    Returns:
        Dictionary with zero arrays of the same shapes and dtypes.
    """
    return {ir: jnp.zeros_like(v) for ir, v in x.items()}
