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

import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex


def equivariant_polynomial(
    poly: cue.EquivariantPolynomial,
    inputs: list[cuex.RepArray | jax.Array],
    outputs_shape_dtype: list[jax.ShapeDtypeStruct]
    | jax.ShapeDtypeStruct
    | None = None,
    indices: None | list[None | jax.Array | tuple[jax.Array | slice]] = None,
    *,
    method: str = "",
    math_dtype: str | None = None,
    name: str | None = None,
    precision: jax.lax.Precision = "undefined",
) -> list[cuex.RepArray] | cuex.RepArray:
    """Compute an equivariant polynomial.

    Evaluates an equivariant polynomial, which represents a mathematical operation that
    respects group symmetries. This function is the equivariant wrapper around the
    :func:`cuex.segmented_polynomial <cuequivariance_jax.segmented_polynomial>` function,
    providing type checking and handling for representation-aware arrays.

    Args:
        poly: The equivariant polynomial descriptor.
        inputs: List of input :class:`cuex.RepArray <cuequivariance_jax.RepArray>`.
        outputs_shape_dtype: Shape and dtype specifications for outputs. If None,
            inferred from inputs when possible. When output indices are provided, this must be specified.
            The last shape dimension can be set to -1 to infer the size from the polynomial descriptor.
        indices: Optional list of indices for inputs and outputs. Length must match
            total number of operands (inputs + outputs). Use None for unindexed
            operands. Defaults to None. Note that indices are not supported for all methods.
        method: Method to use for computation. See :func:`cuex.segmented_polynomial <cuequivariance_jax.segmented_polynomial>` for available methods.
        math_dtype: See :func:`cuex.segmented_polynomial <cuequivariance_jax.segmented_polynomial>` for supported options.
        name: Optional name for the operation. Defaults to None.

    Returns:
        :class:`cuex.RepArray <cuequivariance_jax.RepArray>` or list of :class:`cuex.RepArray <cuequivariance_jax.RepArray>`

    Note:
        See :func:`cuex.segmented_polynomial <cuequivariance_jax.segmented_polynomial>` for more details on the
        implementation and usage of the underlying CUDA and JAX implementations.

    Examples:
        Create and compute spherical harmonics of degree 0, 1, and 2:

        >>> e = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
        >>> e
        ╭ a=1 -> B=0+1+2
        │  []➜B[] ───────── num_paths=1
        │  []·a[]➜B[] ───── num_paths=3
        ╰─ []·a[]·a[]➜B[] ─ num_paths=11

        Basic usage with single input:

        >>> with cue.assume(cue.SO3, cue.ir_mul):
        ...    x = cuex.RepArray("1", jnp.array([0.0, 1.0, 0.0]))
        >>> cuex.equivariant_polynomial(e, [x], method="naive")
        {0: 0+1+2}
        [1. ... ]

        Using indices:

        >>> i_out = jnp.array([0, 1, 1], dtype=jnp.int32)
        >>> with cue.assume(cue.SO3, cue.ir_mul):
        ...    x = cuex.RepArray("1", jnp.array([
        ...         [0.0, 1.0, 0.0],
        ...         [0.0, 0.0, 1.0],
        ...         [1.0, 0.0, 0.0],
        ...    ]))
        >>> result = cuex.equivariant_polynomial(
        ...   e,
        ...   [x],
        ...   jax.ShapeDtypeStruct((2, e.outputs[0].dim), jnp.float32),
        ...   indices=[None, i_out],
        ...   method="naive",
        ... )
        >>> result
        {1: 0+1+2}
        [[ 1. ... ]
         [ 2. ... ]]
    """
    if name is None:
        name = "equivariant_polynomial"

    if precision != "undefined":
        raise ValueError(
            "precision is not anymore supported. Please use math_dtype instead."
        )

    if len(inputs) != poly.num_inputs:
        raise ValueError(
            f"Unexpected number of inputs. Expected {poly.num_inputs}, got {len(inputs)}."
        )

    for i, (x, rep) in enumerate(zip(inputs, poly.inputs)):
        if isinstance(x, cuex.RepArray):
            assert x.rep(-1) == rep, (
                f"Input {i} should have representation {rep}, got {x.rep(-1)}."
            )
        else:
            assert x.ndim >= 1, (
                f"Input {i} should have at least one dimension, got {x.ndim}."
            )
            assert x.shape[-1] == rep.dim, (
                f"Input {i} should have dimension {rep.dim}, got {x.shape[-1]}."
            )
            if not rep.is_scalar():
                raise ValueError(
                    f"Input {i} should be a RepArray unless the input is scalar. Got {type(x)} for {rep}."
                )

    inputs: list[jax.Array] = [getattr(x, "array", x) for x in inputs]

    if indices is None:
        indices = [None] * poly.num_operands

    if len(indices) != poly.num_operands:
        raise ValueError(
            f"Unexpected number of indices. indices should None or a list of length {poly.num_operands}, got a list of length {len(indices)}."
        )

    return_as_list = True

    if outputs_shape_dtype is None:
        if not all(i is None for i in indices[poly.num_inputs :]):
            raise ValueError(
                "When output indices are provided, outputs_shape_dtype must be provided."
            )
        if poly.num_inputs == 0:
            raise ValueError(
                "When no inputs are provided, outputs_shape_dtype must be provided."
            )
        inferred_shape = jnp.broadcast_shapes(
            *[
                x.shape[:-1]
                if i is None
                else (
                    (i.total_repeat_length,) + x.shape[1:-1]
                    if isinstance(i, cuex.Repeats)
                    else i.shape + x.shape[1:-1]
                )
                for i, x in zip(indices, inputs)
                if not isinstance(i, cuex.Repeats) or i.total_repeat_length is not None
            ]
        )
        inferred_dtype = jnp.result_type(*inputs)
        outputs_shape_dtype = [
            jax.ShapeDtypeStruct(inferred_shape + (rep.dim,), inferred_dtype)
            for rep in poly.outputs
        ]
        if poly.num_outputs == 1:
            return_as_list = False

    if hasattr(outputs_shape_dtype, "shape"):
        return_as_list = False
        outputs_shape_dtype = [outputs_shape_dtype]

    if len(outputs_shape_dtype) != poly.num_outputs:
        raise ValueError(
            f"Unexpected number of outputs. Expected {poly.num_outputs}, got {len(outputs_shape_dtype)}."
        )

    outputs = cuex.segmented_polynomial(
        poly.polynomial,
        inputs,
        outputs_shape_dtype,
        indices,
        math_dtype=math_dtype,
        name=name,
        method=method,
    )
    outputs = [cuex.RepArray(rep, x) for rep, x in zip(poly.outputs, outputs)]

    if not return_as_list:
        return outputs[0]
    return outputs
