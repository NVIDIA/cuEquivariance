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
    indices: list[jax.Array | None] | None = None,
    math_dtype: jnp.dtype | None = None,
    name: str | None = None,
    impl: str = "auto",
) -> tuple[cuex.RepArray, ...] | cuex.RepArray:
    """Compute an equivariant polynomial.

    Args:
        poly (:class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`): The equivariant tensor product descriptor.
        *inputs (RepArray or jax.Array): The input arrays.
        indices (list of jax.Array or None, optional): The optional indices of the inputs and output.
        output_batch_shape (tuple of int, optional): The batch shape of the output array.
        output_dtype (jnp.dtype, optional): The data type for the output array. Defaults to None.
        math_dtype (jnp.dtype, optional): The data type for computational operations. Defaults to None.
        name (str, optional): The name of the operation. Defaults to None.

    Returns:
        tuple of RepArray or RepArray: The output array(s).

    Examples:

        Let's create a descriptor for the spherical harmonics of degree 0, 1, and 2.

        >>> e = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
        >>> e
        ╭ a=1 -> B=0+1+2
        │  B ─────    sizes=9     num_segments=9     num_paths=1
        │  a B ─── ,  sizes=3,9   num_segments=3,9   num_paths=3
        ╰─ a a B ─ ,, sizes=3,3,9 num_segments=3,3,9 num_paths=8

        We need some input data.

        >>> with cue.assume(cue.SO3, cue.ir_mul):
        ...    x = cuex.RepArray("1", jnp.array([0.0, 1.0, 0.0]))
        >>> x
        {0: 1} [0. 1. 0.]

        Now we can execute the equivariant tensor product.

        >>> cuex.equivariant_polynomial(e, [x])
        {0: 0+1+2}
        [1. ... ]

        The `indices` argument allows to specify a list of optional int32 arrays for each input and for the output (`None` means no index and `indices[-1]` is the output index). The indices are used to select the elements of the input arrays and to specify the output index.
        In the following example, we will index the output. The input has a batch shape of (3,) and the output has a batch shape of (2,).

        >>> i_out = jnp.array([0, 1, 1], dtype=jnp.int32)

        The `i_out` array is used to map the result to the output indices.

        >>> with cue.assume(cue.SO3, cue.ir_mul):
        ...    x = cuex.RepArray("1", jnp.array([
        ...         [0.0, 1.0, 0.0],
        ...         [0.0, 0.0, 1.0],
        ...         [1.0, 0.0, 0.0],
        ...    ]))
        >>> cuex.equivariant_polynomial(
        ...   e,
        ...   [x],
        ...   [jax.ShapeDtypeStruct((2, e.outputs[0].dim), jnp.float32)],
        ...   indices=[None, i_out],
        ... )
        {1: 0+1+2}
        [[ 1. ... ]
         [ 2. ... ]]
    """
    if name is None:
        name = "equivariant_polynomial"

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
                x.shape[:-1] if i is None else i.shape + x.shape[1:-1]
                for i, x in zip(indices, inputs)
            ]
        )
        inferred_dtype = jnp.result_type(*inputs)
        outputs_shape_dtype = [
            jax.ShapeDtypeStruct(inferred_shape + (rep.dim,), inferred_dtype)
            for rep in poly.outputs
        ]

    if hasattr(outputs_shape_dtype, "shape"):
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
        impl=impl,
    )
    outputs = [cuex.RepArray(rep, x) for rep, x in zip(poly.outputs, outputs)]

    if poly.num_outputs == 1:
        return outputs[0]
    return tuple(outputs)
