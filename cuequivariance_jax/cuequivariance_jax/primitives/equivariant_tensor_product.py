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


def equivariant_tensor_product(
    e: cue.EquivariantTensorProduct,
    *inputs: cuex.RepArray | jax.Array,
    indices: list[jax.Array | None] | None = None,
    output_batch_shape: tuple[int, ...] | None = None,
    output_dtype: jnp.dtype | None = None,
    math_dtype: jnp.dtype | None = None,
    name: str | None = None,
    impl: str = "auto",
) -> cuex.RepArray:
    """Compute the equivariant tensor product of the input arrays.

    Args:
        e (:class:`cue.EquivariantTensorProduct <cuequivariance.EquivariantTensorProduct>`): The equivariant tensor product descriptor.
        *inputs (RepArray or jax.Array): The input arrays.
        indices (list of jax.Array or None, optional): The optional indices of the inputs and output.
        output_batch_shape (tuple of int, optional): The batch shape of the output array.
        output_dtype (jnp.dtype, optional): The data type for the output array. Defaults to None.
        math_dtype (jnp.dtype, optional): The data type for computational operations. Defaults to None.
        name (str, optional): The name of the operation. Defaults to None.

    Returns:
        RepArray: The result of the equivariant tensor product.

    Examples:

        Let's create a descriptor for the spherical harmonics of degree 0, 1, and 2.

        >>> e = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
        >>> e
        EquivariantTensorProduct((1)^(0..2) -> 0+1+2)

        We need some input data.

        >>> with cue.assume(cue.SO3, cue.ir_mul):
        ...    x = cuex.RepArray("1", jnp.array([0.0, 1.0, 0.0]))
        >>> x
        {0: 1} [0. 1. 0.]

        Now we can execute the equivariant tensor product.

        >>> cuex.equivariant_tensor_product(e, x)
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
        >>> cuex.equivariant_tensor_product(
        ...   e,
        ...   x,
        ...   indices=[None, i_out],
        ...   output_batch_shape=(2,),
        ... )
        {1: 0+1+2}
        [[ 1. ... ]
         [ 2. ... ]]
    """
    assert e.num_inputs > 0

    if len(inputs) == 0:
        return lambda *inputs: equivariant_tensor_product(
            e,
            *inputs,
            indices=indices,
            output_batch_shape=output_batch_shape,
            output_dtype=output_dtype,
            math_dtype=math_dtype,
            name=name,
            impl=impl,
        )

    if len(inputs) != e.num_inputs:
        raise ValueError(
            f"Unexpected number of inputs. Expected {e.num_inputs}, got {len(inputs)}."
        )

    for i, (x, rep) in enumerate(zip(inputs, e.inputs)):
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
        indices = [None] * e.num_operands

    if len(indices) != e.num_operands:
        raise ValueError(
            f"Unexpected number of indices. indices should None or a list of length {e.num_operands}, got a list of length {len(indices)}."
        )

    if output_dtype is None:
        output_dtype = jnp.result_type(*inputs)

    if output_batch_shape is None:
        if indices[-1] is not None:
            raise ValueError(
                "When output indices are provided, output_batch_shape must be provided."
            )
        output_batch_shape = jnp.broadcast_shapes(
            *[
                x.shape[:-1] if i is None else i.shape + x.shape[1:-1]
                for i, x in zip(indices, inputs)
            ]
        )

    descriptors = [(cue.Operation(e.map_operands(d.num_operands)), d) for d in e.ds]

    [x] = cuex.tensor_product(
        descriptors,
        inputs,
        [jax.ShapeDtypeStruct(output_batch_shape + (e.output.dim,), output_dtype)],
        indices,
        math_dtype=math_dtype,
        name=name,
        impl=impl,
    )

    return cuex.RepArray(e.output, x)
