# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    dtype_output: jnp.dtype | None = None,
    dtype_math: jnp.dtype | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    algorithm: str = "sliced",
    use_custom_primitive: bool = True,
    use_custom_kernels: bool = False,
) -> cuex.RepArray:
    """Compute the equivariant tensor product of the input arrays.

    Args:
        e (EquivariantTensorProduct): The equivariant tensor product descriptor.
        *inputs (RepArray or jax.Array): The input arrays.
        dtype_output (jnp.dtype, optional): The data type for the output array. Defaults to None.
        dtype_math (jnp.dtype, optional): The data type for computational operations. Defaults to None.
        precision (jax.lax.Precision, optional): The precision for the computation. Defaults to ``jax.lax.Precision.HIGHEST``.
        algorithm (str, optional): One of "sliced", "stacked", "compact_stacked", "indexed_compact", "indexed_vmap", "indexed_for_loop". Defaults to "sliced". See :class:`cuex.tensor_product <cuequivariance_jax.tensor_product>` for more information.
        use_custom_primitive (bool, optional): Whether to use custom JVP rules. Defaults to True.
        use_custom_kernels (bool, optional): Whether to use custom kernels. Defaults to True.

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
    """
    assert e.num_inputs > 0

    if len(inputs) == 0:
        return lambda *inputs: equivariant_tensor_product(
            e,
            *inputs,
            dtype_output=dtype_output,
            dtype_math=dtype_math,
            precision=precision,
            algorithm=algorithm,
            use_custom_primitive=use_custom_primitive,
            use_custom_kernels=use_custom_kernels,
        )

    if len(inputs) != e.num_inputs:
        raise ValueError(
            f"Unexpected number of inputs. Expected {e.num_inputs}, got {len(inputs)}."
        )

    for x, rep in zip(inputs, e.inputs):
        if isinstance(x, cuex.RepArray):
            assert x.rep(-1) == rep
        else:
            assert x.ndim >= 1
            assert x.shape[-1] == rep.dim
            if not rep.is_scalar():
                raise ValueError(
                    f"Inputs should be RepArray unless the input is scalar. Got {type(x)} for {rep}."
                )

    inputs: list[jax.Array] = [getattr(x, "array", x) for x in inputs]

    x = cuex.symmetric_tensor_product(
        e.ds,
        *inputs,
        dtype_output=dtype_output,
        dtype_math=dtype_math,
        precision=precision,
        algorithm=algorithm,
        use_custom_primitive=use_custom_primitive,
        use_custom_kernels=use_custom_kernels,
    )

    return cuex.RepArray(e.output, x)
