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
    *inputs: cuex.IrrepsArray | jax.Array,
    dtype_output: jnp.dtype | None = None,
    dtype_math: jnp.dtype | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    algorithm: str = "sliced",
    use_custom_primitive: bool = True,
    use_custom_kernels: bool = False,
):
    """Compute the equivariant tensor product of the input arrays.

    Args:
        e (EquivariantTensorProduct): The equivariant tensor product descriptor.
        *inputs (IrrepsArray or jax.Array): The input arrays.
        dtype_output (jnp.dtype, optional): The data type for the output array. Defaults to None.
        dtype_math (jnp.dtype, optional): The data type for computational operations. Defaults to None.
        precision (jax.lax.Precision, optional): The precision for the computation. Defaults to jax.lax.Precision.HIGHEST.
        algorithm (str, optional): One of "sliced", "stacked", "compact_stacked", "indexed_compact", "indexed_vmap", "indexed_for_loop". Defaults to "sliced".
        use_custom_primitive (bool, optional): Whether to use custom JVP rules. Defaults to True.
        use_custom_kernels (bool, optional): Whether to use custom kernels. Defaults to True.

    Returns:
        IrrepsArray: The result of the equivariant tensor product.

    Examples:
        >>> e = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2])
        >>> x = cuex.IrrepsArray(cue.Irreps("SO3", "1"), jnp.array([0.0, 1.0, 0.0]), cue.ir_mul)
        >>> cuex.equivariant_tensor_product(e, x)
        {0: 0+1+2}
        [1. ... ]
    """
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

    if len(inputs) != len(e.inputs):
        raise ValueError(
            f"Unexpected number of inputs. Expected {len(e.inputs)}, got {len(inputs)}."
        )

    for x, ope in zip(inputs, e.inputs):
        if isinstance(x, cuex.IrrepsArray):
            assert x.is_simple()
            assert x.irreps() == ope.irreps
            assert x.layout == ope.layout
        else:
            assert x.ndim >= 1
            assert x.shape[-1] == ope.irreps.dim
            if not ope.irreps.is_scalar():
                raise ValueError(
                    f"Inputs should be IrrepsArray unless the input is scalar. Got {type(x)} for {ope.irreps}."
                )

    inputs = [x.array if isinstance(x, cuex.IrrepsArray) else x for x in inputs]
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

    return cuex.IrrepsArray(e.output.irreps, x, e.output.layout)
