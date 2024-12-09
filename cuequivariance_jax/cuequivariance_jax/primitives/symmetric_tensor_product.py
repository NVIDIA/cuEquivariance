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
import logging

import jax
import jax.numpy as jnp

import cuequivariance_jax as cuex
from cuequivariance import segmented_tensor_product as stp

logger = logging.getLogger(__name__)


def symmetric_tensor_product(
    ds: list[stp.SegmentedTensorProduct],
    *inputs: jax.Array,
    dtype_output: jnp.dtype | None = None,
    dtype_math: jnp.dtype | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    algorithm: str = "sliced",
    use_custom_primitive: bool = True,
    use_custom_kernels: bool = False,
) -> jax.Array:
    """
    Compute the sum of the STPs evaluated on the input (all input operands are the same).

    Args:
        ds (list[stp.SegmentedTensorProduct]): The segmented tensor product descriptors.
        *inputs (jax.Array): The input arrays. The last input is repeated to match the number of input operands of each STP.
        dtype_output (jnp.dtype, optional): The data type for the output array.
        dtype_math (jnp.dtype, optional): The data type for mathematical operations.
        precision (jax.lax.Precision, optional): The precision for the computation. Defaults to jax.lax.Precision.HIGHEST.
        algorithm (str, optional): One of "sliced", "stacked", "compact_stacked", "indexed_compact", "indexed_vmap", "indexed_for_loop". Defaults to "sliced".
        use_custom_primitive (bool, optional): Whether to use custom JVP rules. Defaults to True.
        use_custom_kernels (bool, optional): Whether to use custom kernels. Defaults to True.

    Returns:
        jax.Array: The result of the tensor product computation.

    See Also:
        :func:`SegmentedTensorProduct <cuequivariance.segmented_tensor_product.SegmentedTensorProduct>`
    """
    assert any(d.num_operands >= 2 for d in ds)

    # currying
    if len(inputs) == 0:

        def fn(*inputs) -> jax.Array:
            return symmetric_tensor_product(
                ds,
                *inputs,
                dtype_output=dtype_output,
                dtype_math=dtype_math,
                precision=precision,
                algorithm=algorithm,
                use_custom_primitive=use_custom_primitive,
                use_custom_kernels=use_custom_kernels,
            )

        return fn

    # vet STPs
    operands_in: list[stp.Operand] = []
    operand_out: stp.Operand | None = None
    for d in ds:
        for oid, operand in enumerate(d.operands[:-1]):
            if oid >= len(operands_in):
                operands_in.append(operand)
            else:
                if operands_in[oid].size != operand.size:
                    raise ValueError(
                        f"cuex.symmetric_tensor_product: operand {oid} size mismatch"
                    )
        if operand_out is None:
            operand_out = d.operands[-1]
        else:
            if operand_out.size != d.operands[-1].size:
                raise ValueError(
                    "cuex.symmetric_tensor_product: output operand size mismatch"
                )

    # vet inputs
    for oid, input in enumerate(inputs):
        if input.ndim == 0:
            raise ValueError(
                f"cuex.symmetric_tensor_product: input {oid} has zero dimensions"
            )
        if input.shape[-1] != d.operands[oid].size:
            raise ValueError(
                f"cuex.tensor_product: expected operand {oid} to have size {d.operands[oid].size}, got {input.shape[-1]}"
            )

    # symmetrize STPs
    n_un = len(inputs) - 1
    ds = [d.symmetrize_operands(range(n_un, d.num_operands - 1)) for d in ds]

    # set default options
    if dtype_output is None:
        dtype_output = jnp.result_type(*inputs)
    if dtype_math is None:
        if dtype_output.itemsize <= jnp.dtype(jnp.float32).itemsize:
            dtype_math = jnp.float32
        else:
            dtype_math = dtype_output

    unique_inputs = inputs[:n_un]
    repeated_input = inputs[n_un]

    output = 0
    for d in ds:
        n_in = d.num_operands - 1

        if n_in > n_un:
            d_inputs = unique_inputs + (repeated_input,) * (n_in - n_un)
        else:
            d_inputs = unique_inputs[:n_in]

        output += cuex.tensor_product(
            d,
            *d_inputs,
            dtype_output=dtype_output,
            dtype_math=dtype_math,
            precision=precision,
            algorithm=algorithm,
            use_custom_primitive=use_custom_primitive,
            use_custom_kernels=use_custom_kernels,
        )

    return output
