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
import warnings

import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex

logger = logging.getLogger(__name__)


def tensor_product(
    descriptors: list[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    inputs: list[jax.Array],
    outputs_shape_dtype: list[jax.ShapeDtypeStruct],
    indices: list[jax.Array | None] | None = None,
    *,
    math_dtype: jnp.dtype | None = None,
    name: str | None = None,
    impl: str = "auto",
) -> list[jax.Array]:
    r"""Compute a polynomial described by a list of descriptors.

    Features:
      - Calls a CUDA kernel if:
          - STPs have a single mode which is a multiple of 32 (e.g. a channelwise tensor product that has subscripts ``u,u,,u`` with u=128)
          - math data type is float32 or float64
          - in/out data type is a mix of float32, float64, float16 and bfloat16
          - indices are int32
      - Supports of infinite derivatives (JVP and tranpose rules maps to a single corresponding primitive)
      - Limited support for batching (we cannot batch a buffer that has indices and if the batching is non trivial the performace will be bad)
      - Automatic optimizations based on the symmetries of the STPs and on the repetition of the input buffers
      - Automatic drop of unused buffers and indices

    Args:
        descriptors (list of pairs): The list of descriptors.
            Each descriptor is formed by a pair of :class:`cue.Operation <cuequivariance.Operation>` and :class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>`.
        inputs (list of jax.Array): The input buffers.
        outputs_shape_dtype (list of jax.ShapeDtypeStruct): The output shapes and dtypes.
        indices (list of jax.Array or None, optional): The optional indices of the inputs and outputs.
        math_dtype (jnp.dtype, optional): The data type for computational operations. Defaults to None.
        name (str, optional): The name of the operation. Defaults to None.
        impl (str, optional): The implementation to use. Defaults to "auto".
            If "auto", it will use the CUDA implementation if available, otherwise it will use the JAX implementation.
            If "cuda", it will use the CUDA implementation.
            If "jax", it will use the JAX implementation.

    Returns:
        list of jax.Array: The result of the tensor product.
    """
    warnings.warn(
        "tensor_product is deprecated and will be removed in a future version. "
        "Please use cuex.segmented_polynomial instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if name is None:
        name = "tensor_product"

    return cuex.segmented_polynomial(
        cue.SegmentedPolynomial(len(inputs), len(outputs_shape_dtype), descriptors),
        inputs,
        outputs_shape_dtype,
        indices,
        math_dtype=math_dtype,
        name=name,
        impl=impl,
    )
