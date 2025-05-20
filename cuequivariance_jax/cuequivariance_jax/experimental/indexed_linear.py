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
import jax
import jax.lax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex


def indexed_linear(
    poly: cue.SegmentedPolynomial,
    counts: jax.Array,
    w: jax.Array,
    x: jax.Array,
    math_dtype: jnp.dtype | None = None,
    impl: str = "auto",
) -> jax.Array:
    """Linear layer with different weights for different parts of the input.

    Args:
        poly: The polynomial descriptor. Only works for descriptors of a linear layer.
        counts: Number of elements in each partition. Shape (C,).
        w: Weights of the linear layer. Shape (C, num_weights).
        x: Input data. Shape (Z, num_inputs). Z is equal to the sum of counts.
        math_dtype: Data type for computational operations. If
            None, automatically determined from input types. Defaults to None.
        impl: Implementation to use, one of ["auto", "cuda", "jax", "naive_jax"].
            See :func:`cuex.segmented_polynomial <cuequivariance_jax.segmented_polynomial>` for more details.
            Defaults to "auto".
    Returns:
        Output data. Shape (Z, num_outputs).

    Examples:
        This example demonstrates using indexed_linear for a batch of inputs with
        different species:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import cuequivariance as cue
        >>> import cuequivariance_jax as cuex
        >>>
        >>> # Define problem parameters
        >>> num_species_total = 3  # Total number of different species
        >>> batch_size = 10        # Number of samples in batch
        >>> input_dim = 8          # Input feature dimension
        >>> output_dim = 16        # Output feature dimension
        >>> dtype = jnp.float32
        >>>
        >>> # Define how many elements belong to each species
        >>> num_species = jnp.array([3, 4, 3], dtype=jnp.int32)  # Sum equals batch_size
        >>>
        >>> # Generate random input data
        >>> input_array = jax.random.normal(jax.random.key(0), (batch_size, input_dim), dtype)
        >>>
        >>> # Define irreps for input and output features
        >>> input_irreps = cue.Irreps(cue.O3, f"{input_dim}x0e")   # Scalar features
        >>> output_irreps = cue.Irreps(cue.O3, f"{output_dim}x0e") # Scalar features
        >>>
        >>> # Create a linear descriptor
        >>> e = cue.descriptors.linear(input_irreps, output_irreps)
        >>>
        >>> # Generate weights for each species
        >>> w = jax.random.normal(jax.random.key(1), (num_species_total, e.inputs[0].dim), dtype)
        >>>
        >>> # Apply the indexed linear layer
        >>> result = cuex.experimental.indexed_linear(e.polynomial, num_species, w, input_array)
        >>>
        >>> # Verify output shape
        >>> assert result.shape == (batch_size, output_dim)
    """
    assert poly.num_inputs == 2
    assert poly.num_outputs == 1

    (C, _) = w.shape
    (Z, _) = x.shape
    assert counts.shape == (C,)

    y = jax.ShapeDtypeStruct((Z, poly.outputs[0].size), x.dtype)
    [y] = cuex.segmented_polynomial(
        poly,
        [w, x],
        [y],
        [cuex.Repeats(counts), None, None],
        math_dtype=math_dtype,
        impl=impl,
    )
    return y
