.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Segmented Polynomial in JAX
===========================

In JAX, execution is functional. The function :func:`cuequivariance_jax.segmented_polynomial` executes an SP on JAX arrays.

Basic Usage
-----------

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    import cuequivariance as cue
    import cuequivariance_jax as cuex

    # 1. Create a polynomial (e.g. x^2)
    stp = cue.SegmentedTensorProduct.from_subscripts("i,j,ij")
    stp.add_segment(0, (3,))
    stp.add_segment(1, (3,))
    stp.add_segment(2, (3, 3))
    stp.add_path(0, 0, 0, c=1.0)
    
    x = stp.operands[0]
    sp = cue.SegmentedPolynomial(
        inputs=[x],
        outputs=[stp.operands[2]],
        operations=[(cue.Operation([0, 0, 1]), stp)]
    )

    # 2. Define output shapes/dtypes
    # We need to tell JAX what the output will look like
    output_structs = [jax.ShapeDtypeStruct((3, 3), jnp.float32)]

    # 3. Execute
    input_arr = jnp.ones((3,))
    [output_arr] = cuex.segmented_polynomial(
        sp, 
        [input_arr], 
        output_structs,
        method="naive"
    )
    
    print(f"Output shape: {output_arr.shape}")

High Performance (Uniform 1D)
-----------------------------

For "Uniform 1D" polynomials, use the ``uniform_1d`` method. This requires the polynomial to have exactly one mode (after canonicalization). Let's use a channelwise tensor product descriptor which creates a proper uniform_1d polynomial:

.. jupyter-execute::

    # Create a Uniform 1D compatible polynomial (Element-wise product)
    # x * y -> z (all vectors of size 32)
    stp = cue.SegmentedTensorProduct.from_subscripts("i,i,i")
    stp.add_segment(0, (32,))
    stp.add_segment(1, (32,))
    stp.add_segment(2, (32,))
    stp.add_path(0, 0, 0, c=1.0)
    
    sp = cue.SegmentedPolynomial(
        inputs=[stp.operands[0], stp.operands[1]], 
        outputs=[stp.operands[2]],
        operations=[(cue.Operation([0, 1, 2]), stp)]
    )
    
    # Execute with method="uniform_1d"
    x = jax.random.normal(jax.random.key(0), (32,))
    y = jax.random.normal(jax.random.key(1), (32,))
    output_struct = [jax.ShapeDtypeStruct((32,), jnp.float32)]
    
    [z] = cuex.segmented_polynomial(sp, [x, y], output_struct, method="uniform_1d")
    print(f"Output shape: {z.shape}")

    # Batched execution using vmap
    # x: (Batch, 32), y: (32,) -> z: (Batch, 32)
    batched_x = jnp.ones((10, 32))
    
    def forward(bx, y):
        [z] = cuex.segmented_polynomial(sp, [bx, y], output_struct, method="uniform_1d")
        return z
        
    batched_z = jax.vmap(forward, in_axes=(0, None))(batched_x, y)
    print(f"Batched Output shape: {batched_z.shape}")

Indexing
--------

Indexing allows you to select specific elements from input batches and write to specific locations in output batches. This is useful when different examples in your batch need different weights or when outputs should be accumulated at specific positions.

The `indices` parameter is a list with one entry per operand (inputs + outputs). Each entry can be:
- ``None``: No indexing (use all elements)
- A tuple of arrays/slices: Multi-dimensional indexing for batched inputs

.. jupyter-execute::

    # Example: Indexed weights for different examples
    # Suppose we have 3 different weight sets and 10 examples
    # We want to use weight set 0 for examples 0-2, weight set 1 for examples 3-6, etc.
    
    num_weights, num_examples, dim = 3, 10, 32
    weights = jax.random.normal(jax.random.key(0), (num_weights, dim))
    inputs = jax.random.normal(jax.random.key(1), (num_examples, dim))
    
    # Index array: Assign weight set 0 to first 3 examples, etc.
    # Shape: (10,)
    w_idx = jnp.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    
    # Execute
    # Input 0 (weights) is indexed by w_idx
    # Input 1 (inputs) is not indexed (uses corresponding batch dimension)
    output_struct = [jax.ShapeDtypeStruct((num_examples, dim), jnp.float32)]
    
    [z] = cuex.segmented_polynomial(
        sp, 
        [weights, inputs], 
        output_struct, 
        indices=[w_idx, None, None], # [Weights Index, Input Index, Output Index]
        method="uniform_1d"
    )
    
    print(f"Output shape: {z.shape}")

