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

Segmented Polynomial in PyTorch
===============================

The :class:`cuequivariance_torch.SegmentedPolynomial` class wraps a `SegmentedPolynomial` into a standard `torch.nn.Module`.

Basic Usage
-----------

.. jupyter-execute::

    import torch
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    # 1. Define a polynomial: Linear Layer (y = W @ x)
    # 2 inputs (W, x), 1 output (y)
    # Using a descriptor for convenience
    equiv_poly = cue.descriptors.linear(
        cue.Irreps("SO3", "4x0"), 
        cue.Irreps("SO3", "2x0")
    )
    sp = equiv_poly.polynomial

    # 2. Wrap in Module and Execute
    model = cuet.SegmentedPolynomial(sp, method="naive")
    
    # Inputs: [Weights, Input Vector]
    # PyTorch expects batched inputs (Batch, Dim)
    W = torch.randn(1, equiv_poly.inputs[0].dim)
    x = torch.randn(1, equiv_poly.inputs[1].dim)
    
    [y] = model([W, x])
    print(f"Output shape: {y.shape}")

High Performance (Uniform 1D)
-----------------------------

For "Uniform 1D" polynomials (all segments are same-sized 1D vectors), use ``method="uniform_1d"`` to enable optimized CUDA kernels.

.. jupyter-execute::

    # Create a Uniform 1D compatible polynomial (Element-wise product)
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

    # Initialize model with uniform_1d method
    model = cuet.SegmentedPolynomial(sp, method="uniform_1d")

    # Execute with batched inputs
    batch_size = 10
    x = torch.randn(batch_size, 32)
    y = torch.randn(batch_size, 32)
    
    [z] = model([x, y])
    print(f"Output shape: {z.shape}")

Indexing
--------

Indexing allows you to route data flexibly, such as applying different weights to different examples.
PyTorch uses dictionaries to map operand indices to index tensors.

.. jupyter-execute::

    # Example: Select from 3 weight sets for 10 input examples
    num_weights, num_examples, dim = 3, 10, 32
    weights = torch.randn(num_weights, dim)
    inputs = torch.randn(num_examples, dim)
    
    # Index tensor: Assign weight set 0 to first 3 examples, etc.
    w_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    
    # Execute
    # input_indices maps Input 0 (weights) to w_idx
    [z] = model(
        [weights, inputs],
        input_indices={0: w_idx}
    )
    
    print(f"Output shape: {z.shape}")
    
    # Output Indexing: Accumulate results into specific bins
    # We want 5 output bins. We map the 10 results to bins [0, 0, 1, 1, ...]
    out_idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    output_shape = torch.empty(5, dim)
    
    [z_accum] = model(
        [weights, inputs],
        input_indices={0: w_idx},
        output_indices={0: out_idx},
        output_shapes={0: output_shape}
    )
    
    print(f"Accumulated Output shape: {z_accum.shape}")

