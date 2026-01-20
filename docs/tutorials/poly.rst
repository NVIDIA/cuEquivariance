.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Segmented Polynomial
====================

From Math to Execution
----------------------

The :class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>` (STP) gave us the mathematical blueprint for a contraction. But a blueprint isn't a building. To actually execute this math, we need to connect it to real data.

The :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` (SP) acts as the **circuit board**. It defines:

1.  **Global Memory**: The actual inputs and outputs of your function.
2.  **Wiring**: How these inputs connect to the STP blueprints.

Like the STP, **SP is agnostic to group theory**. It is a general-purpose engine for executing computations on segmented tensors.

The Dataflow (The Wiring)
-------------------------

An SP is a collection of operations. Each operation pairs an STP (the math) with a wiring instruction.

.. code-block:: python

    (operation, stp)

The wiring tells the system: "Take Global Input #0 and plug it into STP Operand #1."

Example: Wiring a Square Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say we want to compute :math:`y = x \otimes x` (the tensor product of :math:`x` with itself).
*   **The Math (STP)**: Requires two input operands (Left, Right) and produces one output.
*   **The Circuit (SP)**: Has only *one* global input (:math:`x`). We need to wire this single input to *both* the Left and Right operands of the STP.

.. jupyter-execute::

    import cuequivariance as cue
    import numpy as np
    from cuequivariance.segmented_polynomials import visualize_polynomial

    # 1. Create the STP (The Math)
    # A simple contraction: A_i * B_j -> C_ij
    stp = cue.SegmentedTensorProduct.from_subscripts("i,j,ij")
    stp.add_segment(0, (3,))  # Left operand: size 3
    stp.add_segment(1, (3,))  # Right operand: size 3
    stp.add_segment(2, (3, 3))  # Output segment: shape (3, 3) for modes i,j
    stp.add_path(0, 0, 0, c=1.0)  # Scalar coefficient

    # 2. Create the SP (The Wiring)
    # Global Inputs: [x] (We define x using the shape from the STP)
    x = stp.operands[0]
    
    # The Operation defines the wiring: [0, 0, 1]
    #   - STP Operand 0 gets Global Input 0 (x)
    #   - STP Operand 1 gets Global Input 0 (x) -- REUSE!
    #   - STP Operand 2 becomes the Output
    
    sp = cue.SegmentedPolynomial(
        inputs=[x],
        outputs=[stp.operands[2]],
        operations=[(cue.Operation([0, 0, 1]), stp)]
    )

    print(sp)

Visualization
-------------

Dataflow can be hard to read from text. Visualizing the graph makes the "wiring" obvious.

.. jupyter-execute::

    # Visualize the polynomial
    # Blue nodes: Inputs
    # Yellow nodes: The STP computation
    # Green nodes: Outputs
    graph = visualize_polynomial(sp, input_names=["x"], output_names=["y"])
    graph

In the diagram, you can clearly see the single input node ``x`` splitting into two branches to feed the yellow computation node. This visually confirms we are computing a quadratic function :math:`x^2`.

Automatic Differentiation (AD)
------------------------------

One of the most powerful features of SP is that it knows how to differentiate itself.
Since SP defines the entire dataflow graph, it can apply the rules of calculus (like the product rule) to generate new SPs that compute gradients.

Forward Mode (JVP)
~~~~~~~~~~~~~~~~~~

:meth:`cue.SegmentedPolynomial.jvp` (Jacobian-Vector Product) computes the directional derivative.
If our SP calculates :math:`y = x^2`, the JVP will calculate :math:`dy = 2x \cdot dx`.

.. jupyter-execute::

    # Compute JVP with respect to input 0
    sp_jvp, mapping = sp.jvp([True])
    print(sp_jvp)

The output shows a larger, more complex graph. It now handles two types of signals: values (:math:`x`) and tangents (:math:`dx`).

Reverse Mode (Backpropagation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`cue.SegmentedPolynomial.backward` is the high-level method for backpropagation. It combines forward and reverse mode to compute gradients.
It takes gradients from the output (:math:`dL/dy`) and computes gradients for the inputs (:math:`dL/dx`).

.. jupyter-execute::

    # Compute backward pass
    # requires_gradient=[True]: Input 0 needs gradients
    # has_cotangent=[True]: Output 0 has a gradient coming in
    sp_bwd, mapping = sp.backward(
        requires_gradient=[True], 
        has_cotangent=[True]
    )
    
    print(sp_bwd)
    
    # Visualize the backward pass
    # The mapping function helps rename operands for clarity
    operand_names = (["x"], ["y"])
    operand_names_bwd = mapping(operand_names, lambda n: f"d{n}")
    
    graph = visualize_polynomial(sp_bwd, input_names=operand_names_bwd[0], output_names=operand_names_bwd[1])
    graph

Performance: The "Uniform 1D" Case
----------------------------------

While SP can handle complex, ragged, sparse data, there is a special case that is extremely fast.

We call it **Uniform 1D**.
This happens when every operand is made of segments that are:
1.  **Uniform**: All segments in the operand have the same shape.
2.  **1D**: That shape is just a vector ``(d,)`` (or a scalar ``()``).

**Why does this matter?**
If your data is "Uniform 1D", it fits into regular tensors. This means we don't need slow, sparse lookups. We can use highly optimized code:
*   **Vectorization**: Using ``vmap`` in JAX or PyTorch.
*   **CUDA Kernels**: We provide specialized GPU kernels for this case that are very fast.

Most standard Neural Network layers (Linear, Convolution, Tensor Product) fall into this category.

.. jupyter-execute::

    # Check if our example is Uniform 1D
    # Uniform 1D means: all segments have same shape AND shape is 1D (or scalar)
    is_uniform_1d = all(
        op.all_same_segment_shape() and op.ndim <= 1 
        for op in sp.operands
    )
    print(f"Is squared tensor product uniform 1D? {is_uniform_1d}")

    # Let's create a Uniform 1D example: Element-wise product
    # x * y -> z (all vectors of size 5)
    stp_1d = cue.SegmentedTensorProduct.from_subscripts("i,i,i")
    stp_1d.add_segment(0, (5,))
    stp_1d.add_segment(1, (5,))
    stp_1d.add_segment(2, (5,))
    stp_1d.add_path(0, 0, 0, c=1.0)
    
    sp_1d = cue.SegmentedPolynomial(
        inputs=[stp_1d.operands[0], stp_1d.operands[1]],
        outputs=[stp_1d.operands[2]],
        operations=[(cue.Operation([0, 1, 2]), stp_1d)]
    )

    is_uniform_1d = all(
        op.all_same_segment_shape() and op.ndim <= 1 
        for op in sp_1d.operands
    )
    print(f"Is element-wise product uniform 1D? {is_uniform_1d}")

If you are building standard models, you will mostly stay in this high-performance regime.

Framework Guides
----------------

Now that you understand the concepts, see how to run these polynomials in your framework of choice:

*   :doc:`Using Segmented Polynomials in PyTorch <pytorch/poly>`
*   :doc:`Using Segmented Polynomials in JAX <jax/poly>`
