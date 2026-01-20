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

Now that you understand the concepts, see how to run these polynomials in your framework of choice.

First, let's create a linear layer descriptor that we'll use in the following examples:

.. jupyter-execute::

    # Create a linear layer descriptor
    e = cue.descriptors.linear(
        cue.Irreps("SO3", "4x0"),
        cue.Irreps("SO3", "2x0")
    )
    print(e)

Execution on JAX
~~~~~~~~~~~~~~~~

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import cuequivariance as cue
    import cuequivariance_jax as cuex

    w = cuex.randn(jax.random.key(0), e.inputs[0])
    x = cuex.randn(jax.random.key(1), e.inputs[1])

    cuex.equivariant_polynomial(e, [w, x], method="uniform_1d")

The function :func:`cuex.randn <cuequivariance_jax.randn>` generates random :class:`cuex.RepArray <cuequivariance_jax.RepArray>` objects.
The function :func:`cuex.equivariant_polynomial <cuequivariance_jax.equivariant_polynomial>` executes the tensor product.
The output is a :class:`cuex.RepArray <cuequivariance_jax.RepArray>` object.

Execution on PyTorch
~~~~~~~~~~~~~~~~~~~~

The same descriptor can be used in PyTorch using the class :class:`cuet.SegmentedPolynomial <cuequivariance_torch.SegmentedPolynomial>`.

.. jupyter-execute::

    import torch
    import cuequivariance_torch as cuet

    module = cuet.SegmentedPolynomial(e.polynomial, method="uniform_1d")

    w = torch.randn(1, e.inputs[0].dim)
    x = torch.randn(1, e.inputs[1].dim)

    module([w, x])

Details
-------

An :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>` is composed of two main components:

1. Lists of :class:`cue.Rep <cuequivariance.Rep>` objects that define the inputs and outputs of the polynomial
2. A :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` that describes how to compute the polynomial

The :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` itself consists of:

* A list of :class:`cue.SegmentedOperand <cuequivariance.SegmentedOperand>` objects that represent the operands used in the computation
* A list of operations, where each operation is a pair containing:
    * An :class:`cue.Operation <cuequivariance.Operation>` object that defines what operation to perform
    * A :class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>` that specifies how to perform the tensor product

This hierarchical structure allows for efficient representation and computation of equivariant polynomials. Below we can examine these components for a specific example:

.. jupyter-execute::

    e.inputs, e.outputs

.. jupyter-execute::

    p = e.polynomial
    p

.. jupyter-execute::

    p.inputs, p.outputs

.. jupyter-execute::

    p.operations

Visualization
-------------

You can visualize the dataflow of a :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` using graphviz. This creates a diagram showing how inputs flow through segmented tensor products to produce outputs.

First, install graphviz:

.. code-block:: bash

    pip install graphviz

Then create a visualization:

.. jupyter-execute::

    from cuequivariance.segmented_polynomials import visualize_polynomial

    # Visualize the spherical harmonics polynomial
    sh_poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2]).polynomial
    graph = visualize_polynomial(sh_poly, input_names=["x"], output_names=["Y"])

    # Display the graph (in Jupyter it renders inline)
    graph

The diagram shows:

* **Input nodes** (blue): Display the input name, number of segments, and total size
* **STP nodes** (yellow): Show the subscripts and number of computation paths
* **Output nodes** (green): Display the output name, number of segments, and total size
* **Edges**: Represent the dataflow, with multiple edges drawn when an input is used multiple times

You can save the diagram to a file:

.. jupyter-execute::
    :hide-output:

    # Save as PNG (or 'svg', 'pdf', etc.)
    graph.render('spherical_harmonics', format='png', cleanup=True)

For more complex examples:

.. jupyter-execute::

    # Visualize a linear layer
    irreps_in = cue.Irreps("O3", "8x0e + 8x1o")
    irreps_out = cue.Irreps("O3", "4x0e + 4x1o")
    linear_poly = cue.descriptors.linear(irreps_in, irreps_out).polynomial

    graph = visualize_polynomial(linear_poly, input_names=["weights", "input"], output_names=["output"])
    graph

.. jupyter-execute::

    # Visualize a tensor product
    irreps = cue.Irreps("O3", "0e + 1o")
    tp_poly = cue.descriptors.channelwise_tensor_product(irreps, irreps, irreps).polynomial

    graph = visualize_polynomial(tp_poly, input_names=["weights", "x1", "x2"], output_names=["y"])
    graph

Visualizing Backward Pass
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also visualize the backward pass of a polynomial. The mapping function returned by :meth:`cue.SegmentedPolynomial.backward <cuequivariance.SegmentedPolynomial.backward>` accepts an optional `into_grad` parameter that can transform operand names, which is useful for labeling gradients:

.. jupyter-execute::

    # Create a polynomial and compute its backward pass
    irreps = cue.Irreps("O3", "0e + 1o")
    tp_poly = cue.descriptors.channelwise_tensor_product(irreps, irreps, irreps).polynomial
    
    # Compute backward pass (all inputs require gradients, output has cotangent)
    poly_bwd, m = tp_poly.backward([True, True, True], [True])
    
    # Transform operand names using the mapping function with into_grad
    # The mapping function takes (inputs, outputs) and returns (new_inputs, new_outputs)
    operand_names = (["weights", "x1", "x2"], ["y"])
    operand_names_bwd = m(operand_names, lambda n: f"d{n}")
    
    # Visualize the backward polynomial
    graph = visualize_polynomial(poly_bwd, input_names=operand_names_bwd[0], output_names=operand_names_bwd[1])
    graph
