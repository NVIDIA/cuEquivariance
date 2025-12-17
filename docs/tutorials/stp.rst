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

Segmented Tensor Product
========================

What Is a Segmented Tensor Product?
-----------------------------------

Imagine you want to perform a tensor contraction (like matrix multiplication or `np.einsum`), but your tensors are not just simple grids of numbers. Instead, they are composed of multiple distinct blocks, or **segments**, potentially of different sizes.

The :class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>` (STP) is a **blueprint** for such an operation. It describes exactly how these segments should be multiplied and summed together.

Crucially, **STP is agnostic to group theory**. It doesn't know about rotations, irreps, or symmetry. It is a pure mathematical descriptor for sparse, segmented linear algebra. The group theory (the "magic" of equivariance) is hidden inside the numerical values of the coefficients that we put into this blueprint.

Anatomy of a Descriptor
-----------------------

An STP descriptor is like a recipe. It doesn't hold the ingredients (the input data), but it tells you what to do with them. It consists of three main parts:

1.  **Subscripts**: A rule like ``"uv,ui,vj+ij"`` that tells us how indices contract (similar to `einsum`).
2.  **Operands**: Definitions of the input and output tensors structure (how many segments they have and their shapes).
3.  **Paths**: The specific connections between segments. A "path" says: "Take segment 0 from input A, segment 1 from input B, multiply them by this coefficient matrix, and add the result to segment 0 of output C."

Let's build one step by step.

Building a Descriptor
---------------------

First, we define the contraction rule using subscripts.

.. jupyter-execute::

    import numpy as np
    import cuequivariance as cue

    # Define the rule: A_uv * B_ui * C_vj -> D_ij
    # This looks complicated, but it's just a specific way to contract indices.
    d = cue.SegmentedTensorProduct.from_subscripts("uv,ui,vj+ij")
    print(d)

At this point, our blueprint is empty. We need to define the shape of our data.

Adding Segments
~~~~~~~~~~~~~~~

We define the structure of our operands by adding **segments**.
Think of an operand as a list of tensors.
For example, if Operand 1 has two segments—one of shape (2, 5) and one of shape (2, 4)—we add them as follows.

.. jupyter-execute::

    # Operand 0: One segment of shape (2, 3)
    d.add_segment(0, (2, 3))

    # Operand 1: Two segments of shape (2, 5) and (2, 4)
    d.add_segments(1, [(2, 5), (2, 4)])

    # Operand 2: One segment of shape (3, 6)
    d.add_segment(2, (3, 6))

    print(d)

Adding Paths
~~~~~~~~~~~~

Now for the connectivity. A **path** defines a single term in our sparse calculation.
It specifies which segments interact.

If we want to connect:
*   Segment 0 from Operand 0
*   Segment 1 from Operand 1
*   Segment 0 from Operand 2

We add a path. We also need to provide a **coefficient** array that weights this interaction. The shape of this coefficient is determined by the subscripts we defined earlier (indices `i` and `j` are not contracted, so they form the coefficient dimensions).

.. jupyter-execute::

    # Connect: Op0[0], Op1[1], Op2[0]
    # The coefficient shape matches the free indices (i=4, j=6)
    coeff = np.ones((4, 6))
    d.add_path(0, 1, 0, c=coeff)

    print(d.to_text())

The output shows ``num_paths=1``. We have successfully described one specific multiplication operation between these blocks.

Normalization
-------------

In deep learning, keeping the scale of signals under control is vital. If we multiply many random numbers, values can explode or vanish.
STP helps us by providing tools to normalize these paths automatically.

:meth:`cue.SegmentedTensorProduct.normalize_paths_for_operand` adjusts the coefficients so that the output variance remains stable (usually close to 1), assuming the inputs are standard normal variables.

.. jupyter-execute::

    # Normalize assuming Operand 1 is the input signal
    d = d.normalize_paths_for_operand(1)
    
    # The coefficients are now scaled down
    print(d.paths[0].coefficients[0, 0])

Optimization
------------

Before executing this blueprint, we can optimize it. Just like compiling code, we can simplify the descriptor to make it run faster.

*   :meth:`cue.SegmentedTensorProduct.consolidate_paths`: Merges duplicate paths.
*   :meth:`cue.SegmentedTensorProduct.flatten_modes`: Merges dimensions (e.g., treating a 3x3 matrix as a size-9 vector) to simplify the underlying loops.

.. jupyter-execute::

    # Example: Consolidate paths (merge duplicates and remove zeros)
    d_consolidated = d.consolidate_paths()
    print(d_consolidated)
    
    # Example: Flatten the 'u' mode (must be at the beginning of subscripts)
    d_flat = d.flatten_modes("u")
    print(d_flat)

Summary
-------

*   **STP is a Blueprint**: It describes the math of the contraction.
*   **Segments & Paths**: It handles data that comes in chunks, connected sparsely.
*   **General Purpose**: It works for any sparse tensor algebra, not just equivariant ones.

Now that we have the math descriptor, how do we actually run it on data? That is the job of the :doc:`Segmented Polynomial <poly>`.
