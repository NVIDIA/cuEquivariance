.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

In this example, we will show how to create a custom tensor product descriptor and execute it.
First we need to import the necessary modules.

.. jupyter-execute::

   import itertools
   import torch
   import jax
   import jax.numpy as jnp

   import cuequivariance as cue
   import cuequivariance.segmented_tensor_product as stp
   import cuequivariance_torch as cuet # to execute the tensor product with PyTorch
   import cuequivariance_jax as cuex   # to execute the tensor product with JAX

.. currentmodule:: cuequivariance

Now, we will create a custom tensor product descriptor that represents the tensor product of the two representations. See :ref:`tuto_irreps` for more information on irreps.

.. jupyter-execute::

   irreps1 = cue.Irreps("O3", "32x0e + 32x1o")
   irreps2 = cue.Irreps("O3", "16x0e + 48x1o")

The tensor product descriptor is created step by step. First, we create an empty descriptor given its subscripts.
In the case of the linear layer, we have 3 operands: the weight, the input, and the output.
The subscripts of this tensor product are "uv,iu,iv" where "uv" represents the modes of the weight, "iu" represents the modes of the input, and "iv" represents the modes of the output.

.. jupyter-execute::

   d = stp.SegmentedTensorProduct.from_subscripts("uv,iu,iv")
   d

Each operand of the tensor product descriptor has a list of segments.
We can add segments to the descriptor using the `add_segment` method.
We can add the segments of the input and output representations to the descriptor.

.. jupyter-execute::

   for mul, ir in irreps1:
      d.add_segment(1, (ir.dim, mul))
   for mul, ir in irreps2:
      d.add_segment(2, (ir.dim, mul))

   d

Now we can enumerate all the possible pairs of irreps and add weight segements and paths between them when the irreps are the same.

.. jupyter-execute::

   for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
      enumerate(irreps1), enumerate(irreps2)
   ):
      if ir1 == ir2:
         d.add_path(None, i1, i2, c=1.0)

   d

We can see the two paths we added:

.. jupyter-execute::

      d.paths


Finally, we can normalize the paths for the last operand such that the output is normalized to variance 1.

.. jupyter-execute::

   d = d.normalize_paths_for_operand(-1)
   d.paths

As we can see, the paths coefficients has been normalized.

Now we can create a tensor product from the descriptor and execute it. In PyTorch, we can use the :class:`cuet.TensorProduct` class.

.. jupyter-execute::

   linear_torch = cuet.TensorProduct(d)
   linear_torch


In JAX, we can use the :func:`cuex.tensor_product` function.

.. jupyter-execute::

   linear_jax = cuex.tensor_product(d)
   linear_jax

Now we can execute the linear layer with random input and weight tensors.

.. jupyter-execute::

   w = torch.randn(d.operands[0].size)
   x1 = torch.randn(3000, irreps1.dim)

   x2 = linear_torch([w, x1])

   assert x2.shape == (3000, irreps2.dim)

Now we can verify that the output is well normalized.

.. jupyter-execute::

   x2.var()

And finally the JAX version.

.. jupyter-execute::

   w = jax.random.normal(jax.random.key(0), (d.operands[0].size,))
   x1 = jax.random.normal(jax.random.key(1), (3000, irreps1.dim))

   x2 = linear_jax(w, x1)

   assert x2.shape == (3000, irreps2.dim)
   x2.var()
