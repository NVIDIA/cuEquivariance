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

Equivariant Tensor Product
==========================

The submodule :class:`cuequivariance.descriptors` contains many descriptors of Equivariant Tensor Products (:class:`cuequivariance.EquivariantTensorProduct`).

Examples
--------

Linear layer
^^^^^^^^^^^^

.. jupyter-execute::

    import cuequivariance as cue

    cue.descriptors.linear(cue.Irreps("O3", "32x0e + 32x1o"), cue.Irreps("O3", "16x0e + 48x1o"))

In this example, the first operand is the weights, they are always scalars.
There is ``32 * 16 = 512`` weights to connect the ``0e`` together and ``32 * 48 = 1536`` weights to connect the ``1o`` together. This gives a total of ``2048`` weights.

Spherical Harmonics
^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2, 3])

The spherical harmonics are polynomials of an input vector.
This descriptor specifies the polynomials of degree 0, 1, 2 and 3.

Rotation
^^^^^^^^

.. jupyter-execute::

    cue.descriptors.yxy_rotation(cue.Irreps("O3", "32x0e + 32x1o"))

This case is a bit of an edge case, it is a rotation of the input by angles encoded as :math:`sin(\theta)` and :math:`cos(\theta)`. See the function :func:`cuet.encode_rotation_angle <cuequivariance_torch.encode_rotation_angle>` for more details.

Execution on JAX
----------------

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    import cuequivariance_jax as cuex

    e = cue.descriptors.linear(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "8x0e + 4x1o")
    )
    w = cuex.randn(jax.random.key(0), e.inputs[0])
    x = cuex.randn(jax.random.key(1), e.inputs[1])

    cuex.equivariant_tensor_product(e, w, x)

The function :func:`cuex.randn <cuequivariance_jax.randn>` generates random :class:`cuex.RepArray <cuequivariance_jax.RepArray>` objects.
The function :func:`cuex.equivariant_tensor_product <cuequivariance_jax.equivariant_tensor_product>` executes the tensor product.
The output is a :class:`cuex.RepArray <cuequivariance_jax.RepArray>` object.


Execution on PyTorch
--------------------

We can execute an :class:`cuequivariance.EquivariantTensorProduct` with PyTorch.

.. jupyter-execute::

    import torch
    import cuequivariance_torch as cuet

    e = cue.descriptors.linear(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "8x0e + 4x1o")
    )
    module = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul, use_fallback=True)

    w = torch.randn(1, e.inputs[0].dim)
    x = torch.randn(1, e.inputs[1].dim)

    module([w, x])

Note that you have to specify the layout. If the layout specified is different from the one in the descriptor, the module will transpose the inputs/output to match the layout.
