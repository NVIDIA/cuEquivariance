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

Equivariant Polynomials
=======================

The submodule ``cue.descriptors`` contains many descriptors of equivariant polynomials represented by the class :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`.

Examples
--------

Linear layer
^^^^^^^^^^^^

.. jupyter-execute::

    import cuequivariance as cue

    irreps_in = cue.Irreps("O3", "32x0e + 32x1o")
    irreps_out = cue.Irreps("O3", "16x0e + 48x1o")
    cue.descriptors.linear(irreps_in, irreps_out)

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

Symmetric Contraction
^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    irreps = 128 * cue.Irreps("O3", "0e + 1o + 2e")
    e = cue.descriptors.symmetric_contraction(irreps, irreps, [0, 1, 2, 3])
    e


Execution on JAX
----------------

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    import cuequivariance as cue
    import cuequivariance_jax as cuex

    w = cuex.randn(jax.random.key(0), e.inputs[0])
    x = cuex.randn(jax.random.key(1), e.inputs[1])

    cuex.equivariant_polynomial(e, [w, x])

The function :func:`cuex.randn <cuequivariance_jax.randn>` generates random :class:`cuex.RepArray <cuequivariance_jax.RepArray>` objects.
The function :func:`cuex.equivariant_polynomial <cuequivariance_jax.equivariant_polynomial>` executes the tensor product.
The output is a :class:`cuex.RepArray <cuequivariance_jax.RepArray>` object.


Execution on PyTorch
--------------------

The same descriptor can be used in PyTorch using the class :class:`cuet.SegmentedPolynomial <cuequivariance_torch.SegmentedPolynomial>`.

.. jupyter-execute::

    import torch
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    if torch.cuda.is_available():
        module = cuet.SegmentedPolynomial(e.polynomial)

        w = torch.randn(1, e.inputs[0].dim).cuda()
        x = torch.randn(1, e.inputs[1].dim).cuda()

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
    