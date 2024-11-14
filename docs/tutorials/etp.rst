.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

Equivariant Tensor Product
==========================

The submodule :class:`cuequivariance.descriptors` contains many descriptors of Equivariant Tensor Products (:class:`cuequivariance.EquivariantTensorProduct`).
Here are some examples:

.. jupyter-execute::

    import cuequivariance as cue

    cue.descriptors.linear(cue.Irreps("O3", "32x0e + 32x1o"), cue.Irreps("O3", "16x0e + 48x1o"))

.. jupyter-execute::

    cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2, 3])

.. jupyter-execute::

    cue.descriptors.yxy_rotation(cue.Irreps("O3", "32x0e + 32x1o"))

The object returned contains a description of the inputs and output of the tensor product.

.. jupyter-execute::

    e = cue.descriptors.linear(
        cue.Irreps("O3", "32x0e + 32x1o"),
        cue.Irreps("O3", "8x0e + 4x1o")
    )
    e.inputs, e.output

Execution on JAX
----------------

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    import cuequivariance_jax as cuex

    w = cuex.randn(jax.random.key(0), e.inputs[0])
    x = cuex.randn(jax.random.key(1), e.inputs[1])

    cuex.equivariant_tensor_product(e, w, x)

The function :func:`cuex.randn <cuequivariance_jax.randn>` generates random :class:`cuex.IrrepsArray <cuequivariance_jax.IrrepsArray>` objects.
The function :func:`cuex.equivariant_tensor_product <cuequivariance_jax.equivariant_tensor_product>` executes the tensor product.
The output is a :class:`cuex.IrrepsArray <cuequivariance_jax.IrrepsArray>` object.


Execution on PyTorch
--------------------

We can execute an :class:`cuequivariance.EquivariantTensorProduct` with PyTorch.

.. jupyter-execute::

    import torch
    import cuequivariance_torch as cuet

    module = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul)

    w = torch.randn(e.inputs[0].irreps.dim)
    x = torch.randn(e.inputs[1].irreps.dim)

    module(w, x)

Note that you have to specify the layout. If the layout specified is different from the one in the descriptor, the module will transpose the inputs/output to match the layout.
