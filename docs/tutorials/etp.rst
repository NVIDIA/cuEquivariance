.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
   SPDX-License-Identifier: Apache-2.0

Equivariant Tensor Product
==========================

The submodule :class:`cuequivariance.descriptors` contains many descriptors of Equivariant Tensor Products (:class:`cuequivariance.EquivariantTensorProduct`).

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
        cue.Irreps("O3", "16x0e + 48x1o")
    )
    e.inputs, e.output

We can execute an :class:`cuequivariance.EquivariantTensorProduct` with PyTorch.

.. jupyter-execute::

    import torch
    import cuequivariance_torch as cuet

    module = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul)

    w = torch.randn(e.inputs[0].irreps.dim)
    x = torch.randn(e.inputs[1].irreps.dim)

    module(w, x)

Note that you have to specify the layout. If the layout specified is different from the one in the descriptor, the module will transpose the inputs/output to match the layout.
