.. SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

How to accelerate MACE
======================



The layout
----------
*cuEquivariance* offers the possibility to use a more efficient layout for the irreps.
The old layout, compatible with ``e3nn`` operations, is called ``cue.mul_ir``, while the new layout is called ``cue.ir_mul``.


Irreps
------
*(check the irreps guide for more information about this)*

If we stick to the old layout, we can equivalently define ``e3nn`` and ``cue`` Irreps as:

.. jupyter-execute::

    from e3nn import o3
    import cuequivariance as cue

    old_irreps = o3.Irreps('1x0e+1x1o')
    new_irreps = cue.Irreps(cue.O3, '1x0e+1x1o')

*A note about the O3 group:*
The official MACE implementation uses ``e3nn`` version ``0.4.4``, which employs a slightly different group definition with respect to more recent ``e3nn`` versions (and *cuEquivariance*).
If compatibility with old models is desired, it is possible to enforce the use of this group by defining the new group:

.. jupyter-execute::

    from typing import Iterator
    import numpy as np
    import itertools

    class O3_e3nn(cue.O3):
        def __mul__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> Iterator["O3_e3nn"]:
            return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

        @classmethod
        def clebsch_gordan(
            cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(rep3.dim)
            else:
                return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> bool:
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator["O3_e3nn"]:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)

``O3_e3nn`` should then be used throughout the code in place of ``cue.O3``, like in the following example:

.. jupyter-execute::

    from cuequivariance.experimental.mace import O3_e3nn  # also available here
    new_irreps = cue.Irreps(O3_e3nn, '1x0e+1x1o')


Here is some snippets useful for accelerating MACE:
    - `Using PyTorch <#pytorch>`_
    - `Using JAX <#jax>`_

.. _pytorch:

Using PyTorch
^^^^^^^^^^^^^

  **Note:** in the following we will refer to the ``cuequivariance`` library as ``cue`` and to the   ``cuequivariance_torch`` library as ``cuet``.

To accelerate MACE, we want to substitute the *e3nn* operations with the equivalent *cuEquivariance* operations.

In particular, there are 4 operations within MACE:

- ``SymmetricContraction`` → ``cuet.SymmetricContraction``
- ``tp_out_irreps_with_instructions`` + ``e3nn.o3.TensorProduct`` → ``cuet.ChannelWiseTensorProduct``
- ``e3nn.o3.Linear`` → ``cuet.Linear``
- ``e3nn.o3.FullyConnectedTensorProduct`` → ``cuet.FullyConnectedTensorProduct``

All of these have a cuequivariance counterpart, but for now only the first two result in a significant performance improvement.
The layout can be changed throughout the code, but this requires all operations to be upgraded to their ``cuet`` counterparts.
For the operations where the kernel is not yet optimized, we can fall back to the FX implementation (implementation using ``torch.fx`` like in *e3nn*) with a simple flag.

Common options
--------------
These general rules are valid for most operations:

``layout``
  ``cue.mul_ir`` or ``cue.ir_mul``, as explained above
``dtype``
  ``torch.float32`` or ``torch.float64``
``use_fallback``
  ``bool``, use this when calling the function to select the FX implementation instead of the kernel

We can thus set some of this common options:

.. jupyter-execute::

    import torch
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dtype = torch.float32  # or torch.float64

.. _mace_tutorial_pytorch_symmetric_contraction:

SymmetricContraction
--------------------

The original SymmetricContraction was an operation written specifically for MACE.
It performs operations on a single input_feature repeated multiple times, but uses a second input (or attribute, 1-hot encoded) to select weights depending on the atomic species.

For performance reasons, the cuequivariance implementation uses direct indexing in place of 1-hot vectors, i.e. the attributes are now integers, indicating the index of each atom in the species list.

The SymmetricContraction code should look like this:

.. jupyter-execute::

    feats_irreps = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")
    target_irreps = cue.Irreps("O3", "32x0e + 32x1o")

    # OLD FUNCTION DEFINITION:
    # symmetric_contractions = SymmetricContraction(
    #     irreps_in=feats_irreps,
    #     irreps_out=target_irreps,
    #     correlation=3,
    #     num_elements=10,
    # )

    # NEW FUNCTION DEFINITION:
    symmetric_contractions = cuet.SymmetricContraction(
        irreps_in=feats_irreps,
        irreps_out=target_irreps,
        contraction_degree=3,
        num_elements=10,
        layout_in=cue.ir_mul,
        layout_out=cue.mul_ir,
        original_mace=True,
        dtype=dtype,
        device=device,
    )

    node_feats = torch.randn(128, 32, feats_irreps.dim // 32, dtype=dtype, device=device)

    # with node_attrs_index being the index version of node_attrs, sth like:
    # node_attrs_index = torch.nonzero(node_attrs)[:, 1].int()
    node_attrs_index = torch.randint(0, 10, (128,), dtype=torch.int32, device=device)

    # OLD CALL:
    # symmetric_contractions(node_feats, node_attrs)

    # NEW CALL:
    node_feats = torch.transpose(node_feats, 1, 2).flatten(1)
    symmetric_contractions(node_feats, node_attrs_index)

We can see that in this case we can specify separately the ``layout_in`` and ``layout_out``.
In this particular case, we have selected to use ``cue.ir_mul`` as an input, but have explicitly performed the transposition before calling the operation. If you were using this layout throughout, this would not be needed.

The flag ``original_mace`` ensures compatibility with the old SymmetricContraction, where operations had a slightly different order than the new version.

.. _mace_tutorial_pytorch_channel_wise:

ChannelWiseTensorProduct
------------------------

The ``ChannelWiseTensorProduct`` replaces the custom operation that was obtained in MACE by defining custom instructions and calling a ``TensorProduct``.
This particular operation was also called with external weights computed through a MLP. The same can be done in ``cuet``.

The new version for this part of the code will thus read:

.. jupyter-execute::

    feats_irreps = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")
    edge_attrs_irreps = target_irreps = "0e + 1o + 2e + 3o"
    edge_feats = torch.randn(128, feats_irreps.dim, dtype=dtype, device=device)
    edge_vectors = torch.randn(128, 3, dtype=dtype, device=device)

    edge_sh = cuet.spherical_harmonics([0, 1, 2, 3], edge_vectors)

    # OLD FUNCTION DEFINITION
    # irreps_mid, instructions = tp_out_irreps_with_instructions(
    #     feats_irreps,
    #     edge_attrs_irreps,
    #     target_irreps,
    # )
    # conv_tp = o3.TensorProduct(
    #     feats_irreps,
    #     edge_attrs_irreps,
    #     irreps_mid,
    #     instructions=instructions,
    #     shared_weights=False,
    #     internal_weights=False,
    # )

    # NEW FUNCTION DEFINITION (single function)
    conv_tp = cuet.ChannelWiseTensorProduct(
        feats_irreps,
        cue.Irreps("O3", edge_attrs_irreps),
        cue.Irreps("O3", target_irreps),
        shared_weights=False,
        internal_weights=False,
        layout=cue.mul_ir,
        math_dtype=dtype,
        device=device,
    )

    # Weights (would normally come from conv_tp_weights)
    tp_weights = torch.randn(128, conv_tp.weight_numel, dtype=dtype, device=device)

    # OLD CALL:
    # mji = conv_tp(edge_feats, edge_sh, tp_weights)

    # NEW CALL: (unchanged)
    conv_tp(edge_feats, edge_sh, tp_weights)


.. _mace_tutorial_pytorch_linear:

Linear
------

This is one of the simplest operations, and it is essentially unchanged.
Depending on the irreps size, the kernel might not improve above the naive implementation, we thus show an example where the fallback is employed.

.. jupyter-execute::

    feats_irreps = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")

    # OLD FUNCTION DEFINITION:
    # linear = o3.Linear(
    #     feats_irreps,
    #     feats_irreps,
    #     internal_weights=True,
    #     shared_weights=True,
    # )

    # NEW FUNCTION DEFINITION:
    linear = cuet.Linear(
        feats_irreps,
        feats_irreps,
        layout=cue.mul_ir,
        internal_weights=True,
        shared_weights=True,
        dtype=dtype,
        device=device,
    )

    node_feats = torch.randn(128, feats_irreps.dim, dtype=dtype, device=device)

    # OLD CALL:
    # linear(node_feats)

    # NEW CALL: (unchanged, using fallback)
    linear(node_feats, use_fallback=True)


.. _mace_tutorial_pytorch_fully_connected_tp:

FullyConnectedTensorProduct
---------------------------

The ``FullyConnectedTensorProduct`` is used in MACE for the ``skip-tp`` operation.
In this case, the "node attributes" used to select the weights are still accepted as 1-hot.
This operation is also essentially unchanged, and we show a version using the fallback.

.. jupyter-execute::

    feats_irreps = cue.Irreps("O3", "32x0e + 32x1o + 32x2e")
    attrs_irreps = cue.Irreps("O3", "10x0e")

    # OLD FUNCTION DEFINITION:
    # skip_tp = o3.FullyConnectedTensorProduct(
    #     feats_irreps,
    #     attrs_irreps,
    #     feats_irreps,
    #     internal_weights=True,
    #     shared_weights=True,
    # )

    # NEW FUNCTION DEFINITION:
    skip_tp = cuet.FullyConnectedTensorProduct(
        feats_irreps,
        attrs_irreps,
        feats_irreps,
        layout=cue.mul_ir,
        internal_weights=True,
        shared_weights=True,
        dtype=dtype,
        device=device,
    )

    node_feats = torch.randn(128, feats_irreps.dim, dtype=dtype, device=device)
    node_attrs = torch.nn.functional.one_hot(torch.randint(0, 10, (128,), dtype=torch.int64, device=device), 10).to(dtype)

    # OLD CALL:
    # skip_tp(node_feats, node_attrs)

    # NEW CALL: (unchanged, using fallback)
    skip_tp(node_feats, node_attrs, use_fallback=True)




.. _jax:

Using JAX
^^^^^^^^^

  **Note:** In the following, we will refer to the ``cuequivariance`` library as ``cue`` and the ``cuequivariance_jax`` library as ``cuex``.

The following code snippets demonstrate the main components of a MACE layer implemented in JAX.
For the sake of simplicity, we will not implement the entire MACE layer, but rather focus on the main components.
First, we import the necessary libraries.

.. jupyter-execute::

    import cuequivariance as cue
    import cuequivariance_jax as cuex
    import jax
    import jax.numpy as jnp
    from cuequivariance import equivariant_tensor_product as etp
    from cuequivariance.experimental.mace import symmetric_contraction
    from cuequivariance_jax.experimental.utils import MultiLayerPerceptron, gather

The input data consists of node features, edge vectors, radial embeddings, and sender and receiver indices.

.. jupyter-execute::

    num_species = 3
    num_nodes = 12
    num_edges = 26
    vectors = cuex.randn(
        jax.random.key(0), cue.Irreps("O3", "1o"), (num_edges,), cue.ir_mul
    )
    node_feats = cuex.randn(
        jax.random.key(0), cue.Irreps("O3", "16x0e + 16x1o"), (num_nodes,), cue.ir_mul
    )
    node_species = jax.random.randint(jax.random.key(0), (num_nodes,), 0, num_species)
    radial_embeddings = jax.random.normal(jax.random.key(0), (num_edges, 4))
    senders, receivers = jax.random.randint(jax.random.key(0), (2, num_edges), 0, num_nodes)

    def param(name: str, init_fn, shape, dtype):
        # dummy function to obtain parameters (when using flax, one should use self.param instead)
        print(f"param({name!r}, {init_fn!r}, {shape!r}, {dtype!r})")
        return init_fn(jax.random.key(0), shape, dtype)

Next, we define the layer's hyperparameters.

.. jupyter-execute::

    num_features = 32
    interaction_irreps = cue.Irreps("O3", "0e + 1o + 2e + 3o")
    hidden_out = cue.Irreps("O3", "0e + 1o")
    max_ell = 3
    dtype = node_feats.dtype

The MACE layer is composed of two types of linear layers: those that depend on the species and those that do not.

.. jupyter-execute::

    def lin(irreps: cue.Irreps, input: cuex.IrrepsArray, name: str):
        e = descriptors.linear(input.irreps(), irreps)
        w = param(name, jax.random.normal, (e.inputs[0].irreps.dim,), dtype)
        return cuex.equivariant_tensor_product(e, w, input, precision="HIGH")


    def linZ(irreps: cue.Irreps, input: cuex.IrrepsArray, name: str):
        e = descriptors.linear(input.irreps(), irreps)
        w = param(
            name,
            jax.random.normal,
            (num_species, e.inputs[0].irreps.dim),
            dtype,
        )
        return cuex.equivariant_tensor_product(
            e, w[node_species], input, precision="HIGH"
        ) / jnp.sqrt(num_species)

The first part involves operations before the convolutional part.

.. jupyter-execute::

    self_connection = linZ(num_features * hidden_out, node_feats, "linZ_skip_tp")
    node_feats = lin(node_feats.irreps(), node_feats, "linear_up")

Next, we implement the convolutional part.

.. jupyter-execute::

    messages = node_feats[senders]
    sph = cuex.spherical_harmonics(range(max_ell + 1), vectors)
    e = descriptors.channelwise_tensor_product(messages.irreps(), sph.irreps(), interaction_irreps)
    e = e.squeeze_modes().flatten_coefficient_modes()

    mlp = MultiLayerPerceptron(
        [64, 64, 64, e.inputs[0].irreps.dim],
        jax.nn.silu,
        output_activation=False,
        with_bias=False,
    )
    w = mlp.init(jax.random.key(0), radial_embeddings)  # dummy parameters
    mix = mlp.apply(w, radial_embeddings)

    messages = cuex.equivariant_tensor_product(e, mix, messages, sph)

    avg_num_neighbors = num_edges / num_nodes  # you should use a constant here
    node_feats = gather(receivers, messages, node_feats.shape[0]) / avg_num_neighbors

Now, we perform the symmetric contraction part.

.. jupyter-execute::

    node_feats = lin(num_features * interaction_irreps, node_feats, "linear_down")
    e, projection = symmetric_contraction(
        node_feats.irreps(),
        num_features * hidden_out,
        [1, 2, 3],
    )
    n = projection.shape[0]
    w = param(
        "symmetric_contraction", jax.random.normal, (num_species, n, num_features), dtype
    )
    w = jnp.einsum("zau,ab->zbu", w, projection)
    w = jnp.reshape(w, (num_species, -1))
    node_feats = cuex.equivariant_tensor_product(e, w[node_species], node_feats)


Finally, we apply the remaining linear layers.

.. jupyter-execute::

    node_feats = lin(num_features * hidden_out, node_feats, "linear_post_sc")
    node_feats = node_feats + self_connection  # [n_nodes, feature * hidden_irreps]

    node_outputs = lin(cue.Irreps("O3", "0e"), node_feats, "linear_readout")

    print(node_outputs)