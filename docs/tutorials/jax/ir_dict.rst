.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

The ``ir_dict`` Interface
=========================

The :mod:`cuequivariance_jax.ir_dict` module provides an alternative to :class:`~cuequivariance_jax.RepArray` for working with equivariant data. Instead of a single contiguous array, features are stored as ``dict[Irrep, Array]`` where each value has shape ``(..., multiplicity, irrep_dim)``.

This representation works naturally with ``jax.tree`` operations and is used by the :mod:`cuequivariance_jax.nnx` layers.

From Descriptor to ``ir_dict``
------------------------------

Descriptors produce :class:`~cuequivariance.EquivariantPolynomial` objects with dense operands (e.g., ``32x0+32x1``). A dense operand requires all irreps to be packed into a single contiguous buffer. By splitting each operand by irrep with :meth:`~cuequivariance.EquivariantPolynomial.split_operand_by_irrep`, each irrep gets its own separate buffer. This relaxes the memory layout constraint: the buffers for different irreps no longer need to be contiguous with each other.

This is especially useful when the polynomial is preceded or followed by linear layers that act independently on each irrep (like :class:`~cuequivariance_jax.nnx.IrrepsLinear`). With split operands, there is no need for any transpose or copy between the linear layers and the polynomial — the ``dict[Irrep, Array]`` flows directly through the pipeline.

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    from einops import rearrange
    import cuequivariance as cue
    import cuequivariance_jax as cuex

    # Build a channelwise tensor product
    e = cue.descriptors.channelwise_tensor_product(
        32 * cue.Irreps("SO3", "0 + 1"),
        cue.Irreps("SO3", "0 + 1"),
        cue.Irreps("SO3", "0 + 1"),
        simplify_irreps3=True,
    )
    print("Before split:")
    print(e)

.. jupyter-execute::

    # Split operands by irrep
    # Order: split inner operands first to preserve indices
    e_split = (
        e.split_operand_by_irrep(2)
         .split_operand_by_irrep(1)
         .split_operand_by_irrep(-1)
    )
    poly = e_split.polynomial

    print("After split:")
    print(e_split)
    print()
    for i, op in enumerate(poly.inputs):
        print(f"  Input {i}: num_segments={op.num_segments}, uniform={op.all_same_segment_shape()}")
    for i, op in enumerate(poly.outputs):
        print(f"  Output {i}: num_segments={op.num_segments}, uniform={op.all_same_segment_shape()}")


Executing with ``segmented_polynomial_uniform_1d``
--------------------------------------------------

The :func:`~cuequivariance_jax.ir_dict.segmented_polynomial_uniform_1d` function handles the flattening/unflattening between the ``dict[Irrep, Array]`` pytree structure and the flat arrays that the kernel expects.

Each input array has shape ``(..., num_segments, *segment_shape)``. For the weight operand, we reshape the flat weights into this form. For ``dict[Irrep, Array]`` operands, each value is one leaf of the pytree.

.. jupyter-execute::

    batch = 16

    # Weights: reshape flat -> (batch, num_segments, segment_size)
    w_flat = jax.random.normal(jax.random.key(0), (batch, poly.inputs[0].size))
    w = rearrange(w_flat, "b (s m) -> b s m", s=poly.inputs[0].num_segments)
    print(f"Weights: {w.shape}  (batch, num_segments, segment_size)")

    # Inputs as dict[Irrep, Array]
    # Shape convention: (batch, ir.dim, mul) for ir_mul layout
    node_feats = {
        cue.SO3(0): jax.random.normal(jax.random.key(1), (batch, 32, 1)),
        cue.SO3(1): jax.random.normal(jax.random.key(2), (batch, 32, 3)),
    }
    # Rearrange from (batch, mul, ir.dim) to (batch, ir.dim, mul) for ir_mul layout
    x = jax.tree.map(lambda v: rearrange(v, "b m i -> b i m"), node_feats)
    print(f"Input l=0: {x[cue.SO3(0)].shape}  (batch, ir.dim, mul)")
    print(f"Input l=1: {x[cue.SO3(1)].shape}  (batch, ir.dim, mul)")

    # Second input (e.g. spherical harmonics): (batch, ir.dim)
    sph = {
        cue.SO3(0): jax.random.normal(jax.random.key(3), (batch, 1)),
        cue.SO3(1): jax.random.normal(jax.random.key(4), (batch, 3)),
    }

    # Build output template: one entry per split output
    irreps_out = e.outputs[0].irreps
    out_template = {
        ir: jax.ShapeDtypeStruct(
            (batch, desc.num_segments) + desc.segment_shape, w.dtype
        )
        for (_, ir), desc in zip(irreps_out, poly.outputs)
    }
    print(f"Output template: { {str(k): v.shape for k, v in out_template.items()} }")

.. jupyter-execute::

    # Execute
    y = cuex.ir_dict.segmented_polynomial_uniform_1d(
        poly, [w, x, sph], out_template,
    )

    for ir, v in y.items():
        print(f"  Output {ir}: {v.shape}")


Indexing (Gather/Scatter)
-------------------------

In graph neural networks, features live on nodes and edges with different batch sizes. Index arrays handle the gather (for inputs) and scatter (for outputs):

.. jupyter-execute::

    num_edges, num_nodes = 100, 30

    w = jax.random.normal(jax.random.key(0), (num_edges, poly.inputs[0].size))
    w = rearrange(w, "e (s m) -> e s m", s=poly.inputs[0].num_segments)

    node_feats = {
        cue.SO3(0): jax.random.normal(jax.random.key(1), (num_nodes, 1, 32)),
        cue.SO3(1): jax.random.normal(jax.random.key(2), (num_nodes, 3, 32)),
    }

    sph = {
        cue.SO3(0): jax.random.normal(jax.random.key(3), (num_edges, 1)),
        cue.SO3(1): jax.random.normal(jax.random.key(4), (num_edges, 3)),
    }

    senders = jax.random.randint(jax.random.key(5), (num_edges,), 0, num_nodes)
    receivers = jax.random.randint(jax.random.key(6), (num_edges,), 0, num_nodes)

    out_template = {
        ir: jax.ShapeDtypeStruct(
            (num_nodes, desc.num_segments) + desc.segment_shape, w.dtype
        )
        for (_, ir), desc in zip(irreps_out, poly.outputs)
    }

    # Gather node features at senders, scatter results to receivers
    y = cuex.ir_dict.segmented_polynomial_uniform_1d(
        poly,
        [w, node_feats, sph],
        out_template,
        input_indices=[None, senders, None],
        output_indices=receivers,
    )

    for ir, v in y.items():
        print(f"  Output {ir}: {v.shape}")


Utility Functions
-----------------

The ``ir_dict`` module provides helpers for converting between flat arrays and ``dict[Irrep, Array]``:

.. jupyter-execute::

    irreps = cue.Irreps(cue.SO3, "4x0 + 2x1")

    # Flat array -> dict
    flat = jnp.ones((8, irreps.dim))
    d = cuex.ir_dict.flat_to_dict(irreps, flat)
    print(f"flat_to_dict: l=0 {d[cue.SO3(0)].shape}, l=1 {d[cue.SO3(1)].shape}")

    # Dict -> flat array
    flat_back = cuex.ir_dict.dict_to_flat(irreps, d)
    print(f"dict_to_flat: {flat_back.shape}")

    # Arithmetic
    z = cuex.ir_dict.irreps_add(d, d)
    print(f"irreps_add: l=0 sum={float(z[cue.SO3(0)].sum())}")

    # Validation
    cuex.ir_dict.assert_mul_ir_dict(irreps, d)
    print("assert_mul_ir_dict: passed")
