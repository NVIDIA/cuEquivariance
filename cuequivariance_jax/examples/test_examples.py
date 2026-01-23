# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import cuequivariance as cue

from .mace_linen import MACEModel
from .mace_nnx import MACEModel as MACEModelNNX
from .nequip import NEQUIPModel


def test_mace_model_basic():
    """Test basic MACE model functionality."""
    # Small test case
    num_atoms = 10
    num_edges = 20
    num_species = 5
    num_graphs = 2
    dtype = jnp.float32

    # Create small model
    model = MACEModel(
        num_layers=1,
        num_features=32,
        num_species=num_species,
        max_ell=2,
        correlation=2,
        num_radial_basis=4,
        interaction_irreps=cue.Irreps(cue.O3, "0e+1o"),
        hidden_irreps=cue.Irreps(cue.O3, "0e"),
        offsets=np.zeros(num_species),
        cutoff=3.0,
        epsilon=0.1,
        skip_connection_first_layer=True,
    )

    # Create dummy data
    vecs = jax.random.normal(jax.random.key(0), (num_edges, 3), dtype)
    species = jax.random.randint(
        jax.random.key(1), (num_atoms,), 0, num_species, jnp.int32
    )
    senders, receivers = jax.random.randint(
        jax.random.key(2), (2, num_edges), 0, num_atoms, jnp.int32
    )
    graph_index = jax.random.randint(
        jax.random.key(3), (num_atoms,), 0, num_graphs, jnp.int32
    )
    graph_index = jnp.sort(graph_index)
    nats = jnp.zeros((num_graphs,), dtype=jnp.int32).at[graph_index].add(1)
    mask = jnp.ones((num_edges,), dtype=jnp.int32)

    batch_dict = dict(
        nn_vecs=vecs,
        species=species,
        inda=senders,
        indb=receivers,
        inde=graph_index,
        nats=nats,
        mask=mask,
    )

    # Initialize and run forward pass
    params = model.init(jax.random.key(0), batch_dict)
    E, F = model.apply(params, batch_dict)

    # Check output shapes
    assert E.shape == (num_graphs,), f"Energy shape {E.shape} != {(num_graphs,)}"
    assert F.shape == (num_atoms, 3), f"Forces shape {F.shape} != {(num_atoms, 3)}"
    assert E.dtype == dtype, f"Energy dtype {E.dtype} != {dtype}"
    assert F.dtype == dtype, f"Forces dtype {F.dtype} != {dtype}"


def test_mace_nnx_model_basic():
    """Test basic MACE NNX model functionality."""
    # Small test case
    num_atoms = 10
    num_edges = 20
    num_species = 5
    num_graphs = 2
    dtype = jnp.float32

    # Create small model
    rngs = nnx.Rngs(0)
    model = MACEModelNNX(
        num_layers=1,
        num_features=32,
        num_species=num_species,
        max_ell=2,
        correlation=2,
        num_radial_basis=4,
        interaction_irreps=cue.Irreps(cue.O3, "0e+1o"),
        hidden_irreps=cue.Irreps(cue.O3, "0e"),
        offsets=np.zeros(num_species),
        cutoff=3.0,
        epsilon=0.1,
        skip_connection_first_layer=True,
        dtype=dtype,
        rngs=rngs,
    )

    # Create dummy data
    vecs = jax.random.normal(jax.random.key(0), (num_edges, 3), dtype)
    species = jax.random.randint(
        jax.random.key(1), (num_atoms,), 0, num_species, jnp.int32
    )
    senders, receivers = jax.random.randint(
        jax.random.key(2), (2, num_edges), 0, num_atoms, jnp.int32
    )
    graph_index = jax.random.randint(
        jax.random.key(3), (num_atoms,), 0, num_graphs, jnp.int32
    )
    graph_index = jnp.sort(graph_index)
    nats = jnp.zeros((num_graphs,), dtype=jnp.int32).at[graph_index].add(1)
    mask = jnp.ones((num_edges,), dtype=jnp.int32)

    batch_dict = dict(
        nn_vecs=vecs,
        species=species,
        inda=senders,
        indb=receivers,
        inde=graph_index,
        nats=nats,
        mask=mask,
    )

    # Run forward pass
    E, F = model(batch_dict)

    # Check output shapes
    assert E.shape == (num_graphs,), f"Energy shape {E.shape} != {(num_graphs,)}"
    assert F.shape == (num_atoms, 3), f"Forces shape {F.shape} != {(num_atoms, 3)}"
    assert E.dtype == dtype, f"Energy dtype {E.dtype} != {dtype}"
    assert F.dtype == dtype, f"Forces dtype {F.dtype} != {dtype}"

    # Check outputs are finite
    assert jnp.all(jnp.isfinite(E)), "Energy contains non-finite values"
    assert jnp.all(jnp.isfinite(F)), "Forces contain non-finite values"


def _convert_irreps_linear(w, irreps_in, irreps_out):
    """Convert Linen linear weights to NNX dict format."""
    e = cue.descriptors.linear(irreps_in, irreps_out)
    result, offset, seg_idx = {}, 0, 0
    for mul_in, ir_in in irreps_in:
        for mul_out, ir_out in irreps_out:
            if ir_in == ir_out:
                seg = e.polynomial.operands[0].segments[seg_idx]
                result[str(ir_in)] = w[offset : offset + seg[0] * seg[1]].reshape(seg)
                offset += seg[0] * seg[1]
                seg_idx += 1
    return result


def _convert_linen_to_nnx(
    linen_params,
    nnx_model,
    num_features,
    interaction_irreps,
    hidden_irreps,
    num_radial_basis,
):
    """Convert Linen MACE params to NNX model."""
    params = linen_params["params"]
    nnx_model.linear_embedding[...] = params["linear_embedding"]

    for layer_idx, nnx_layer in enumerate(nnx_model.layers):
        lp = params[f"layer_{layer_idx}"]
        first = layer_idx == 0
        last = layer_idx == len(nnx_model.layers) - 1
        hidden_out = hidden_irreps.filter(keep="0e") if last else hidden_irreps
        input_irreps = cue.Irreps(
            cue.O3, [(num_features, ir) for _, ir in hidden_irreps]
        )
        if first:
            input_irreps = input_irreps.filter(keep="0e")

        if nnx_layer.linZ_skip is not None:
            nnx_layer.linZ_skip.w[...] = lp["linZ_skip_tp"]
        if nnx_layer.linZ_first is not None:
            nnx_layer.linZ_first.w[...] = lp["linZ_skip_tp_first"]

        for ir, w in _convert_irreps_linear(
            lp["linear_up"], input_irreps, input_irreps
        ).items():
            nnx_layer.linear_up.w[ir][...] = w

        mlp_sizes = [num_radial_basis, 64, 64, 64, nnx_layer.tp.weight_dim]
        for i in range(len(mlp_sizes) - 1):
            nnx_layer.radial_mlp.linears[i][...] = lp["MultiLayerPerceptron_0"][
                f"Dense_{i}"
            ]["kernel"]

        for ir, w in _convert_irreps_linear(
            lp["linear_down"],
            nnx_layer.tp.irreps_out,
            num_features * interaction_irreps,
        ).items():
            nnx_layer.linear_down.w[ir][...] = w

        nnx_layer.symmetric_contraction.w[...] = lp["symmetric_contraction"]

        for ir, w in _convert_irreps_linear(
            lp["linear_post_sc"], num_features * hidden_out, num_features * hidden_out
        ).items():
            nnx_layer.linear_post_sc.w[ir][...] = w

        if last:
            mlp_irreps = cue.Irreps(cue.O3, "16x0e")
            for ir, w in _convert_irreps_linear(
                lp["linear_mlp_readout"], num_features * hidden_out, mlp_irreps
            ).items():
                nnx_layer.linear_mlp_readout.w[ir][...] = w
            for ir, w in _convert_irreps_linear(
                lp["linear_readout"], mlp_irreps, cue.Irreps(cue.O3, "1x0e")
            ).items():
                nnx_layer.linear_readout.w[ir][...] = w
        else:
            for ir, w in _convert_irreps_linear(
                lp["linear_readout"],
                num_features * hidden_out,
                cue.Irreps(cue.O3, "1x0e"),
            ).items():
                nnx_layer.linear_readout.w[ir][...] = w


def test_mace_linen_to_nnx_conversion():
    """Test that NNX model matches Linen after weight conversion."""
    num_atoms, num_edges, num_species, num_graphs = 5, 10, 3, 2
    num_features, num_radial_basis = 16, 4
    interaction_irreps = cue.Irreps(cue.O3, "0e+1o")
    hidden_irreps = cue.Irreps(cue.O3, "0e+1o")
    dtype = jnp.float32

    config = dict(
        num_layers=1,
        num_features=num_features,
        num_species=num_species,
        max_ell=2,
        correlation=2,
        num_radial_basis=num_radial_basis,
        interaction_irreps=interaction_irreps,
        hidden_irreps=hidden_irreps,
        offsets=np.zeros(num_species),
        cutoff=3.0,
        epsilon=0.1,
        skip_connection_first_layer=False,
    )

    # Create batch
    key = jax.random.key(42)
    keys = jax.random.split(key, 4)
    vecs = jax.random.normal(keys[0], (num_edges, 3), dtype)
    species = jax.random.randint(keys[1], (num_atoms,), 0, num_species, jnp.int32)
    senders, receivers = jax.random.randint(keys[2], (2, num_edges), 0, num_atoms)
    graph_index = jnp.sort(jax.random.randint(keys[3], (num_atoms,), 0, num_graphs))
    nats = jnp.zeros((num_graphs,), jnp.int32).at[graph_index].add(1)
    batch = dict(
        nn_vecs=vecs,
        species=species,
        inda=senders,
        indb=receivers,
        inde=graph_index,
        nats=nats,
        mask=jnp.ones((num_edges,), jnp.int32),
    )

    # Create and init Linen model
    linen_model = MACEModel(**config)
    linen_params = linen_model.init(jax.random.key(0), batch)

    # Create NNX model and convert weights
    nnx_model = MACEModelNNX(**config, dtype=dtype, rngs=nnx.Rngs(0))
    _convert_linen_to_nnx(
        linen_params,
        nnx_model,
        num_features,
        interaction_irreps,
        hidden_irreps,
        num_radial_basis,
    )

    # Compare outputs
    E_linen, F_linen = linen_model.apply(linen_params, batch)
    E_nnx, F_nnx = nnx_model(batch)

    assert jnp.allclose(E_linen, E_nnx, atol=1e-2), (
        f"E diff: {jnp.max(jnp.abs(E_linen - E_nnx))}"
    )
    assert jnp.allclose(F_linen, F_nnx, atol=1e-2), (
        f"F diff: {jnp.max(jnp.abs(F_linen - F_nnx))}"
    )


def test_nequip_model_basic():
    """Test basic NEQUIP model functionality."""
    # Small test case
    num_atoms = 10
    num_edges = 20
    num_species = 5
    num_graphs = 2
    dtype = jnp.float32
    avg_num_neighbors = 10

    # Create small model
    model = NEQUIPModel(
        num_layers=2,
        num_features=32,
        num_species=num_species,
        max_ell=2,
        cutoff=3.0,
        normalization_factor=1 / avg_num_neighbors,
    )

    # Create dummy data
    vecs = jax.random.normal(jax.random.key(0), (num_edges, 3), dtype)
    species = jax.random.randint(
        jax.random.key(1), (num_atoms,), 0, num_species, jnp.int32
    )
    senders, receivers = jax.random.randint(
        jax.random.key(2), (2, num_edges), 0, num_atoms, jnp.int32
    )
    graph_index = jax.random.randint(
        jax.random.key(3), (num_atoms,), 0, num_graphs, jnp.int32
    )
    graph_index = jnp.sort(graph_index)
    nats = jnp.zeros((num_graphs,), dtype=jnp.int32).at[graph_index].add(1)
    mask = jnp.ones((num_edges,), dtype=jnp.int32)

    batch_dict = dict(
        nn_vecs=vecs,
        species=species,
        inda=senders,
        indb=receivers,
        inde=graph_index,
        nats=nats,
        mask=mask,
    )

    # Initialize and run forward pass
    params = model.init(jax.random.key(0), batch_dict)
    E, F = model.apply(params, batch_dict)

    # Check output shapes
    assert E.shape == (num_graphs,), f"Energy shape {E.shape} != {(num_graphs,)}"
    assert F.shape == (num_atoms, 3), f"Forces shape {F.shape} != {(num_atoms, 3)}"
    assert E.dtype == dtype, f"Energy dtype {E.dtype} != {dtype}"
    assert F.dtype == dtype, f"Forces dtype {F.dtype} != {dtype}"
