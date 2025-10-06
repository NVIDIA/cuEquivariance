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

import cuequivariance as cue

from .mace import MACEModel
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
