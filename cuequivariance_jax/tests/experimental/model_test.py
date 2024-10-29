# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import *

import jax
import numpy as np
import pytest

import cuequivariance as cue
from cuequivariance_jax.experimental import FusionModel, NequipLikeModel


def make_inputs(num_elements: int, num_nodes: int, num_edges: int):
    positions = jax.random.normal(jax.random.key(0), (num_nodes, 3))
    node_features = jax.random.normal(jax.random.key(1), (num_nodes, 64))
    node_elements = jax.random.randint(jax.random.key(2), (num_nodes,), 0, num_elements)
    senders = jax.random.randint(jax.random.key(3), (num_edges,), 0, num_nodes)
    receivers = jax.random.randint(jax.random.key(4), (num_edges,), 0, num_nodes)
    edge_features = jax.random.normal(jax.random.key(5), (num_edges, 64))
    return positions, node_features, node_elements, senders, receivers, edge_features


def test_nequip_like():
    model = NequipLikeModel(
        num_layers=1,
        irreps_mid=8 * cue.Irreps("O3", "0e + 1o + 2e"),
        irreps_out="0e",
        cutoff_distance=10.0,
        num_radial_basis=10,
        mlp_num_hidden=32,
        mlp_num_layers=2,
        normalization_factor=0.1,
        use_cutoff_envelope=True,
    )
    positions, node_features, _, senders, receivers, edge_features = make_inputs(
        1, 10, 20
    )
    w = model.init(
        jax.random.PRNGKey(0),
        positions,
        node_features,
        senders,
        receivers,
        edge_features,
    )
    f = jax.jit(model.apply)
    out1 = f(w, positions, node_features, senders, receivers, edge_features).array
    R = -cue.SO3(1).rotation(np.array([1.0, 0.3, 0.4]), 1.2)
    out2 = f(w, positions @ R, node_features, senders, receivers, edge_features).array
    np.testing.assert_allclose(out1, out2, atol=1e-4)


@pytest.mark.parametrize("use_escn", [True, False])
def test_fusion(use_escn: bool):
    num_elements = 8
    model = FusionModel(
        num_elements=num_elements,
        num_layers=2,
        irreps_mid=8 * cue.Irreps("O3", "0e + 0o + 1o + 1e"),
        irreps_msg=cue.Irreps("O3", "0e"),
        irreps_out="0e",
        cutoff_distance=10.0,
        num_radial_basis=10,
        mlp_num_hidden=32,
        mlp_num_layers=2,
        normalization_factor=0.1,
        correlation=2,
        lmax_sh=3,
        use_cutoff_envelope=True,
        use_escn=use_escn,
    )
    positions, node_features, node_elements, senders, receivers, edge_features = (
        make_inputs(num_elements, 10, 20)
    )
    w = model.init(
        jax.random.PRNGKey(0),
        positions,
        node_features,
        node_elements,
        senders,
        receivers,
        edge_features,
    )
    f = jax.jit(model.apply)
    out1 = f(
        w, positions, node_features, node_elements, senders, receivers, edge_features
    ).array
    R = -cue.SO3(1).rotation(np.array([1.0, 0.3, 0.4]), 1.2)
    out2 = f(
        w,
        positions @ R,
        node_features,
        node_elements,
        senders,
        receivers,
        edge_features,
    ).array
    np.testing.assert_allclose(out1, out2, atol=1e-4)
