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
import pytest
from flax import nnx

import cuequivariance as cue

from .mace_linen import MACEModel
from .mace_nnx import MACEModel as MACEModelNNX
from .nequip_linen import NEQUIPModel
from .nequip_nnx import NEQUIPModel as NEQUIPModelNNX


def _create_test_graph_batch(num_atoms=10, num_edges=20, num_species=5, num_graphs=2):
    """Create random graph batch for testing."""
    key = jax.random.key(42)
    keys = jax.random.split(key, 4)
    graph_index = jnp.sort(jax.random.randint(keys[3], (num_atoms,), 0, num_graphs))
    return dict(
        nn_vecs=jax.random.normal(keys[0], (num_edges, 3)),
        species=jax.random.randint(keys[1], (num_atoms,), 0, num_species, jnp.int32),
        inda=jax.random.randint(keys[2], (num_edges,), 0, num_atoms, jnp.int32),
        indb=jax.random.randint(keys[2], (num_edges,), 0, num_atoms, jnp.int32),
        inde=graph_index,
        nats=jnp.zeros((num_graphs,), jnp.int32).at[graph_index].add(1),
        mask=jnp.ones((num_edges,), jnp.int32),
    )


def test_mace_linen_forward_pass_output_shapes():
    """Test Linen MACE model forward pass produces correct output shapes."""
    batch = _create_test_graph_batch()
    model = MACEModel(
        num_layers=1,
        num_features=32,
        num_species=5,
        max_ell=2,
        correlation=2,
        num_radial_basis=4,
        interaction_irreps=cue.Irreps(cue.O3, "0e+1o"),
        hidden_irreps=cue.Irreps(cue.O3, "0e"),
        offsets=np.zeros(5),
        cutoff=3.0,
        epsilon=0.1,
        skip_connection_first_layer=True,
    )
    params = model.init(jax.random.key(0), batch)
    E, F = model.apply(params, batch)
    assert E.shape == (2,) and F.shape == (10, 3)


def test_mace_nnx_forward_pass_output_shapes():
    """Test NNX MACE model forward pass produces correct output shapes."""
    batch = _create_test_graph_batch()
    model = MACEModelNNX(
        num_layers=1,
        num_features=32,
        num_species=5,
        max_ell=2,
        correlation=2,
        num_radial_basis=4,
        interaction_irreps=cue.Irreps(cue.O3, "0e+1o"),
        hidden_irreps=cue.Irreps(cue.O3, "0e"),
        offsets=np.zeros(5),
        cutoff=3.0,
        epsilon=0.1,
        skip_connection_first_layer=True,
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    E, F = model(batch)
    assert E.shape == (2,) and F.shape == (10, 3)
    assert jnp.all(jnp.isfinite(E)) and jnp.all(jnp.isfinite(F))


@pytest.mark.parametrize(
    "hidden_irreps_str,interaction_irreps_str,skip_first",
    [
        ("0e", "0e+1o", False),  # OFF-S like config
        ("0e+1o", "0e+1o+2e", False),  # OFF-M like config
        ("0e", "0e+1o", True),  # MP-S like config
        ("0e+1o", "0e+1o+2e", True),  # MP-M like config
    ],
    ids=["OFF-S", "OFF-M", "MP-S", "MP-M"],
)
def test_mace_linen_nnx_weight_transfer_produces_identical_outputs(
    hidden_irreps_str, interaction_irreps_str, skip_first
):
    """Test Linen and NNX MACE models produce identical outputs after weight transfer."""
    num_features, num_species = 16, 3
    interaction_irreps = cue.Irreps(cue.O3, interaction_irreps_str)
    hidden_irreps = cue.Irreps(cue.O3, hidden_irreps_str)

    config = dict(
        num_layers=1,
        num_features=num_features,
        num_species=num_species,
        max_ell=2,
        correlation=2,
        num_radial_basis=4,
        interaction_irreps=interaction_irreps,
        hidden_irreps=hidden_irreps,
        offsets=np.zeros(num_species),
        cutoff=3.0,
        epsilon=0.1,
        skip_connection_first_layer=skip_first,
    )
    batch = _create_test_graph_batch(
        num_atoms=5, num_edges=10, num_species=num_species, num_graphs=1
    )

    linen_model = MACEModel(**config)
    linen_params = linen_model.init(jax.random.key(0), batch)

    nnx_model = MACEModelNNX(**config, dtype=jnp.float32, rngs=nnx.Rngs(0))
    _transfer_mace_linen_params_to_nnx(linen_params, nnx_model, config)

    E_linen, F_linen = linen_model.apply(linen_params, batch)
    E_nnx, F_nnx = nnx_model(batch)

    np.testing.assert_allclose(E_linen, E_nnx, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(F_linen, F_nnx, atol=1e-3, rtol=1e-3)


def _transfer_mace_linen_params_to_nnx(linen_params, nnx_model, config):
    """Transfer Linen MACE parameters to NNX model."""
    num_features = config["num_features"]
    interaction_irreps = config["interaction_irreps"]
    hidden_irreps = config["hidden_irreps"]

    def convert_linear(w, irreps_in, irreps_out):
        """Convert flat linear weights to dict[str, Array] format."""
        e = cue.descriptors.linear(irreps_in, irreps_out)
        result, offset, seg_idx = {}, 0, 0
        for _, ir_in in irreps_in:
            for _, ir_out in irreps_out:
                if ir_in == ir_out:
                    seg = e.polynomial.operands[0].segments[seg_idx]
                    size = seg[0] * seg[1]
                    result[str(ir_in)] = w[offset : offset + size].reshape(seg)
                    offset, seg_idx = offset + size, seg_idx + 1
        return result

    params = linen_params["params"]
    nnx_model.embedding[...] = params["linear_embedding"]

    for layer_idx, layer in enumerate(nnx_model.layers):
        lp = params[f"layer_{layer_idx}"]
        is_first = layer_idx == 0
        is_last = layer_idx == len(nnx_model.layers) - 1
        hidden_out = hidden_irreps.filter(keep="0e") if is_last else hidden_irreps
        input_irreps = cue.Irreps(
            cue.O3, [(num_features, ir) for _, ir in hidden_irreps]
        )
        if is_first:
            input_irreps = input_irreps.filter(keep="0e")

        # Skip connection (linZ_skip_tp -> skip)
        if layer.skip is not None:
            layer.skip.w[...] = lp["linZ_skip_tp"]

        # Linear up
        for ir, w in convert_linear(
            lp["linear_up"], input_irreps, input_irreps
        ).items():
            layer.linear_up.w[ir][...] = w

        # Radial MLP
        for i in range(4):
            layer.radial_mlp.linears[i][...] = lp["MultiLayerPerceptron_0"][
                f"Dense_{i}"
            ]["kernel"]

        # Linear down
        for ir, w in convert_linear(
            lp["linear_down"],
            layer.message.irreps_out,
            num_features * interaction_irreps,
        ).items():
            layer.linear_down.w[ir][...] = w

        # linZ_first (OFF models only, first layer)
        if layer.linZ_first is not None:
            layer.linZ_first.w[...] = lp["linZ_skip_tp_first"]

        # Symmetric contraction
        layer.sc.w[...] = lp["symmetric_contraction"]

        # Linear post SC
        for ir, w in convert_linear(
            lp["linear_post_sc"], num_features * hidden_out, num_features * hidden_out
        ).items():
            layer.linear_sc.w[ir][...] = w

        # Readout
        if is_last:
            mlp_irreps = cue.Irreps(cue.O3, "16x0e")
            for ir, w in convert_linear(
                lp["linear_mlp_readout"], num_features * hidden_out, mlp_irreps
            ).items():
                layer.readout_mlp.w[ir][...] = w
            for ir, w in convert_linear(
                lp["linear_readout"], mlp_irreps, cue.Irreps(cue.O3, "1x0e")
            ).items():
                layer.readout.w[ir][...] = w
        else:
            for ir, w in convert_linear(
                lp["linear_readout"],
                num_features * hidden_out,
                cue.Irreps(cue.O3, "1x0e"),
            ).items():
                layer.readout.w[ir][...] = w


def test_mace_linen_nnx_training_produces_identical_params():
    """Test Linen and NNX MACE models produce identical parameters after training."""
    import optax

    num_features, num_species = 16, 3
    num_steps = 5
    learning_rate = 1e-2

    config = dict(
        num_layers=1,
        num_features=num_features,
        num_species=num_species,
        max_ell=2,
        correlation=2,
        num_radial_basis=4,
        interaction_irreps=cue.Irreps(cue.O3, "0e+1o+2e"),
        hidden_irreps=cue.Irreps(cue.O3, "0e+1o"),
        offsets=np.zeros(num_species),
        cutoff=3.0,
        epsilon=0.1,
        skip_connection_first_layer=True,
    )
    batch = _create_test_graph_batch(
        num_atoms=5, num_edges=10, num_species=num_species, num_graphs=1
    )

    key = jax.random.key(123)
    target_E = jax.random.normal(key, (1,))
    target_F = jax.random.normal(jax.random.split(key)[0], (5, 3))

    # Initialize Linen model
    linen_model = MACEModel(**config)
    linen_params = linen_model.init(jax.random.key(0), batch)

    # Initialize NNX model with converted weights
    nnx_model = MACEModelNNX(**config, dtype=jnp.float32, rngs=nnx.Rngs(0))
    _transfer_mace_linen_params_to_nnx(linen_params, nnx_model, config)

    # Verify initial outputs match (use 1e-3 tolerance like existing test)
    E_linen_init, F_linen_init = linen_model.apply(linen_params, batch)
    E_nnx_init, F_nnx_init = nnx_model(batch)
    np.testing.assert_allclose(E_linen_init, E_nnx_init, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(F_linen_init, F_nnx_init, atol=1e-3, rtol=1e-3)
    print("\nInitial outputs match (within 1e-3):")
    print(f"  E_linen: {E_linen_init}, E_nnx: {E_nnx_init}")

    # Train Linen model
    def loss_fn_linen(w):
        E, F = linen_model.apply(w, batch)
        return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

    tx = optax.adam(learning_rate)
    opt_state_linen = tx.init(linen_params)
    for _ in range(num_steps):
        grad = jax.grad(loss_fn_linen)(linen_params)
        updates, opt_state_linen = tx.update(grad, opt_state_linen, linen_params)
        linen_params = optax.apply_updates(linen_params, updates)

    # Train NNX model
    def loss_fn_nnx(model):
        E, F = model(batch)
        return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

    optimizer_nnx = nnx.Optimizer(nnx_model, optax.adam(learning_rate), wrt=nnx.Param)
    for _ in range(num_steps):
        grads = nnx.grad(loss_fn_nnx)(nnx_model)
        optimizer_nnx.update(nnx_model, grads)

    # Compare outputs after training
    E_linen, F_linen = linen_model.apply(linen_params, batch)
    E_nnx, F_nnx = nnx_model(batch)

    print(f"\nAfter {num_steps} training steps:")
    print(f"  Linen loss: {loss_fn_linen(linen_params):.6f}")
    print(f"  NNX loss:   {loss_fn_nnx(nnx_model):.6f}")
    print(f"  E_linen: {E_linen}")
    print(f"  E_nnx:   {E_nnx}")
    print(f"  max|E diff|: {jnp.max(jnp.abs(E_linen - E_nnx)):.2e}")
    print(f"  max|F diff|: {jnp.max(jnp.abs(F_linen - F_nnx)):.2e}")

    # Compare multiple parameters
    print("  Parameter comparisons:")

    # 1. Embedding
    linen_emb = linen_params["params"]["linear_embedding"]
    nnx_emb = nnx_model.embedding[...]
    emb_diff = jnp.max(jnp.abs(linen_emb - nnx_emb))
    print(f"    embedding: max|diff| = {emb_diff:.2e}")

    # 2. Skip connection weights (linZ_skip_tp -> skip.w)
    linen_skip = linen_params["params"]["layer_0"]["linZ_skip_tp"]
    nnx_skip = nnx_model.layers[0].skip.w[...]
    skip_diff = jnp.max(jnp.abs(linen_skip - nnx_skip))
    print(f"    layer_0/skip.w: max|diff| = {skip_diff:.2e}")

    # 3. Symmetric contraction weights
    linen_sc = linen_params["params"]["layer_0"]["symmetric_contraction"]
    nnx_sc = nnx_model.layers[0].sc.w[...]
    sc_diff = jnp.max(jnp.abs(linen_sc - nnx_sc))
    print(f"    layer_0/sc.w: max|diff| = {sc_diff:.2e}")

    # 4. Radial MLP weights (first layer)
    linen_mlp = linen_params["params"]["layer_0"]["MultiLayerPerceptron_0"]["Dense_0"][
        "kernel"
    ]
    nnx_mlp = nnx_model.layers[0].radial_mlp.linears[0][...]
    mlp_diff = jnp.max(jnp.abs(linen_mlp - nnx_mlp))
    print(f"    layer_0/radial_mlp[0]: max|diff| = {mlp_diff:.2e}")

    np.testing.assert_allclose(E_linen, E_nnx, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(F_linen, F_nnx, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(linen_emb, nnx_emb, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(linen_skip, nnx_skip, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(linen_sc, nnx_sc, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(linen_mlp, nnx_mlp, atol=1e-3, rtol=1e-3)


def test_nequip_linen_forward_pass_output_shapes():
    """Test Linen NequIP model forward pass produces correct output shapes."""
    batch = _create_test_graph_batch()
    model = NEQUIPModel(
        num_layers=2,
        num_features=32,
        num_species=5,
        max_ell=2,
        cutoff=3.0,
        normalization_factor=0.1,
    )
    params = model.init(jax.random.key(0), batch)
    E, F = model.apply(params, batch)
    assert E.shape == (2,) and F.shape == (10, 3)


def test_nequip_nnx_forward_pass_output_shapes():
    """Test NNX NequIP model forward pass produces correct output shapes."""
    batch = _create_test_graph_batch()
    model = NEQUIPModelNNX(
        num_layers=2,
        num_features=32,
        num_species=5,
        max_ell=2,
        cutoff=3.0,
        normalization_factor=0.1,
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    E, F = model(batch)
    assert E.shape == (2,) and F.shape == (10, 3)
    assert jnp.all(jnp.isfinite(E)) and jnp.all(jnp.isfinite(F))


def _transfer_nequip_linen_params_to_nnx(linen_params, nnx_model, config):
    """Transfer Linen NequIP parameters to NNX model."""
    num_features = config["num_features"]
    max_ell = config["max_ell"]
    output_irreps = num_features * cue.Irreps(
        cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in range(max_ell + 1)]
    )

    params = linen_params["params"]

    # Transfer embedding
    nnx_model.embedding[...] = params["linear_embedding"]

    # Transfer layers
    for layer_idx, layer in enumerate(nnx_model.layers):
        lp = params[f"layer_{layer_idx}"]
        is_first = layer_idx == 0

        # Input irreps for this layer
        input_irreps = output_irreps.filter(keep="0e") if is_first else output_irreps

        # linear_up (IrrepsLinear): Linen flat -> NNX per-irrep dict
        # Weight layout: [0e->0e, 1o->1o, 2e->2e, ...] in order of irreps
        linear_up_w = lp["linear_up"]
        offset = 0
        for mul, ir in input_irreps:
            ir_key = str(ir)
            size = mul * mul  # square matrix per irrep
            layer.linear_up.w[ir_key][...] = linear_up_w[
                offset : offset + size
            ].reshape(mul, mul)
            offset += size

        # Radial MLP
        mlp_params = lp["MultiLayerPerceptron_0"]
        for i in range(len(layer.radial_mlp.linears)):
            layer.radial_mlp.linears[i][...] = mlp_params[f"Dense_{i}"]["kernel"]

        # linear_skip (GatedIndexedLinear): Linen (num_species, weight_dim)
        # Linen weight layout: [0e->scalars, 0e->gates, 1o->1o, 2e->2e, 3o->3o, ...]
        linear_skip = lp["linear_skip"]
        in_0e_mul = sum(m for m, ir in input_irreps if ir == cue.O3(0, 1))
        num_scalars = layer.num_scalars
        num_gates = layer.num_nonscalar

        skip_offset = 0
        # 0e -> scalars
        size = in_0e_mul * num_scalars
        layer.linear_skip.w_0e_scalar[...] = linear_skip[
            :, skip_offset : skip_offset + size
        ]
        skip_offset += size
        # 0e -> gates
        size = in_0e_mul * num_gates
        layer.linear_skip.w_0e_gates[...] = linear_skip[
            :, skip_offset : skip_offset + size
        ]
        skip_offset += size
        # Non-scalar irreps
        skip_w_keys = set(layer.linear_skip.w_nonscalar.keys())
        for mul_out, ir in layer.linear_skip.irreps_nonscalar:
            ir_key = str(ir)
            in_mul_ir = sum(m for m, i in input_irreps if i == ir)
            if ir_key in skip_w_keys and in_mul_ir > 0:
                size = in_mul_ir * mul_out
                layer.linear_skip.w_nonscalar[ir_key][...] = linear_skip[
                    :, skip_offset : skip_offset + size
                ]
                skip_offset += size

        # linear_down (GatedLinear): Linen flat -> NNX split structure
        # Weight layout: [0e->scalars, 0e->gates, 1o->1o, 2e->2e, 3o->3o, ...]
        linear_down = lp["linear_down"]
        msg_irreps = layer.message.irreps_out
        msg_0e_mul = sum(m for m, ir in msg_irreps if ir == cue.O3(0, 1))

        offset = 0
        # 0e -> scalars
        size = msg_0e_mul * num_scalars
        layer.linear_down.w_0e_scalar[...] = linear_down[
            offset : offset + size
        ].reshape(msg_0e_mul, num_scalars)
        offset += size
        # 0e -> gates
        size = msg_0e_mul * num_gates
        layer.linear_down.w_0e_gates[...] = linear_down[offset : offset + size].reshape(
            msg_0e_mul, num_gates
        )
        offset += size
        # Non-scalar irreps
        for mul, ir in layer.linear_down.irreps_nonscalar:
            ir_key = str(ir)
            if ir_key in set(layer.linear_down.w_nonscalar.keys()):
                in_mul_ir = sum(m for m, i in msg_irreps if i == ir)
                if in_mul_ir > 0:
                    size = in_mul_ir * mul
                    layer.linear_down.w_nonscalar[ir_key][...] = linear_down[
                        offset : offset + size
                    ].reshape(in_mul_ir, mul)
                    offset += size

    # Transfer energy readout
    nnx_model.readout.w["0e"][...] = params["energy_readout"].reshape(num_features, 1)


def test_nequip_linen_nnx_weight_transfer_produces_identical_outputs():
    """Test Linen and NNX NequIP models produce identical outputs after weight transfer."""
    num_features, num_species = 32, 3

    config = dict(
        num_layers=2,
        num_features=num_features,
        num_species=num_species,
        max_ell=3,
        cutoff=5.0,
        normalization_factor=0.1,
    )
    batch = _create_test_graph_batch(
        num_atoms=5, num_edges=10, num_species=num_species, num_graphs=1
    )

    linen_model = NEQUIPModel(**config)
    linen_params = linen_model.init(jax.random.key(0), batch)

    nnx_model = NEQUIPModelNNX(**config, dtype=jnp.float32, rngs=nnx.Rngs(0))
    _transfer_nequip_linen_params_to_nnx(linen_params, nnx_model, config)

    E_linen, F_linen = linen_model.apply(linen_params, batch)
    E_nnx, F_nnx = nnx_model(batch)

    np.testing.assert_allclose(E_linen, E_nnx, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(F_linen, F_nnx, atol=1e-3, rtol=1e-3)


def test_nequip_linen_nnx_training_produces_identical_params():
    """Test Linen and NNX NequIP models produce identical parameters after training."""
    import optax

    num_features, num_species = 32, 3
    num_steps = 5
    learning_rate = 1e-2

    config = dict(
        num_layers=2,
        num_features=num_features,
        num_species=num_species,
        max_ell=3,
        cutoff=5.0,
        normalization_factor=0.1,
    )
    batch = _create_test_graph_batch(
        num_atoms=5, num_edges=10, num_species=num_species, num_graphs=1
    )

    key = jax.random.key(123)
    target_E = jax.random.normal(key, (1,))
    target_F = jax.random.normal(jax.random.split(key)[0], (5, 3))

    # Initialize Linen model
    linen_model = NEQUIPModel(**config)
    linen_params = linen_model.init(jax.random.key(0), batch)

    # Initialize NNX model with converted weights
    nnx_model = NEQUIPModelNNX(**config, dtype=jnp.float32, rngs=nnx.Rngs(0))
    _transfer_nequip_linen_params_to_nnx(linen_params, nnx_model, config)

    # Verify initial outputs match
    E_linen_init, F_linen_init = linen_model.apply(linen_params, batch)
    E_nnx_init, F_nnx_init = nnx_model(batch)
    np.testing.assert_allclose(E_linen_init, E_nnx_init, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(F_linen_init, F_nnx_init, atol=1e-3, rtol=1e-3)

    # Train Linen model
    def loss_fn_linen(w):
        E, F = linen_model.apply(w, batch)
        return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

    tx = optax.adam(learning_rate)
    opt_state_linen = tx.init(linen_params)
    for _ in range(num_steps):
        grad = jax.grad(loss_fn_linen)(linen_params)
        updates, opt_state_linen = tx.update(grad, opt_state_linen, linen_params)
        linen_params = optax.apply_updates(linen_params, updates)

    # Train NNX model
    def loss_fn_nnx(model):
        E, F = model(batch)
        return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

    optimizer_nnx = nnx.Optimizer(nnx_model, optax.adam(learning_rate), wrt=nnx.Param)
    for _ in range(num_steps):
        grads = nnx.grad(loss_fn_nnx)(nnx_model)
        optimizer_nnx.update(nnx_model, grads)

    # Compare outputs after training
    E_linen, F_linen = linen_model.apply(linen_params, batch)
    E_nnx, F_nnx = nnx_model(batch)

    # Compare key parameters
    linen_emb = linen_params["params"]["linear_embedding"]
    nnx_emb = nnx_model.embedding[...]

    # Reconstruct NNX linear_skip weights to compare with Linen's flat format
    skip_layer = nnx_model.layers[0].linear_skip
    nnx_skip_parts = [skip_layer.w_0e_scalar[...], skip_layer.w_0e_gates[...]]
    for ir_key in skip_layer.w_nonscalar.keys():
        nnx_skip_parts.append(skip_layer.w_nonscalar[ir_key][...])
    nnx_skip = jnp.concatenate(nnx_skip_parts, axis=-1)
    linen_skip = linen_params["params"]["layer_0"]["linear_skip"]

    np.testing.assert_allclose(E_linen, E_nnx, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(F_linen, F_nnx, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(linen_emb, nnx_emb, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(linen_skip, nnx_skip, atol=1e-3, rtol=1e-3)
