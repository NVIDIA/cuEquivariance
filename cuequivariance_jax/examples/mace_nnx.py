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

"""
MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields

This implementation uses NNX and dict[Irrep, Array] representation instead of RepArray.

Based on the original MACE paper:
Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022).
MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.
arXiv preprint arXiv:2206.07697. https://arxiv.org/abs/2206.07697
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from cuequivariance.group_theory.experimental.mace import (
    symmetric_contraction as mace_symmetric_contraction,
)
from cuequivariance_jax.nnx import (
    MLP,
    IrrepsIndexedLinear,
    IrrepsLinear,
    SphericalHarmonics,
)
from einops import rearrange
from flax import nnx
from jax import Array

import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance import Irrep


def polynomial_envelope(x: Array, r_max: float) -> Array:
    """Polynomial cutoff envelope function."""
    p = 5
    xs = x / r_max
    xp = jnp.power(xs, p)
    return (
        1.0
        - 0.5 * (p + 1.0) * (p + 2.0) * xp
        + p * (p + 2.0) * xp * xs
        - 0.5 * p * (p + 1.0) * xp * xs * xs
    )


def bessel(x: Array, n: int, x_max: float = 1.0) -> Array:
    x = jnp.asarray(x)
    x = x[..., None]
    ns = jnp.arange(1, n + 1, dtype=x.dtype)
    return jnp.sqrt(2.0 / x_max) * jnp.pi * ns / x_max * jnp.sinc(ns * x / x_max)


def radial_basis_function(
    edge_length: Array, r_max: float, num_radial_basis: int
) -> Array:
    """Radial basis function with polynomial cutoff."""
    cutoff = jnp.where(
        edge_length < r_max, polynomial_envelope(edge_length, r_max), 0.0
    )
    return bessel(edge_length, num_radial_basis, r_max) * cutoff


def irreps_scalar_activation(
    x: dict[Irrep, Array], activation: Callable
) -> dict[Irrep, Array]:
    """Apply activation to scalar components only."""
    activation = cuex.normalize_function(activation)
    return {ir: activation(v) if ir.is_scalar() else v for ir, v in x.items()}


class ChannelwiseTensorProduct(nnx.Module):
    """Channelwise tensor product for message passing.

    Uses split_operand_by_irrep to work with dict[Irrep, Array] inputs/outputs
    via segmented_polynomial_uniform_1d, avoiding memory contiguity constraints.
    """

    def __init__(
        self,
        irreps_in1: cue.Irreps,
        irreps_in2: cue.Irreps,
        irreps_out: cue.Irreps,
        *,
        epsilon: float = 1.0,
        name: str = "channelwise_tp",
    ):
        assert irreps_in1.regroup() == irreps_in1
        assert irreps_in2.regroup() == irreps_in2
        assert irreps_out.regroup() == irreps_out

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.name = name

        e = (
            cue.descriptors.channelwise_tensor_product(
                irreps_in1, irreps_in2, irreps_out, True
            )
            * epsilon
        )
        self.weight_dim = e.inputs[0].dim
        self.irreps_out = e.outputs[0].irreps

        self.p = (
            e.split_operand_by_irrep(2)  # split x2 (spherical harmonics) by irrep
            .split_operand_by_irrep(1)  # split x1 (node features) by irrep
            .split_operand_by_irrep(-1)  # split output by irrep
            .polynomial
        )

    def __call__(
        self,
        weights: Array,
        x1: dict[Irrep, Array],
        x2: dict[Irrep, Array],
        *,
        senders: Array | None = None,
        receivers: Array | None = None,
        num_nodes: int | None = None,
    ) -> dict[Irrep, Array]:
        """
        Args:
            weights: radial weights of shape (num_edges, weight_dim)
            x1: node features to gather by senders, shape (num_nodes, mul, ir.dim)
            x2: edge features (spherical harmonics), shape (num_edges, mul, ir.dim)
            senders: indices for gathering x1
            receivers: indices for scattering output
            num_nodes: number of nodes for output shape
        """
        cuex.ir_dict.assert_mul_ir_dict(self.irreps_in1, x1)
        cuex.ir_dict.assert_mul_ir_dict(self.irreps_in2, x2)

        w = rearrange(weights, "... (x m) -> ... x m", x=self.p.inputs[0].num_segments)
        x1 = jax.tree.map(lambda v: rearrange(v, "... m i -> ... i m"), x1)
        x2 = jax.tree.map(lambda v: rearrange(v, "... 1 i -> ... i"), x2)

        y = {
            ir: jax.ShapeDtypeStruct(
                (num_nodes, desc.num_segments) + desc.segment_shape, w.dtype
            )
            for (_, ir), desc in zip(self.irreps_out, self.p.outputs)
        }
        y = cuex.ir_dict.segmented_polynomial_uniform_1d(
            self.p,
            [w, x1, x2],
            y,
            input_indices=[None, senders, None],
            output_indices=receivers,
            name=self.name,
        )

        y = {
            ir: rearrange(v, "... (i x) m -> ... (x m) i", i=ir.dim)
            for ir, v in y.items()
        }

        cuex.ir_dict.assert_mul_ir_dict(self.irreps_out, y)
        return y


class SymmetricContraction(nnx.Module):
    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        correlation: int,
        num_species: int,
        num_features: int,
        *,
        name: str = "symmetric_contraction",
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.name = name
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.num_species = num_species
        self.num_features = num_features

        e, projection = mace_symmetric_contraction(
            irreps_in, irreps_out, range(1, correlation + 1)
        )
        self.projection = jnp.array(projection, dtype=dtype)
        n = self.projection.shape[0]

        self.p = (
            e.split_operand_by_irrep(1)  # split node features input by irrep
            .split_operand_by_irrep(-1)  # split output by irrep
            .polynomial
        )

        self.w = nnx.Param(
            jax.random.normal(rngs.params(), (num_species, n, num_features), dtype)
        )

    def __call__(
        self, x: dict[Irrep, Array], num_species_counts: Array
    ) -> dict[Irrep, Array]:
        cuex.ir_dict.assert_mul_ir_dict(self.irreps_in, x)

        w = jnp.einsum("zau,ab->zbu", self.w[...], self.projection)

        num_nodes = jax.tree.leaves(x)[0].shape[0]
        i = jnp.repeat(
            jnp.arange(self.num_species),
            num_species_counts,
            total_repeat_length=num_nodes,
        )

        x = jax.tree.map(lambda v: rearrange(v, "... m i -> ... i m"), x)

        y = cuex.ir_dict.mul_ir_dict(self.irreps_out, None)
        y = cuex.ir_dict.segmented_polynomial_uniform_1d(
            self.p,
            [w, x],
            y,
            input_indices=[i, None],
            name=self.name,
        )

        y = jax.tree.map(lambda v: rearrange(v, "... i m -> ... m i"), y)

        cuex.ir_dict.assert_mul_ir_dict(self.irreps_out, y)
        return y


class MACELayer(nnx.Module):
    """Single MACE layer using NNX and dict[Irrep, Array] representation."""

    def __init__(
        self,
        first: bool,
        last: bool,
        num_species: int,
        num_features: int,
        interaction_irreps: cue.Irreps,
        hidden_irreps: cue.Irreps,
        activation: Callable,
        epsilon: float,
        max_ell: int,
        correlation: int,
        output_irreps: cue.Irreps,
        readout_mlp_irreps: cue.Irreps,
        radial_embedding_dim: int,
        skip_connection_first_layer: bool = False,
        *,
        name: str = "mace_layer",
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.last = last
        self.activation = activation

        hidden_out = hidden_irreps.filter(keep=output_irreps) if last else hidden_irreps

        sph_irreps = cue.Irreps(
            cue.O3,
            [(1, cue.O3(L, (-1) ** L)) for L in range(max_ell + 1)],
        )

        input_irreps = hidden_irreps.set_mul(num_features)
        if first:
            input_irreps = input_irreps.filter(keep="0e")

        self.linear_up = IrrepsLinear(
            input_irreps, input_irreps, dtype=dtype, rngs=rngs
        )

        self.tp = ChannelwiseTensorProduct(
            input_irreps,
            sph_irreps,
            num_features * interaction_irreps,
            epsilon=epsilon,
            name=f"{name}_tp",
        )

        self.radial_mlp = MLP(
            [radial_embedding_dim, 64, 64, 64, self.tp.weight_dim],
            activation,
            output_activation=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.linear_down = IrrepsLinear(
            self.tp.irreps_out,
            num_features * interaction_irreps,
            dtype=dtype,
            rngs=rngs,
        )

        if not first or skip_connection_first_layer:
            self.linZ_skip = IrrepsIndexedLinear(
                input_irreps,
                num_features * hidden_out,
                num_species,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.linZ_skip = None

        if first and not skip_connection_first_layer:
            self.linZ_first = IrrepsIndexedLinear(
                num_features * interaction_irreps,
                num_features * interaction_irreps,
                num_species,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.linZ_first = None

        self.symmetric_contraction = SymmetricContraction(
            num_features * interaction_irreps,
            num_features * hidden_out,
            correlation,
            num_species,
            num_features,
            name=f"{name}_sc",
            dtype=dtype,
            rngs=rngs,
        )

        self.linear_post_sc = IrrepsLinear(
            num_features * hidden_out,
            num_features * hidden_out,
            dtype=dtype,
            rngs=rngs,
        )

        if last:
            self.linear_mlp_readout = IrrepsLinear(
                num_features * hidden_out,
                readout_mlp_irreps,
                dtype=dtype,
                rngs=rngs,
            )

        self.linear_readout = IrrepsLinear(
            readout_mlp_irreps if last else num_features * hidden_out,
            output_irreps,
            dtype=dtype,
            rngs=rngs,
        )

        self.spherical_harmonics = SphericalHarmonics(max_ell)

    def __call__(
        self,
        vectors: Array,
        node_feats: dict[Irrep, Array],
        num_species_counts: Array,
        radial_embeddings: Array,
        senders: Array,
        receivers: Array,
        num_nodes: int,
    ) -> tuple[dict[Irrep, Array], dict[Irrep, Array]]:
        """
        Args:
            vectors: edge vectors of shape (num_edges, 3)
            node_feats: node features as dict[Irrep, Array]
            num_species_counts: count of atoms per species of shape (num_species,)
            radial_embeddings: radial basis of shape (num_edges, radial_dim)
            senders: sender indices of shape (num_edges,)
            receivers: receiver indices of shape (num_edges,)
            num_nodes: total number of nodes
        """
        sph = self.spherical_harmonics(vectors)

        self_connection = None
        if self.linZ_skip is not None:
            self_connection = self.linZ_skip(node_feats, num_species_counts)

        node_feats = self.linear_up(node_feats)

        mix = self.radial_mlp(radial_embeddings)

        node_feats = self.tp(
            mix,
            node_feats,
            sph,
            senders=senders,
            receivers=receivers,
            num_nodes=num_nodes,
        )

        node_feats = self.linear_down(node_feats)

        if self.linZ_first is not None:
            node_feats = self.linZ_first(node_feats, num_species_counts)

        node_feats = self.symmetric_contraction(node_feats, num_species_counts)
        node_feats = self.linear_post_sc(node_feats)

        if self_connection is not None:
            node_feats = cuex.ir_dict.irreps_add(node_feats, self_connection)

        node_outputs = node_feats
        if self.last:
            node_outputs = self.linear_mlp_readout(node_outputs)
            node_outputs = irreps_scalar_activation(node_outputs, self.activation)

        node_outputs = self.linear_readout(node_outputs)

        return node_outputs, node_feats


class MACEModel(nnx.Module):
    """Full MACE model using NNX and dict[Irrep, Array] representation."""

    def __init__(
        self,
        offsets: np.ndarray,
        num_species: int,
        cutoff: float,
        num_layers: int,
        num_features: int,
        interaction_irreps: cue.Irreps,
        hidden_irreps: cue.Irreps,
        max_ell: int,
        correlation: int,
        num_radial_basis: int,
        epsilon: float,
        skip_connection_first_layer: bool,
        *,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.offsets = offsets
        self.num_species = num_species
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.num_features = num_features
        self.interaction_irreps = interaction_irreps
        self.hidden_irreps = hidden_irreps
        self.max_ell = max_ell
        self.correlation = correlation
        self.num_radial_basis = num_radial_basis
        self.epsilon = epsilon
        self.skip_connection_first_layer = skip_connection_first_layer

        output_irreps = cue.Irreps(cue.O3, "1x0e")
        readout_mlp_irreps = cue.Irreps(cue.O3, "16x0e")

        self.linear_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (num_species, num_features), dtype)
        )

        layers = []
        for i in range(num_layers):
            layer = MACELayer(
                first=(i == 0),
                last=(i == num_layers - 1),
                num_species=num_species,
                num_features=num_features,
                interaction_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                activation=jax.nn.silu,
                epsilon=epsilon,
                max_ell=max_ell,
                correlation=correlation,
                output_irreps=output_irreps,
                readout_mlp_irreps=readout_mlp_irreps,
                radial_embedding_dim=num_radial_basis,
                skip_connection_first_layer=skip_connection_first_layer,
                name=f"layer_{i}",
                dtype=dtype,
                rngs=rngs,
            )
            layers.append(layer)
        self.layers = nnx.List(layers)

    def __call__(self, batch: dict[str, Array]) -> tuple[Array, Array]:
        vecs = batch["nn_vecs"]
        species = batch["species"]
        senders = batch["inda"]
        receivers = batch["indb"]
        graph_index = batch["inde"]
        mask = batch["mask"]
        num_graphs = jnp.shape(batch["nats"])[0]

        perm = jnp.argsort(species)
        species = species[perm]
        graph_index = graph_index[perm]
        inv_perm = jnp.zeros_like(perm).at[perm].set(jnp.arange(perm.shape[0]))
        senders, receivers = inv_perm[senders], inv_perm[receivers]
        num_species_counts = (
            jnp.zeros((self.num_species,), dtype=jnp.int32).at[species].add(1)
        )
        num_nodes = jnp.shape(species)[0]

        def model(vecs: Array) -> tuple[Array, Array]:
            embedding = self.linear_embedding[...]
            node_feats_flat = jnp.repeat(
                embedding, num_species_counts, axis=0, total_repeat_length=num_nodes
            ) / jnp.sqrt(self.num_species)

            scalar_ir = cue.O3(0, 1)  # 0e - scalar with even parity
            node_feats = {scalar_ir: node_feats_flat[:, :, None]}

            radial_embeddings = jax.vmap(
                lambda x: radial_basis_function(x, self.cutoff, self.num_radial_basis)
            )(jnp.linalg.norm(vecs, axis=1))

            Es = jnp.zeros((num_nodes,), dtype=vecs.dtype)

            for layer in self.layers:
                output, node_feats = layer(
                    vecs,
                    node_feats,
                    num_species_counts,
                    radial_embeddings,
                    senders,
                    receivers,
                    num_nodes,
                )
                scalar_ir = list(output.keys())[0]
                Es = Es + jnp.squeeze(output[scalar_ir], axis=(-1, -2))

            return jnp.sum(Es), Es

        Fterms, Ei = jax.grad(model, has_aux=True)(vecs)
        Fterms = Fterms * jnp.expand_dims(mask, -1)

        Ei = Ei + jnp.repeat(
            jnp.asarray(self.offsets, dtype=Ei.dtype),
            num_species_counts,
            axis=0,
            total_repeat_length=num_nodes,
        )

        E = jnp.zeros((num_graphs,), Ei.dtype).at[graph_index].add(Ei)
        F = (
            jnp.zeros((num_nodes, 3), Ei.dtype)
            .at[senders]
            .add(Fterms)
            .at[receivers]
            .add(-Fterms)[inv_perm]
        )

        return E, F


def benchmark(
    model_size: str,
    num_atoms: int,
    num_edges: int,
    dtype: jnp.dtype,
    mode: str = "both",
):
    import time

    import optax

    assert model_size in ["MP-S", "MP-M", "MP-L", "OFF-S", "OFF-M", "OFF-L"]
    assert mode in ["train", "inference", "both"]
    dtype = jnp.dtype(dtype)

    num_species = 50
    num_graphs = 100
    avg_num_neighbors = 20

    rngs = nnx.Rngs(0)
    model = MACEModel(
        num_layers=2,
        num_features={
            "MP-S": 128,
            "MP-M": 128,
            "MP-L": 128,
            "OFF-S": 64 + 32,
            "OFF-M": 128,
            "OFF-L": 128 + 64,
        }[model_size],
        num_species=num_species,
        max_ell=3,
        correlation=3,
        num_radial_basis=8,
        interaction_irreps=cue.Irreps(cue.O3, "0e+1o+2e+3o"),
        hidden_irreps=cue.Irreps(
            cue.O3,
            {
                "MP-S": "0e",
                "MP-M": "0e+1o",
                "MP-L": "0e+1o+2e",
                "OFF-S": "0e",
                "OFF-M": "0e+1o",
                "OFF-L": "0e+1o+2e",
            }[model_size],
        ),
        offsets=np.zeros(num_species),
        cutoff=5.0,
        epsilon=1 / avg_num_neighbors,
        skip_connection_first_layer=("MP" in model_size),
        dtype=dtype,
        rngs=rngs,
    )

    vecs = jax.random.normal(jax.random.key(0), (num_edges, 3), dtype)
    species = jax.random.randint(
        jax.random.key(0), (num_atoms,), 0, num_species, jnp.int32
    )
    senders, receivers = jax.random.randint(
        jax.random.key(0), (2, num_edges), 0, num_atoms, jnp.int32
    )
    graph_index = jax.random.randint(
        jax.random.key(0), (num_atoms,), 0, num_graphs, jnp.int32
    )
    graph_index = jnp.sort(graph_index)
    target_E = jax.random.normal(jax.random.key(0), (num_graphs,), dtype)
    target_F = jax.random.normal(jax.random.key(0), (num_atoms, 3), dtype)
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

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))

    @nnx.jit
    def step(model, opt_state, batch_dict, target_E, target_F):
        def loss_fn(model):
            E, F = model(batch_dict)
            return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

        grad = nnx.grad(loss_fn)(model)
        params = nnx.state(model, nnx.Param)
        grad_state = nnx.state(grad, nnx.Param)
        updates, opt_state_new = optimizer.update(grad_state, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        return opt_state_new

    @nnx.jit
    def inference(model, batch_dict):
        return model(batch_dict)

    runtime_per_training_step = 0
    runtime_per_inference = 0

    if mode in ["train", "both"]:
        opt_state = step(model, opt_state, batch_dict, target_E, target_F)
        jax.block_until_ready(opt_state)
        t0 = time.perf_counter()
        for _ in range(10):
            opt_state = step(model, opt_state, batch_dict, target_E, target_F)
        jax.block_until_ready(opt_state)
        runtime_per_training_step = 1e3 * (time.perf_counter() - t0) / 10

    if mode in ["inference", "both"]:
        out = inference(model, batch_dict)
        jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(10):
            out = inference(model, batch_dict)
        jax.block_until_ready(out)
        runtime_per_inference = 1e3 * (time.perf_counter() - t0) / 10

    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(
        f"MACE-NNX {model_size}: {num_atoms} atoms, {num_edges} edges, {dtype}, {num_params:,} params"
    )

    if mode == "both":
        print(
            f"train: {runtime_per_training_step:.1f}ms, inference: {runtime_per_inference:.1f}ms"
        )
    elif mode == "train":
        print(f"train: {runtime_per_training_step:.1f}ms")
    else:
        print(f"inference: {runtime_per_inference:.1f}ms")


def main():
    import argparse

    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["float32", "float64", "float16", "bfloat16"],
        default=["float32"],
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=["MP-S", "MP-M", "MP-L", "OFF-S", "OFF-M", "OFF-L"],
        default=["MP-S"],
    )
    parser.add_argument(
        "--mode", choices=["train", "inference", "both"], default="both"
    )
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--edges", type=int)
    args = parser.parse_args()

    defaults = {"MP": (3_000, 160_000), "OFF": (4_000, 70_000)}

    for dtype_str in args.dtype:
        for model_size in args.model:
            prefix = model_size.split("-")[0]
            num_atoms = args.nodes or defaults[prefix][0]
            num_edges = args.edges or defaults[prefix][1]
            benchmark(
                model_size, num_atoms, num_edges, getattr(jnp, dtype_str), args.mode
            )


if __name__ == "__main__":
    main()
