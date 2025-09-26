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
E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials

This implementation is based on the original NequIP paper:
Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N.,
Smidt, T. E., & Kozinsky, B. (2022). E(3)-Equivariant Graph Neural Networks for Data-Efficient
and Accurate Interatomic Potentials. Nature Communications, 13(1), 2453.
arXiv preprint arXiv:2101.03164. https://arxiv.org/abs/2101.03164
"""

import argparse
import ctypes
import time
from typing import Callable

import flax
import flax.linen
import jax
import jax.numpy as jnp
import optax
from cuequivariance_jax.experimental.utils import MultiLayerPerceptron, bessel

import cuequivariance as cue
import cuequivariance_jax as cuex


def polynomial_envelope(x, r_max):
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


def radial_basis_function(edge, r_max, num_radial_basis):
    """Radial basis function with polynomial cutoff."""
    cutoff = jnp.where(edge < r_max, polynomial_envelope(edge, r_max), 0.0)
    return bessel(edge, num_radial_basis, r_max) * cutoff[:, None]


class NEQUIPLayer(flax.linen.Module):
    normalization_factor: (
        float  # typically 1/avg_num_neighbors or 1/sqrt(avg_num_neighbors)
    )
    num_species: int = 1
    max_ell: int = 3
    output_irreps: cue.Irreps = 64 * cue.Irreps(cue.O3, "0e + 1o + 2e")
    even_activation: Callable[[jax.Array], jax.Array] = jax.nn.silu
    odd_activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    gate_activation: Callable[[jax.Array], jax.Array] = jax.nn.silu
    mlp_activation: Callable[[jax.Array], jax.Array] = jax.nn.silu
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    cutoff: float = 5.0
    n_radial_basis: int = 8

    @flax.linen.compact
    def __call__(
        self,
        vectors: cuex.RepArray,
        node_feats: cuex.RepArray,
        node_specie: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ):
        dtype = node_feats.dtype
        num_nodes = node_feats.shape[0]
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert node_feats.shape == (num_nodes, node_feats.irreps.dim)
        assert node_specie.shape == (num_nodes,)
        assert senders.shape == (num_edges,)
        assert receivers.shape == (num_edges,)
        assert self.output_irreps == self.output_irreps.regroup(), (
            f"{self.output_irreps} != {self.output_irreps.regroup()}"
        )

        e = cue.descriptors.linear(node_feats.irreps, node_feats.irreps)
        w = self.param("linear_up", jax.random.normal, (e.inputs[0].dim,), dtype)
        x = cuex.equivariant_polynomial(e, [w, node_feats], method="naive")

        y = cuex.spherical_harmonics(range(self.max_ell + 1), vectors)

        e = cue.descriptors.channelwise_tensor_product(
            x.irreps, y.irreps, self.output_irreps + "0e", simplify_irreps3=True
        )
        assert isinstance(self.normalization_factor, float)
        e = e * self.normalization_factor

        # Radial part
        with jax.ensure_compile_time_eval():
            assert abs(self.mlp_activation(0.0)) < 1e-6

        lengths = cuex.norm(vectors).array.squeeze(1)
        assert lengths.shape == (num_edges,)

        mix = MultiLayerPerceptron(
            self.mlp_n_layers * (self.mlp_n_hidden,) + (e.inputs[0].dim,),
            self.mlp_activation,
            output_activation=False,
            with_bias=False,
        )(radial_basis_function(lengths, self.cutoff, self.n_radial_basis))

        # Discard 0 length edges that come from graph padding
        mix = jnp.where(lengths[:, None] == 0.0, 0.0, mix)
        assert mix.shape == (num_edges, e.inputs[0].dim)

        [z] = cuex.equivariant_polynomial(
            e,
            [mix, x, y],
            [jax.ShapeDtypeStruct((num_nodes, e.outputs[0].dim), dtype)],
            [None, senders, None, receivers],
            method="uniform_1d",
        )

        irreps = self.output_irreps.filter(keep=z.irreps)
        num_nonscalar = irreps.filter(drop="0e + 0o").num_irreps
        if num_nonscalar > 0:
            irreps = irreps + num_nonscalar * cue.Irreps(cue.O3, "0e")

        e = cue.descriptors.linear(node_feats.irreps, irreps)
        w = self.param(
            "linear_skip", jax.random.normal, (self.num_species, e.inputs[0].dim), dtype
        )
        skip = cuex.equivariant_polynomial(
            e,
            [w, node_feats],
            jax.ShapeDtypeStruct((num_nodes, e.outputs[0].dim), dtype),
            [node_specie, None, None],
            method="naive",
        )

        e = cue.descriptors.linear(z.irreps, irreps)
        w = self.param("linear_down", jax.random.normal, (e.inputs[0].dim,), dtype)
        node_feats = cuex.equivariant_polynomial(e, [w, z], method="naive")

        node_feats = node_feats + skip
        assert node_feats.shape == (num_nodes, node_feats.irreps.dim)

        with cue.assume(cue.O3, cue.ir_mul):
            if num_nonscalar > 0:
                g = node_feats.slice_by_mul[-num_nonscalar:]
                x = node_feats.slice_by_mul[:-num_nonscalar]
                s = x.filter(keep="0e + 0o")
                v = x.filter(drop="0e + 0o")
                g = cuex.scalar_activation(g, self.gate_activation)

                e = cue.descriptors.elementwise_tensor_product(g.irreps, v.irreps)
                v = cuex.equivariant_polynomial(e, [g, v], method="naive")

                node_feats = cuex.concatenate([s, v])

            node_feats = cuex.scalar_activation(
                node_feats,
                {"0e": self.even_activation, "0o": self.odd_activation},
                normalize_act=True,
            )
        return node_feats


class NEQUIPModel(flax.linen.Module):
    num_species: int
    cutoff: float
    num_layers: int
    num_features: int
    max_ell: int
    normalization_factor: float

    @flax.linen.compact
    def __call__(self, batch):
        vecs, species, senders, receivers, graph_index, mask = (
            batch["nn_vecs"],
            batch["species"],
            batch["inda"],
            batch["indb"],
            batch["inde"],
            batch["mask"],
        )
        num_graphs = jnp.shape(batch["nats"])[0]

        # Sort atoms by species
        perm = jnp.argsort(species)
        species, graph_index = species[perm], graph_index[perm]
        inv_perm = jnp.zeros_like(perm).at[perm].set(jnp.arange(perm.shape[0]))
        senders, receivers = inv_perm[senders], inv_perm[receivers]
        num_species = jnp.zeros((self.num_species,), dtype=jnp.int32).at[species].add(1)
        nats = jnp.shape(species)[0]

        def model(vecs):
            with cue.assume(cue.O3, cue.ir_mul):
                # Initial node embeddings
                w = self.param(
                    "linear_embedding",
                    jax.random.normal,
                    (self.num_species, self.num_features),
                    vecs.dtype,
                )
                node_feats = cuex.as_irreps_array(
                    jnp.repeat(w, num_species, axis=0, total_repeat_length=nats)
                ) / jnp.sqrt(self.num_species)

                vecs = cuex.RepArray("1o", vecs)

                for i in range(self.num_layers):
                    node_feats = NEQUIPLayer(
                        normalization_factor=self.normalization_factor,
                        num_species=self.num_species,
                        max_ell=self.max_ell,
                        output_irreps=self.num_features
                        * cue.Irreps(cue.O3, "0e + 1o + 2e + 3o"),
                        cutoff=self.cutoff,
                        name=f"layer_{i}",
                    )(vecs, node_feats, species, senders, receivers)

                # Readout layer for energy
                e = cue.descriptors.linear(
                    node_feats.irreps, cue.Irreps(cue.O3, "1x0e")
                )
                w = self.param(
                    "energy_readout", jax.random.normal, (e.inputs[0].dim,), vecs.dtype
                )
                energy_per_atom = cuex.equivariant_polynomial(
                    e, [w, node_feats], method="naive"
                )
                energy_per_atom = jnp.squeeze(energy_per_atom.array, 1)

                return jnp.sum(energy_per_atom), energy_per_atom

        Fterms, Ei = jax.grad(model, has_aux=True)(vecs)
        Fterms *= jnp.expand_dims(mask, -1)
        E = jnp.zeros((num_graphs,), Ei.dtype).at[graph_index].add(Ei)
        F = (
            jnp.zeros((nats, 3), Ei.dtype)
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
    assert model_size in ["S", "M", "L"]
    assert mode in ["train", "inference", "both"]
    dtype = jnp.dtype(dtype)

    num_species = 50
    num_graphs = 100
    avg_num_neighbors = 20

    model = NEQUIPModel(
        num_layers=3,
        num_features={
            "S": 64,
            "M": 128,
            "L": 256,
        }[model_size],
        num_species=num_species,
        max_ell=3,
        cutoff=5.0,
        normalization_factor=1 / avg_num_neighbors,
    )

    # Dummy data
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

    # Initialize
    w = jax.jit(model.init)(jax.random.key(0), batch_dict)
    opt_state = jax.jit(optax.adam(1e-2).init)(w)

    @jax.jit
    def step(w, opt_state, batch_dict, target_E, target_F):
        def loss_fn(w):
            E, F = model.apply(w, batch_dict)
            return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

        grad = jax.grad(loss_fn)(w)
        updates, opt_state = optax.adam(1e-2).update(grad, opt_state)
        return optax.apply_updates(w, updates), opt_state

    @jax.jit
    def inference(w, batch_dict):
        return model.apply(w, batch_dict)

    runtime_per_training_step = 0
    runtime_per_inference = 0

    if mode in ["train", "both"]:
        jax.block_until_ready(
            step(w, opt_state, batch_dict, target_E, target_F)
        )  # compile
        t0 = time.perf_counter()
        for _ in range(10):
            w, opt_state = step(w, opt_state, batch_dict, target_E, target_F)
        jax.block_until_ready(w)
        runtime_per_training_step = 1e3 * (time.perf_counter() - t0) / 10

    if mode in ["inference", "both"]:
        jax.block_until_ready(inference(w, batch_dict))  # compile
        t0 = time.perf_counter()
        for _ in range(10):
            out = inference(w, batch_dict)
        jax.block_until_ready(out)
        runtime_per_inference = 1e3 * (time.perf_counter() - t0) / 10

    # Profile and print results
    num_params = sum(x.size for x in jax.tree.leaves(w))
    print(
        f"NEQUIP {model_size}: {num_atoms} atoms, {num_edges} edges, {dtype}, {num_params:,} params"
    )

    if mode == "both":
        print(
            f"train: {runtime_per_training_step:.1f}ms, inference: {runtime_per_inference:.1f}ms"
        )
    elif mode == "train":
        print(f"train: {runtime_per_training_step:.1f}ms")
    else:
        print(f"inference: {runtime_per_inference:.1f}ms")

    try:
        cuda = ctypes.CDLL("libcudart.so")
        cuda.cudaProfilerStart()
        if mode in ["train", "both"]:
            jax.block_until_ready(step(w, opt_state, batch_dict, target_E, target_F))
        if mode in ["inference", "both"]:
            jax.block_until_ready(inference(w, batch_dict))
        cuda.cudaProfilerStop()
    except Exception:
        pass


def main():
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
        choices=["S", "M", "L"],
        default=["S"],
    )
    parser.add_argument(
        "--mode", choices=["train", "inference", "both"], default="both"
    )
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--edges", type=int)
    args = parser.parse_args()

    defaults = {"S": (1_000, 40_000), "M": (2_000, 80_000), "L": (3_000, 120_000)}

    for dtype in args.dtype:
        for model_size in args.model:
            num_atoms = args.nodes or defaults[model_size][0]
            num_edges = args.edges or defaults[model_size][1]
            benchmark(model_size, num_atoms, num_edges, getattr(jnp, dtype), args.mode)


if __name__ == "__main__":
    main()
