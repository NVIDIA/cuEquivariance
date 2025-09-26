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

This implementation is based on the original MACE paper:
Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022).
MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.
arXiv preprint arXiv:2206.07697. https://arxiv.org/abs/2206.07697
"""

import argparse
import ctypes
import time
from typing import Callable

import flax
import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import optax
from cuequivariance.group_theory.experimental.mace import symmetric_contraction
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
    return bessel(edge, num_radial_basis, r_max) * cutoff


class MACELayer(flax.linen.Module):
    first: bool
    last: bool
    num_species: int
    num_features: int  # typically 128
    interaction_irreps: cue.Irreps  # typically 0e+1o+2e+3o
    hidden_irreps: cue.Irreps  # typically 0e+1o
    activation: Callable  # typically silu
    epsilon: float  # typically 1/avg_num_neighbors
    max_ell: int  # typically 3
    correlation: int  # typically 3
    output_irreps: cue.Irreps  # typically 1x0e
    readout_mlp_irreps: cue.Irreps  # typically 16x0e
    skip_connection_first_layer: bool = False

    @flax.linen.compact
    def __call__(
        self,
        vectors: cuex.RepArray,  # [num_edges, 3]
        node_feats: cuex.RepArray,  # [num_nodes, irreps]
        num_species: jax.Array,  # [self.num_species] int number of atoms per species
        radial_embeddings: jax.Array,  # [num_edges, radial_embedding_dim]
        senders: jax.Array,  # [num_edges]
        receivers: jax.Array,  # [num_edges]
    ):
        dtype = node_feats.dtype
        hidden_out = (
            self.hidden_irreps.filter(keep=self.output_irreps)
            if self.last
            else self.hidden_irreps
        )

        def lin(irreps, input, name):
            e = cue.descriptors.linear(input.irreps, irreps)
            w = self.param(name, jax.random.normal, (e.inputs[0].dim,), dtype)
            return cuex.equivariant_polynomial(
                e, [w, input], name=f"{self.name}_{name}", method="naive"
            )

        def linZ(irreps, input, name):
            e = cue.descriptors.linear(input.irreps, irreps) * (
                1.0 / self.num_species**0.5
            )
            w = self.param(
                name, jax.random.normal, (self.num_species, e.inputs[0].dim), dtype
            )
            return cuex.equivariant_polynomial(
                e,
                [w, input],
                None,
                [cuex.Repeats(num_species), None, None],
                name=f"{self.name}_{name}",
                method="indexed_linear",
            )

        sph = cuex.spherical_harmonics(range(self.max_ell + 1), vectors)

        # Skip connection
        self_connection = (
            linZ(self.num_features * hidden_out, node_feats, "linZ_skip_tp")
            if (not self.first or self.skip_connection_first_layer)
            else None
        )

        # Message passing
        node_feats = lin(node_feats.irreps, node_feats, "linear_up")

        # Convolution
        e = cue.descriptors.channelwise_tensor_product(
            node_feats.irreps, sph.irreps, self.interaction_irreps, True
        )
        e = e * self.epsilon
        mix = MultiLayerPerceptron(
            [64, 64, 64, e.inputs[0].dim],
            self.activation,
            output_activation=False,
            with_bias=False,
        )(radial_embeddings)
        node_feats = cuex.equivariant_polynomial(
            e,
            [mix, node_feats, sph],
            jax.ShapeDtypeStruct((node_feats.shape[0], -1), dtype),
            indices=[None, senders, None, receivers],
            name=f"{self.name}TP",
            method="uniform_1d",
        )

        node_feats = lin(
            self.num_features * self.interaction_irreps, node_feats, "linear_down"
        )

        if self.first and not self.skip_connection_first_layer:
            node_feats = linZ(
                self.num_features * self.interaction_irreps,
                node_feats,
                "linZ_skip_tp_first",
            )

        # Symmetric contraction
        e, projection = symmetric_contraction(
            node_feats.irreps,
            self.num_features * hidden_out,
            range(1, self.correlation + 1),
        )
        projection = jnp.array(projection, dtype=dtype)
        n = projection.shape[0]
        w = self.param(
            "symmetric_contraction",
            jax.random.normal,
            (self.num_species, n, self.num_features),
            dtype,
        )
        w = jnp.einsum("zau,ab->zbu", w, projection)
        w = jnp.reshape(w, (self.num_species, -1))
        i = jnp.repeat(
            jnp.arange(len(num_species)),
            num_species,
            total_repeat_length=node_feats.shape[0],
        )
        node_feats = cuex.equivariant_polynomial(
            e,
            [w, node_feats],
            indices=[i, None, None],
            name=f"{self.name}SC",
            method="uniform_1d",
        )

        node_feats = lin(self.num_features * hidden_out, node_feats, "linear_post_sc")

        if self_connection is not None:
            node_feats = node_feats + self_connection

        node_outputs = node_feats
        if self.last:
            node_outputs = cuex.scalar_activation(
                lin(self.readout_mlp_irreps, node_outputs, "linear_mlp_readout"),
                self.activation,
            )
        node_outputs = lin(self.output_irreps, node_outputs, "linear_readout")

        return node_outputs, node_feats


class MACEModel(flax.linen.Module):
    offsets: np.ndarray
    num_species: int
    cutoff: float
    num_layers: int
    num_features: int
    interaction_irreps: cue.Irreps
    hidden_irreps: cue.Irreps
    max_ell: int
    correlation: int
    num_radial_basis: int
    epsilon: float
    skip_connection_first_layer: bool

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
                w = self.param(
                    "linear_embedding",
                    jax.random.normal,
                    (self.num_species, self.num_features),
                    vecs.dtype,
                )
                node_feats = cuex.as_irreps_array(
                    jnp.repeat(w, num_species, axis=0, total_repeat_length=nats)
                ) / jnp.sqrt(self.num_species)
                radial_embeddings = jax.vmap(
                    lambda x: radial_basis_function(
                        x, self.cutoff, self.num_radial_basis
                    )
                )(jnp.linalg.norm(vecs, axis=1))
                vecs = cuex.RepArray("1o", vecs)

                Es = 0
                for i in range(self.num_layers):
                    output, node_feats = MACELayer(
                        first=(i == 0),
                        last=(i == self.num_layers - 1),
                        num_species=self.num_species,
                        num_features=self.num_features,
                        interaction_irreps=self.interaction_irreps,
                        hidden_irreps=self.hidden_irreps,
                        activation=jax.nn.silu,
                        epsilon=self.epsilon,
                        max_ell=self.max_ell,
                        correlation=self.correlation,
                        output_irreps=cue.Irreps(cue.O3, "1x0e"),
                        readout_mlp_irreps=cue.Irreps(cue.O3, "16x0e"),
                        skip_connection_first_layer=self.skip_connection_first_layer,
                        name=f"layer_{i}",
                    )(
                        vecs,
                        node_feats,
                        num_species,
                        radial_embeddings,
                        senders,
                        receivers,
                    )
                    Es += jnp.squeeze(output.array, 1)
                return jnp.sum(Es), Es

        Fterms, Ei = jax.grad(model, has_aux=True)(vecs)
        Fterms *= jnp.expand_dims(mask, -1)
        Ei = Ei + jnp.repeat(
            jnp.asarray(self.offsets, dtype=Ei.dtype),
            num_species,
            axis=0,
            total_repeat_length=nats,
        )
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
    assert model_size in ["MP-S", "MP-M", "MP-L", "OFF-S", "OFF-M", "OFF-L"]
    assert mode in ["train", "inference", "both"]
    dtype = jnp.dtype(dtype)

    num_species = 50
    num_graphs = 100
    avg_num_neighbors = 20

    model = MACEModel(
        num_layers=2,
        num_features={
            "MP-S": 128,
            "MP-M": 128,
            "MP-L": 128,
            "OFF-S": 64 + 32,  # = 96
            "OFF-M": 128,
            "OFF-L": 128 + 64,  # = 192
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
        f"MACE {model_size}: {num_atoms} atoms, {num_edges} edges, {dtype}, {num_params:,} params"
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

    for dtype in args.dtype:
        for model_size in args.model:
            prefix = model_size.split("-")[0]
            num_atoms = args.nodes or defaults[prefix][0]
            num_edges = args.edges or defaults[prefix][1]
            benchmark(model_size, num_atoms, num_edges, getattr(jnp, dtype), args.mode)


if __name__ == "__main__":
    main()
