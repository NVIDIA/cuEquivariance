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
"""MACE model implementation using Flax NNX and dict[Irrep, Array] representation.

A simplified implementation of MACE (Higher Order Equivariant Message Passing Neural
Networks for Fast and Accurate Force Fields).

Reference: Batatia et al. (2022) https://arxiv.org/abs/2206.07697
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


def radial_basis(r: Array, r_max: float, num_basis: int) -> Array:
    """Bessel radial basis with polynomial envelope cutoff."""
    p = 5
    u = r / r_max
    up = jnp.power(u, p)
    envelope = (
        1.0
        - 0.5 * (p + 1) * (p + 2) * up
        + p * (p + 2) * up * u
        - 0.5 * p * (p + 1) * up * u * u
    )
    envelope = jnp.where(r < r_max, envelope, 0.0)

    ns = jnp.arange(1, num_basis + 1, dtype=r.dtype)
    bessel = (
        jnp.sqrt(2.0 / r_max)
        * jnp.pi
        * ns
        / r_max
        * jnp.sinc(ns * r[..., None] / r_max)
    )
    return bessel * envelope[..., None]


def scalar_activation(x: dict[Irrep, Array], act: Callable) -> dict[Irrep, Array]:
    """Apply activation to scalar irreps only."""
    act = cuex.normalize_function(act)
    return {ir: act(v) if ir.is_scalar() else v for ir, v in x.items()}


class MessagePassing(nnx.Module):
    """Channelwise tensor product with gather/scatter for message passing."""

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_sh: cue.Irreps,
        irreps_out: cue.Irreps,
        epsilon: float,
        dtype: Any,
        rngs: nnx.Rngs,
    ):
        e = (
            cue.descriptors.channelwise_tensor_product(
                irreps_in, irreps_sh, irreps_out, True
            )
            * epsilon
        )
        self.weight_numel = e.inputs[0].dim
        self.irreps_out = e.outputs[0].irreps

        self.poly = (
            e.split_operand_by_irrep(2)
            .split_operand_by_irrep(1)
            .split_operand_by_irrep(-1)
            .polynomial
        )

    def __call__(
        self,
        weights: Array,
        node_feats: dict[Irrep, Array],
        sph: dict[Irrep, Array],
        senders: Array,
        receivers: Array,
        num_nodes: int,
    ) -> dict[Irrep, Array]:
        w = rearrange(weights, "e (s m) -> e s m", s=self.poly.inputs[0].num_segments)
        x1 = jax.tree.map(lambda v: rearrange(v, "n m i -> n i m"), node_feats)
        x2 = jax.tree.map(lambda v: rearrange(v, "e 1 i -> e i"), sph)

        out_template = {
            ir: jax.ShapeDtypeStruct(
                (num_nodes, desc.num_segments) + desc.segment_shape, w.dtype
            )
            for (_, ir), desc in zip(self.irreps_out, self.poly.outputs)
        }
        y = cuex.ir_dict.segmented_polynomial_uniform_1d(
            self.poly,
            [w, x1, x2],
            out_template,
            input_indices=[None, senders, None],
            output_indices=receivers,
        )
        return {
            ir: rearrange(v, "n (i s) m -> n (s m) i", i=ir.dim) for ir, v in y.items()
        }


class SymmetricContraction(nnx.Module):
    """Contract higher-order features using symmetric contraction."""

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        correlation: int,
        num_species: int,
        num_features: int,
        dtype: Any,
        rngs: nnx.Rngs,
    ):
        self.num_species = num_species
        self.irreps_out = irreps_out

        e, projection = mace_symmetric_contraction(
            irreps_in, irreps_out, range(1, correlation + 1)
        )
        self.projection = jnp.array(projection, dtype=dtype)

        self.poly = e.split_operand_by_irrep(1).split_operand_by_irrep(-1).polynomial
        self.w = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (num_species, self.projection.shape[0], num_features),
                dtype,
            )
        )

    def __call__(
        self, x: dict[Irrep, Array], species_counts: Array
    ) -> dict[Irrep, Array]:
        w = jnp.einsum("zau,ab->zbu", self.w[...], self.projection)
        num_nodes = jax.tree.leaves(x)[0].shape[0]
        species_idx = jnp.repeat(
            jnp.arange(self.num_species), species_counts, total_repeat_length=num_nodes
        )

        x = jax.tree.map(lambda v: rearrange(v, "n m i -> n i m"), x)
        out_template = cuex.ir_dict.mul_ir_dict(self.irreps_out, None)
        y = cuex.ir_dict.segmented_polynomial_uniform_1d(
            self.poly, [w, x], out_template, input_indices=[species_idx, None]
        )
        return jax.tree.map(lambda v: rearrange(v, "n i m -> n m i"), y)


class MACELayer(nnx.Module):
    """Single MACE interaction layer."""

    def __init__(
        self,
        input_irreps: cue.Irreps,
        interaction_irreps: cue.Irreps,
        hidden_irreps: cue.Irreps,
        output_irreps: cue.Irreps,
        num_species: int,
        num_features: int,
        max_ell: int,
        correlation: int,
        radial_dim: int,
        epsilon: float,
        has_skip: bool,
        has_linZ_first: bool,
        is_last: bool,
        dtype: Any,
        rngs: nnx.Rngs,
    ):
        self.is_last = is_last

        hidden_out = (
            hidden_irreps.filter(keep=output_irreps) if is_last else hidden_irreps
        )
        sph_irreps = cue.Irreps(
            cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in range(max_ell + 1)]
        )

        self.sph = SphericalHarmonics(max_ell)
        self.linear_up = IrrepsLinear(
            input_irreps, input_irreps, dtype=dtype, rngs=rngs
        )
        self.message = MessagePassing(
            input_irreps,
            sph_irreps,
            num_features * interaction_irreps,
            epsilon,
            dtype,
            rngs,
        )
        self.radial_mlp = MLP(
            [radial_dim, 64, 64, 64, self.message.weight_numel],
            jax.nn.silu,
            False,
            dtype=dtype,
            rngs=rngs,
        )
        self.linear_down = IrrepsLinear(
            self.message.irreps_out,
            num_features * interaction_irreps,
            dtype=dtype,
            rngs=rngs,
        )

        self.skip = (
            IrrepsIndexedLinear(
                input_irreps,
                num_features * hidden_out,
                num_species,
                dtype=dtype,
                rngs=rngs,
            )
            if has_skip
            else None
        )
        self.linZ_first = (
            IrrepsIndexedLinear(
                num_features * interaction_irreps,
                num_features * interaction_irreps,
                num_species,
                dtype=dtype,
                rngs=rngs,
            )
            if has_linZ_first
            else None
        )
        self.sc = SymmetricContraction(
            num_features * interaction_irreps,
            num_features * hidden_out,
            correlation,
            num_species,
            num_features,
            dtype,
            rngs,
        )
        self.linear_sc = IrrepsLinear(
            num_features * hidden_out, num_features * hidden_out, dtype=dtype, rngs=rngs
        )

        if is_last:
            readout_irreps = cue.Irreps(cue.O3, "16x0e")
            self.readout_mlp = IrrepsLinear(
                num_features * hidden_out, readout_irreps, dtype=dtype, rngs=rngs
            )
            self.readout = IrrepsLinear(
                readout_irreps, output_irreps, dtype=dtype, rngs=rngs
            )
        else:
            self.readout = IrrepsLinear(
                num_features * hidden_out, output_irreps, dtype=dtype, rngs=rngs
            )

    def __call__(
        self,
        vectors: Array,
        node_feats: dict[Irrep, Array],
        species_counts: Array,
        radial_embed: Array,
        senders: Array,
        receivers: Array,
        num_nodes: int,
    ) -> tuple[dict[Irrep, Array], dict[Irrep, Array]]:
        sph = self.sph(vectors)
        skip = self.skip(node_feats, species_counts) if self.skip else None

        h = self.linear_up(node_feats)
        h = self.message(
            self.radial_mlp(radial_embed), h, sph, senders, receivers, num_nodes
        )
        h = self.linear_down(h)
        if self.linZ_first is not None:
            h = self.linZ_first(h, species_counts)
        h = self.sc(h, species_counts)
        h = self.linear_sc(h)

        if skip is not None:
            h = cuex.ir_dict.irreps_add(h, skip)

        out = h
        if self.is_last:
            out = scalar_activation(self.readout_mlp(out), jax.nn.silu)
        out = self.readout(out)

        return out, h


class MACEModel(nnx.Module):
    """MACE model for molecular property prediction."""

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
        self.num_radial_basis = num_radial_basis

        output_irreps = cue.Irreps(cue.O3, "1x0e")
        full_hidden = hidden_irreps.set_mul(num_features)

        self.embedding = nnx.Param(
            jax.random.normal(rngs.params(), (num_species, num_features), dtype)
        )

        layers = []
        for i in range(num_layers):
            is_first, is_last = (i == 0), (i == num_layers - 1)
            input_irreps = full_hidden.filter(keep="0e") if is_first else full_hidden
            has_skip = not is_first or skip_connection_first_layer
            has_linZ_first = is_first and not skip_connection_first_layer

            layers.append(
                MACELayer(
                    input_irreps=input_irreps,
                    interaction_irreps=interaction_irreps,
                    hidden_irreps=hidden_irreps,
                    output_irreps=output_irreps,
                    num_species=num_species,
                    num_features=num_features,
                    max_ell=max_ell,
                    correlation=correlation,
                    radial_dim=num_radial_basis,
                    epsilon=epsilon,
                    has_skip=has_skip,
                    has_linZ_first=has_linZ_first,
                    is_last=is_last,
                    dtype=dtype,
                    rngs=rngs,
                )
            )
        self.layers = nnx.List(layers)

    def __call__(self, batch: dict[str, Array]) -> tuple[Array, Array]:
        vecs = batch["nn_vecs"]
        species = batch["species"]
        senders, receivers = batch["inda"], batch["indb"]
        graph_idx = batch["inde"]
        mask = batch["mask"]
        num_graphs = batch["nats"].shape[0]

        perm = jnp.argsort(species)
        species = species[perm]
        graph_idx = graph_idx[perm]
        inv_perm = jnp.zeros_like(perm).at[perm].set(jnp.arange(perm.shape[0]))
        senders, receivers = inv_perm[senders], inv_perm[receivers]

        num_nodes = species.shape[0]
        species_counts = jnp.zeros((self.num_species,), jnp.int32).at[species].add(1)

        def forward(vecs: Array) -> tuple[Array, Array]:
            node_feats = jnp.repeat(
                self.embedding[...],
                species_counts,
                axis=0,
                total_repeat_length=num_nodes,
            )
            node_feats = {
                cue.O3(0, 1): (node_feats / jnp.sqrt(self.num_species))[:, :, None]
            }

            radial_embed = jax.vmap(
                lambda r: radial_basis(
                    jnp.linalg.norm(r), self.cutoff, self.num_radial_basis
                )
            )(vecs)
            energies = jnp.zeros(num_nodes, vecs.dtype)

            for layer in self.layers:
                out, node_feats = layer(
                    vecs,
                    node_feats,
                    species_counts,
                    radial_embed,
                    senders,
                    receivers,
                    num_nodes,
                )
                energies = energies + jnp.squeeze(out[list(out.keys())[0]], (-1, -2))

            return jnp.sum(energies), energies

        forces, atom_energies = jax.grad(forward, has_aux=True)(vecs)
        forces = forces * mask[:, None]

        atom_energies = atom_energies + jnp.repeat(
            jnp.asarray(self.offsets, atom_energies.dtype),
            species_counts,
            total_repeat_length=num_nodes,
        )

        E = jnp.zeros(num_graphs, atom_energies.dtype).at[graph_idx].add(atom_energies)
        F = (
            jnp.zeros((num_nodes, 3), atom_energies.dtype)
            .at[senders]
            .add(forces)
            .at[receivers]
            .add(-forces)[inv_perm]
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

    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, batch_dict, target_E, target_F):
        def loss_fn(model):
            E, F = model(batch_dict)
            return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

        grads = nnx.grad(loss_fn)(model)
        optimizer.update(model, grads)

    @nnx.jit
    def inference(model, batch_dict):
        return model(batch_dict)

    runtime_per_training_step = 0
    runtime_per_inference = 0

    if mode in ["train", "both"]:
        step(model, optimizer, batch_dict, target_E, target_F)
        jax.block_until_ready(nnx.state(model))
        t0 = time.perf_counter()
        for _ in range(10):
            step(model, optimizer, batch_dict, target_E, target_F)
        jax.block_until_ready(nnx.state(model))
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
