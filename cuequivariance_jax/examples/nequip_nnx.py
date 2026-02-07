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
"""NequIP model implementation using Flax NNX and dict[Irrep, Array] representation.

E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials.

Reference: Batzner et al. (2022) https://arxiv.org/abs/2101.03164
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from cuequivariance_jax.nnx import MLP, IrrepsLinear, SphericalHarmonics
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


def scalar_activation(
    x: dict[Irrep, Array], even_act: Callable, odd_act: Callable | None = None
) -> dict[Irrep, Array]:
    """Apply activation to scalar irreps only."""
    even_act = cuex.normalize_function(even_act)
    odd_act = cuex.normalize_function(odd_act) if odd_act else None

    def apply(ir: Irrep, v: Array) -> Array:
        if ir.is_scalar():
            return even_act(v) if ir.p == 1 else (odd_act(v) if odd_act else v)
        return v

    return {ir: apply(ir, v) for ir, v in x.items()}


class GatedLinear(nnx.Module):
    """Linear layer outputting to gated irreps structure (scalars + nonscalars + gates).

    Stores weights in Linen-compatible layout for weight transfer tests.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        num_scalars: int,
        num_gates: int,
        irreps_nonscalar: cue.Irreps,
        *,
        dtype: Any,
        rngs: nnx.Rngs,
    ):
        in_0e = sum(m for m, ir in irreps_in if ir == cue.O3(0, 1))
        self.w_0e_scalar = nnx.Param(
            jax.random.normal(rngs.params(), (in_0e, num_scalars), dtype)
        )
        self.w_0e_gates = nnx.Param(
            jax.random.normal(rngs.params(), (in_0e, num_gates), dtype)
        )

        w_nonscalar = {}
        for mul, ir in irreps_nonscalar:
            in_mul = sum(m for m, i in irreps_in if i == ir)
            if in_mul > 0:
                w_nonscalar[str(ir)] = nnx.Param(
                    jax.random.normal(rngs.params(), (in_mul, mul), dtype)
                )
        self.w_nonscalar = nnx.Dict(w_nonscalar)
        self.irreps_nonscalar = irreps_nonscalar

    def __call__(self, x: dict[Irrep, Array]) -> dict[Irrep, Array]:
        result = {}
        ir_0e = cue.O3(0, 1)
        if ir_0e in x:
            x_0e = x[ir_0e][..., 0]  # (batch, in_mul)
            norm = jnp.sqrt(x_0e.shape[-1])
            scalars = (x_0e @ self.w_0e_scalar[...]) / norm
            gates = (x_0e @ self.w_0e_gates[...]) / norm
            result[ir_0e] = jnp.concatenate([scalars, gates], axis=-1)[..., None]

        w_keys = set(self.w_nonscalar.keys())
        for mul, ir in self.irreps_nonscalar:
            if str(ir) in w_keys and ir in x:
                x_ir = x[ir]
                w = self.w_nonscalar[str(ir)][...]
                result[ir] = jnp.einsum("bni,no->boi", x_ir, w) / jnp.sqrt(
                    x_ir.shape[1]
                )
        return result


class GatedIndexedLinear(nnx.Module):
    """Indexed linear layer outputting to gated irreps structure.

    Used for skip connections indexed by species.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        num_scalars: int,
        num_gates: int,
        num_indices: int,
        irreps_nonscalar: cue.Irreps,
        *,
        dtype: Any,
        rngs: nnx.Rngs,
    ):
        self.num_scalars = num_scalars
        self.num_gates = num_gates
        self.num_indices = num_indices
        self.irreps_nonscalar = irreps_nonscalar
        self.dtype = dtype

        in_0e = sum(m for m, ir in irreps_in if ir == cue.O3(0, 1))
        self.w_0e_scalar = nnx.Param(
            jax.random.normal(rngs.params(), (num_indices, in_0e * num_scalars), dtype)
        )
        self.w_0e_gates = nnx.Param(
            jax.random.normal(rngs.params(), (num_indices, in_0e * num_gates), dtype)
        )

        w_nonscalar = {}
        for mul_out, ir in irreps_nonscalar:
            in_mul = sum(m for m, i in irreps_in if i == ir)
            if in_mul > 0:
                w_nonscalar[str(ir)] = nnx.Param(
                    jax.random.normal(
                        rngs.params(), (num_indices, in_mul * mul_out), dtype
                    )
                )
        self.w_nonscalar = nnx.Dict(w_nonscalar)

    def __call__(
        self, x: dict[Irrep, Array], species_counts: Array
    ) -> dict[Irrep, Array]:
        ir_0e = cue.O3(0, 1)
        if ir_0e not in x:
            return {}

        x_0e = x[ir_0e]
        batch_size, in_0e = x_0e.shape[0], x_0e.shape[1]
        x_0e_flat = x_0e[..., 0]

        def expand(w, in_mul, out_mul):
            return jnp.repeat(
                w.reshape(self.num_indices, in_mul, out_mul),
                species_counts,
                axis=0,
                total_repeat_length=batch_size,
            )

        norm_0e = jnp.sqrt(in_0e)
        w_s = expand(self.w_0e_scalar[...], in_0e, self.num_scalars)
        w_g = expand(self.w_0e_gates[...], in_0e, self.num_gates)
        scalars = jnp.einsum("bi,bio->bo", x_0e_flat, w_s) / norm_0e
        gates = jnp.einsum("bi,bio->bo", x_0e_flat, w_g) / norm_0e
        result = {ir_0e: jnp.concatenate([scalars, gates], axis=-1)[..., None]}

        w_keys = set(self.w_nonscalar.keys())
        for mul_out, ir in self.irreps_nonscalar:
            ir_key = str(ir)
            if ir_key in w_keys and ir in x:
                x_ir = x[ir]
                in_mul = x_ir.shape[1]
                w_ir = expand(self.w_nonscalar[ir_key][...], in_mul, mul_out)
                result[ir] = jnp.einsum("bid,bio->bod", x_ir, w_ir) / jnp.sqrt(in_mul)
            else:
                result[ir] = jnp.zeros((batch_size, mul_out, ir.dim), self.dtype)
        return result


class MessagePassing(nnx.Module):
    """Channelwise tensor product with gather/scatter for message passing."""

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_sh: cue.Irreps,
        irreps_out: cue.Irreps,
        epsilon: float,
        *,
        name: str,
        dtype: Any,
        rngs: nnx.Rngs,
    ):
        self.name = name
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
            name=self.name,
        )
        return {
            ir: rearrange(v, "n (i s) m -> n (s m) i", i=ir.dim) for ir, v in y.items()
        }


class NEQUIPLayer(nnx.Module):
    """Single NequIP interaction layer with gated nonlinearity."""

    def __init__(
        self,
        input_irreps: cue.Irreps,
        output_irreps: cue.Irreps,
        num_species: int,
        max_ell: int,
        radial_dim: int,
        cutoff: float,
        epsilon: float,
        mlp_hidden: int,
        mlp_layers: int,
        even_activation: Callable,
        odd_activation: Callable,
        gate_activation: Callable,
        mlp_activation: Callable,
        name: str,
        dtype: Any,
        rngs: nnx.Rngs,
    ):
        self.even_activation = even_activation
        self.odd_activation = odd_activation
        self.gate_activation = gate_activation
        self.cutoff = cutoff
        self.radial_dim = radial_dim

        sph_irreps = cue.Irreps(
            cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in range(max_ell + 1)]
        )
        filtered = output_irreps.filter(keep="0e+0o+1o+1e+2e+2o+3o+3e").regroup()
        irreps_nonscalar = filtered.filter(drop="0e+0o")
        self.num_scalars = sum(m for m, ir in filtered if ir == cue.O3(0, 1))
        self.num_nonscalar = irreps_nonscalar.num_irreps
        conv_out = (output_irreps + cue.Irreps(cue.O3, "0e")).regroup()

        self.sph = SphericalHarmonics(max_ell)
        self.linear_up = IrrepsLinear(
            input_irreps, input_irreps, dtype=dtype, rngs=rngs
        )
        self.message = MessagePassing(
            input_irreps,
            sph_irreps,
            conv_out,
            epsilon,
            name=f"{name}_tensor_product",
            dtype=dtype,
            rngs=rngs,
        )
        self.radial_mlp = MLP(
            [radial_dim] + [mlp_hidden] * mlp_layers + [self.message.weight_numel],
            mlp_activation,
            False,
            dtype=dtype,
            rngs=rngs,
        )
        self.linear_down = GatedLinear(
            self.message.irreps_out,
            self.num_scalars,
            self.num_nonscalar,
            irreps_nonscalar,
            dtype=dtype,
            rngs=rngs,
        )
        self.linear_skip = GatedIndexedLinear(
            input_irreps,
            self.num_scalars,
            self.num_nonscalar,
            num_species,
            irreps_nonscalar,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        vectors: Array,
        node_feats: dict[Irrep, Array],
        species_counts: Array,
        senders: Array,
        receivers: Array,
        num_nodes: int,
    ) -> dict[Irrep, Array]:
        sph = self.sph(vectors)
        h = self.linear_up(node_feats)

        lengths = jnp.linalg.norm(vectors, axis=-1)
        radial_weights = self.radial_mlp(
            radial_basis(lengths, self.cutoff, self.radial_dim)
        )
        radial_weights = jnp.where(lengths[:, None] == 0.0, 0.0, radial_weights)

        h = self.message(radial_weights, h, sph, senders, receivers, num_nodes)
        h = self.linear_down(h)
        skip = self.linear_skip(node_feats, species_counts)
        h = cuex.ir_dict.irreps_add(h, skip)

        # Gated activation: split 0e into scalars and gates
        if self.num_nonscalar > 0:
            ir_0e = cue.O3(0, 1)
            h_0e = h[ir_0e]
            scalars, gates = (
                h_0e[:, : self.num_scalars, :],
                h_0e[:, self.num_scalars :, :],
            )
            gates = cuex.normalize_function(self.gate_activation)(gates)

            gate_idx, gated = 0, {ir_0e: scalars} if self.num_scalars > 0 else {}
            for ir, v in h.items():
                if not ir.is_scalar():
                    mul = v.shape[1]
                    gated[ir] = v * gates[:, gate_idx : gate_idx + mul, :]
                    gate_idx += mul
            h = gated

        return scalar_activation(h, self.even_activation, self.odd_activation)


class NEQUIPModel(nnx.Module):
    """NequIP model for molecular property prediction."""

    def __init__(
        self,
        num_species: int,
        cutoff: float,
        num_layers: int,
        num_features: int,
        max_ell: int,
        normalization_factor: float,
        *,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_species = num_species
        self.cutoff = cutoff

        output_irreps = num_features * cue.Irreps(
            cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in range(max_ell + 1)]
        )
        self.embedding = nnx.Param(
            jax.random.normal(rngs.params(), (num_species, num_features), dtype)
        )

        layers = []
        for i in range(num_layers):
            layers.append(
                NEQUIPLayer(
                    input_irreps=output_irreps.filter(keep="0e")
                    if i == 0
                    else output_irreps,
                    output_irreps=output_irreps,
                    num_species=num_species,
                    max_ell=max_ell,
                    radial_dim=8,
                    cutoff=cutoff,
                    epsilon=normalization_factor,
                    mlp_hidden=64,
                    mlp_layers=2,
                    even_activation=jax.nn.silu,
                    odd_activation=jax.nn.tanh,
                    gate_activation=jax.nn.silu,
                    mlp_activation=jax.nn.silu,
                    name=f"layer_{i}",
                    dtype=dtype,
                    rngs=rngs,
                )
            )
        self.layers = nnx.List(layers)
        self.readout = IrrepsLinear(
            output_irreps, cue.Irreps(cue.O3, "1x0e"), dtype=dtype, rngs=rngs
        )

    def __call__(self, batch: dict[str, Array]) -> tuple[Array, Array]:
        vecs, species, senders, receivers = (
            batch["nn_vecs"],
            batch["species"],
            batch["inda"],
            batch["indb"],
        )
        graph_idx, mask, num_graphs = (
            batch["inde"],
            batch["mask"],
            batch["nats"].shape[0],
        )

        perm = jnp.argsort(species)
        species, graph_idx = species[perm], graph_idx[perm]
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

            for layer in self.layers:
                node_feats = layer(
                    vecs, node_feats, species_counts, senders, receivers, num_nodes
                )

            out = self.readout(node_feats)
            energies = jnp.squeeze(out[list(out.keys())[0]], (-1, -2))
            return jnp.sum(energies), energies

        forces, atom_energies = jax.grad(forward, has_aux=True)(vecs)
        forces = forces * mask[:, None]

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

    assert model_size in ["S", "M", "L"] and mode in ["train", "inference", "both"]
    dtype = jnp.dtype(dtype)
    num_species, num_graphs, avg_num_neighbors = 50, 100, 20

    rngs = nnx.Rngs(0)
    model = NEQUIPModel(
        num_layers=3,
        num_features={"S": 64, "M": 128, "L": 256}[model_size],
        num_species=num_species,
        max_ell=3,
        cutoff=5.0,
        normalization_factor=1 / avg_num_neighbors,
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
    graph_index = jnp.sort(
        jax.random.randint(jax.random.key(0), (num_atoms,), 0, num_graphs, jnp.int32)
    )
    target_E = jax.random.normal(jax.random.key(0), (num_graphs,), dtype)
    target_F = jax.random.normal(jax.random.key(0), (num_atoms, 3), dtype)
    nats = jnp.zeros((num_graphs,), jnp.int32).at[graph_index].add(1)

    batch_dict = dict(
        nn_vecs=vecs,
        species=species,
        inda=senders,
        indb=receivers,
        inde=graph_index,
        nats=nats,
        mask=jnp.ones((num_edges,), jnp.int32),
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, batch_dict, target_E, target_F):
        grads = nnx.grad(
            lambda m: jnp.mean((m(batch_dict)[0] - target_E) ** 2)
            + jnp.mean((m(batch_dict)[1] - target_F) ** 2)
        )(model)
        optimizer.update(model, grads)

    @nnx.jit
    def inference(model, batch_dict):
        return model(batch_dict)

    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(
        f"NEQUIP-NNX {model_size}: {num_atoms} atoms, {num_edges} edges, {dtype}, {num_params:,} params",
        flush=True,
    )

    jit_train, jit_inf, train_ms, inf_ms = 0, 0, 0, 0
    if mode in ["train", "both"]:
        t0 = time.perf_counter()
        step(model, optimizer, batch_dict, target_E, target_F)
        jax.block_until_ready(nnx.state(model))
        jit_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(10):
            step(model, optimizer, batch_dict, target_E, target_F)
        jax.block_until_ready(nnx.state(model))
        train_ms = 1e3 * (time.perf_counter() - t0) / 10

    if mode in ["inference", "both"]:
        t0 = time.perf_counter()
        out = inference(model, batch_dict)
        jax.block_until_ready(out)
        jit_inf = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(10):
            out = inference(model, batch_dict)
        jax.block_until_ready(out)
        inf_ms = 1e3 * (time.perf_counter() - t0) / 10

    times = {
        "both": f"train: {train_ms:.1f}ms, inference: {inf_ms:.1f}ms, compile: {jit_train:.1f}s + {jit_inf:.1f}s",
        "train": f"train: {train_ms:.1f}ms, compile: {jit_train:.1f}s",
        "inference": f"inference: {inf_ms:.1f}ms, compile: {jit_inf:.1f}s",
    }
    print(times[mode], flush=True)


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
    parser.add_argument("--model", nargs="+", choices=["S", "M", "L"], default=["S"])
    parser.add_argument(
        "--mode", choices=["train", "inference", "both"], default="both"
    )
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--edges", type=int)
    parser.add_argument(
        "--larger", action="store_true", help="Use larger benchmark sizes"
    )
    args = parser.parse_args()

    defaults = {"S": (1_000, 40_000), "M": (2_000, 80_000), "L": (3_000, 120_000)}
    defaults_larger = {
        "S": (4_000, 160_000),
        "M": (8_000, 320_000),
        "L": (12_000, 480_000),
    }

    for dtype_str in args.dtype:
        for model_size in args.model:
            size_defaults = defaults_larger if args.larger else defaults
            benchmark(
                model_size,
                args.nodes or size_defaults[model_size][0],
                args.edges or size_defaults[model_size][1],
                getattr(jnp, dtype_str),
                args.mode,
            )


if __name__ == "__main__":
    main()
