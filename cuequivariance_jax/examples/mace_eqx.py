"""MACE model implementation using Equinox and dict[Irrep, Array] representation.

A simplified implementation of MACE (Higher Order Equivariant Message Passing Neural
Networks for Fast and Accurate Force Fields).

Reference: Batatia et al. (2022) https://arxiv.org/abs/2206.07697
"""

import argparse
import ctypes
import math
import time
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction_ir_dict as mace_symmetric_contraction_ir_dict,
)
from einops import rearrange
from jax import Array

import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance import Irrep


def safe_norm(
    x: Array, eps: float = 0.0, axis: int = -1, keepdims: bool = False
) -> Array:
    squared_norm = jnp.sum(jnp.conj(x) * x, axis=axis, keepdims=keepdims) + eps**2
    if eps == 0.0:
        safe_squared_norm = jnp.where(squared_norm == 0.0, 1.0, squared_norm)
        norm = jnp.sqrt(safe_squared_norm)
        return jnp.where(squared_norm == 0.0, 0.0, norm)
    return jnp.sqrt(squared_norm)


def radial_basis(r: Array, r_max: float, num_basis: int, p: int = 5) -> Array:
    """Bessel radial basis with polynomial envelope cutoff."""
    u = r / r_max  # [...]
    up = jnp.power(u, p)  # [...]
    envelope = (
        1.0
        - 0.5 * (p + 1) * (p + 2) * up
        + p * (p + 2) * up * u
        - 0.5 * p * (p + 1) * up * u * u
    )
    envelope = jnp.where(r < r_max, envelope, 0.0)

    ns = jnp.arange(1, num_basis + 1, dtype=r.dtype)  # [num_basis]
    bessel = (
        jnp.sqrt(2.0 / r_max)
        * jnp.pi
        * ns
        / r_max
        * jnp.sinc(ns * r[..., None] / r_max)
    )  # [..., num_basis]
    return bessel * envelope[..., None]  # [..., num_basis]


def scalar_activation(x: dict[Irrep, Array], act: Callable) -> dict[Irrep, Array]:
    """Apply activation to scalar irreps only."""
    act = cuex.normalize_function(act)
    return {ir: act(v) if ir.is_scalar() else v for ir, v in x.items()}


class MLP(eqx.Module):
    linears: list[Array]
    activation: Callable = eqx.field(static=True)
    output_activation: bool = eqx.field(static=True)
    layer_sizes: tuple[int, ...] = eqx.field(static=True)
    precision: jax.lax.Precision | None = eqx.field(static=True)

    def __init__(
        self,
        layer_sizes: list[int] | tuple[int, ...],
        activation: Callable,
        output_activation: bool = False,
        *,
        precision: jax.lax.Precision | None = None,
        dtype: Any = jnp.float32,
        key: Array,
    ):
        self.activation = cuex.normalize_function(activation)
        self.output_activation = output_activation
        self.layer_sizes = tuple(layer_sizes)
        self.precision = precision
        keys = jax.random.split(key, len(self.layer_sizes) - 1)
        self.linears = [
            jax.random.normal(k, (in_dim, out_dim), dtype)
            for k, in_dim, out_dim in zip(
                keys, self.layer_sizes[:-1], self.layer_sizes[1:]
            )
        ]

    def __call__(self, x: Array) -> Array:
        for i, w in enumerate(self.linears):
            in_dim = self.layer_sizes[i]
            x = jnp.sqrt(1.0 / in_dim) * jnp.matmul(
                x, w, precision=self.precision
            )  # [..., layer_sizes[i + 1]]
            if i < len(self.linears) - 1 or self.output_activation:
                x = self.activation(x)
        return x


class IrrepsLinear(eqx.Module):
    w: dict[str, Array]
    irreps_in: cue.Irreps = eqx.field(static=True)
    irreps_out: cue.Irreps = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    precision: jax.lax.Precision | None = eqx.field(static=True)
    _ir_map: dict[str, Irrep] = eqx.field(static=True)

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        scale: float = 1.0,
        *,
        precision: jax.lax.Precision | None = None,
        dtype: Any = jnp.float32,
        key: Array,
    ):
        assert irreps_in.regroup() == irreps_in
        assert irreps_out.regroup() == irreps_out
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.scale = scale
        self.precision = precision

        weights, ir_map = {}, {}
        pairs = [
            (mul_in, ir_in, mul_out)
            for mul_in, ir_in in irreps_in
            for mul_out, ir_out in irreps_out
            if ir_in == ir_out
        ]
        keys = jax.random.split(key, max(len(pairs), 1))
        for k, (mul_in, ir_in, mul_out) in zip(keys, pairs):
            ir_key = str(ir_in)
            weights[ir_key] = jax.random.normal(k, (mul_in, mul_out), dtype)
            ir_map[ir_key] = ir_in
        self.w = weights
        self._ir_map = ir_map

    def __call__(self, x: dict[Irrep, Array]) -> dict[Irrep, Array]:
        cuex.ir_dict.assert_mul_ir_dict(self.irreps_in, x)
        x0 = jax.tree.leaves(x)[0]
        shape, dtype = x0.shape[:-2], x0.dtype  # x[ir]: [..., mul, ir.dim]
        y = {
            ir: jnp.zeros(shape + (mul, ir.dim), dtype) for mul, ir in self.irreps_out
        }  # y[ir]: [..., mul, ir.dim]
        for key, w in self.w.items():
            ir = self._ir_map[key]
            y[ir] = (
                jnp.einsum("uv,...ui->...vi", w, x[ir], precision=self.precision)
                * self.scale
                / jnp.sqrt(w.shape[0])
            )  # [..., mul_out, ir.dim]
        cuex.ir_dict.assert_mul_ir_dict(self.irreps_out, y)
        return y


class SphericalHarmonics(eqx.Module):
    max_degree: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    irreps_out: cue.Irreps = eqx.field(static=True)
    poly: Any = eqx.field(static=True)

    def __init__(self, max_degree: int, eps: float = 0.0):
        self.eps = eps
        self.max_degree = max_degree
        desc = cue.descriptors.spherical_harmonics_ir_dict(
            cue.O3(1, -1), list(range(max_degree + 1))
        )
        self.poly = desc.polynomial
        (self.irreps_out,) = desc.output_irreps

    def __call__(self, x: Array) -> dict[Irrep, Array]:
        assert x.shape[-1] == 3
        shape = x.shape[:-1]  # x: [..., 3]
        x = x / safe_norm(x, self.eps, axis=-1, keepdims=True)  # [..., 3]
        outputs = cuex.segmented_polynomial(
            self.poly,
            [x],
            [
                jax.ShapeDtypeStruct(shape + (out.size,), x.dtype)
                for out in self.poly.outputs
            ],
            method="naive",
            name="spherical_harmonics",
        )
        return {
            ir: v.reshape(shape + (1, ir.dim))  # [..., 1, ir.dim]
            for (_, ir), v in zip(self.irreps_out, outputs)
        }


class IrrepsIndexedLinear(eqx.Module):
    w: Array
    irreps_in: cue.Irreps = eqx.field(static=True)
    irreps_out: cue.Irreps = eqx.field(static=True)
    num_indices: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    poly: Any = eqx.field(static=True)

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        num_indices: int,
        scale: float = 1.0,
        *,
        name: str = "indexed_linear",
        dtype: Any = jnp.float32,
        key: Array,
    ):
        assert irreps_in.regroup() == irreps_in
        assert irreps_out.regroup() == irreps_out
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.num_indices = num_indices
        self.scale = scale / math.sqrt(num_indices)
        self.name = name
        self.poly = (
            cue.descriptors.linear(irreps_in, irreps_out).polynomial * self.scale
        )
        self.w = jax.random.normal(key, (num_indices, self.poly.inputs[0].size), dtype)

    def __call__(
        self, x: dict[Irrep, Array], num_index_counts: Array
    ) -> dict[Irrep, Array]:
        cuex.ir_dict.assert_mul_ir_dict(self.irreps_in, x)
        x_ir_mul = jax.tree.map(
            lambda v: rearrange(v, "... m i -> ... i m"), x
        )  # x[ir]: [..., ir.dim, mul]
        x_flat = cuex.ir_dict.dict_to_flat(self.irreps_in, x_ir_mul).astype(
            self.w.dtype
        )  # [num_elements, irreps_in.dim]
        num_elements = x_flat.shape[0]  # num_elements=sum(num_index_counts)
        [y_flat] = cuex.segmented_polynomial(
            self.poly,
            [self.w, x_flat],
            [
                jax.ShapeDtypeStruct(
                    (num_elements, self.poly.outputs[0].size), x_flat.dtype
                )
            ],
            [cuex.Repeats(num_index_counts), None, None],
            method="indexed_linear",
            name=self.name,
        )  # [num_elements, irreps_out.dim]
        y = cuex.ir_dict.flat_to_dict(
            self.irreps_out, y_flat, layout="ir_mul"
        )  # y[ir]: [num_elements, mul, ir.dim]
        cuex.ir_dict.assert_mul_ir_dict(self.irreps_out, y)
        return y


class MessagePassing(eqx.Module):
    name: str = eqx.field(static=True)
    weight_numel: int = eqx.field(static=True)
    irreps_out: cue.Irreps = eqx.field(static=True)
    poly: Any = eqx.field(static=True)

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_sh: cue.Irreps,
        irreps_out: cue.Irreps,
        epsilon: float,
        *,
        name: str = "tensor_product",
    ):
        self.name = name
        desc = cue.descriptors.channelwise_tensor_product_ir_dict(
            irreps_in, irreps_sh, irreps_out
        )
        (self.irreps_out,) = desc.output_irreps
        self.poly = desc.polynomial * epsilon
        self.weight_numel = self.poly.inputs[0].size

    def __call__(
        self,
        weights: Array,  # [num_edges, weight_numel]
        node_feats: dict[Irrep, Array],  # [num_nodes, mul, ir.dim]
        sph: dict[Irrep, Array],  # [num_edges, 1, ir.dim]
        senders: Array,  # [num_edges]
        receivers: Array,  # [num_edges]
        num_nodes: int,
    ) -> dict[Irrep, Array]:
        poly = self.poly
        w = rearrange(
            weights, "e (s m) -> e s m", s=poly.inputs[0].num_segments
        )  # [num_edges, num_segments, segment_mul]
        x1 = jax.tree.map(
            lambda v: rearrange(v, "n m i -> n i m"), node_feats
        )  # [num_nodes, ir.dim, mul]
        x2 = jax.tree.map(
            lambda v: rearrange(v, "e 1 i -> e i"), sph
        )  # [num_edges, ir.dim]
        out_template = {
            ir: jax.ShapeDtypeStruct(
                (num_nodes, desc.num_segments) + desc.segment_shape, w.dtype
            )
            for (_, ir), desc in zip(self.irreps_out, poly.outputs)
        }
        y = cuex.ir_dict.segmented_polynomial_uniform_1d(
            poly,
            [w, x1, x2],
            out_template,
            input_indices=[None, senders, None],
            output_indices=receivers,
            name=self.name,
        )
        return {
            ir: rearrange(v, "n (i s) m -> n (s m) i", i=ir.dim) for ir, v in y.items()
        }  # y[ir]: [num_nodes, mul, ir.dim]


class SymmetricContraction(eqx.Module):
    w: Array
    num_species: int = eqx.field(static=True)
    irreps_out: cue.Irreps = eqx.field(static=True)
    name: str = eqx.field(static=True)
    projection: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    poly: Any = eqx.field(static=True)

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
        key: Array,
    ):
        self.num_species = num_species
        self.irreps_out = irreps_out
        self.name = name
        desc, projection = mace_symmetric_contraction_ir_dict(
            irreps_in, irreps_out, tuple(range(1, correlation + 1))
        )
        projection_array = np.asarray(projection, dtype=np.dtype(dtype))
        self.projection = tuple(
            tuple(float(x) for x in row) for row in projection_array
        )
        self.poly = desc.polynomial
        self.w = jax.random.normal(
            key, (num_species, projection_array.shape[0], num_features), dtype
        )

    def __call__(
        self, x: dict[Irrep, Array], species_counts: Array
    ) -> dict[Irrep, Array]:
        projection = jnp.asarray(self.projection, dtype=self.w.dtype)
        w = jnp.einsum(
            "zau,ab->zbu", self.w, projection
        )  # [num_species, num_weights, num_features]
        num_nodes = jax.tree.leaves(x)[0].shape[0]  # x[ir]: [num_nodes, mul, ir.dim]
        species_idx = jnp.repeat(
            jnp.arange(self.num_species), species_counts, total_repeat_length=num_nodes
        )  # [num_nodes]
        x = jax.tree.map(
            lambda v: rearrange(v, "n m i -> n i m"), x
        )  # x[ir]: [num_nodes, ir.dim, mul]
        y = cuex.ir_dict.segmented_polynomial_uniform_1d(
            self.poly,
            [w, x],
            cuex.ir_dict.mul_ir_dict(self.irreps_out, None),
            input_indices=[species_idx, None],
            name=self.name,
        )
        return jax.tree.map(
            lambda v: rearrange(v, "n i m -> n m i"), y
        )  # y[ir]: [num_nodes, mul, ir.dim]


class MACELayer(eqx.Module):
    sph: SphericalHarmonics
    linear_up: IrrepsLinear
    message: MessagePassing
    radial_mlp: MLP
    linear_down: IrrepsLinear
    skip: IrrepsIndexedLinear | None
    linZ_first: IrrepsIndexedLinear | None
    sc: SymmetricContraction
    linear_sc: IrrepsLinear
    readout_mlp: IrrepsLinear | None
    readout: IrrepsLinear
    is_last: bool = eqx.field(static=True)

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
        name: str,
        dtype: Any,
        key: Array,
    ):
        self.is_last = is_last
        hidden_out = (
            hidden_irreps.filter(keep=[ir for _, ir in output_irreps])
            if is_last
            else hidden_irreps
        )
        sph_irreps = cue.Irreps(
            cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in range(max_ell + 1)]
        )
        keys = iter(jax.random.split(key, 9))
        self.sph = SphericalHarmonics(max_ell)
        self.linear_up = IrrepsLinear(
            input_irreps,
            input_irreps,
            precision=jax.lax.Precision.HIGHEST,
            dtype=dtype,
            key=next(keys),
        )
        self.message = MessagePassing(
            input_irreps,
            sph_irreps,
            num_features * interaction_irreps,
            epsilon,
            name=f"{name}_tensor_product",
        )
        self.radial_mlp = MLP(
            [radial_dim, 64, 64, 64, self.message.weight_numel],
            jax.nn.silu,
            False,
            dtype=dtype,
            key=next(keys),
        )
        self.linear_down = IrrepsLinear(
            self.message.irreps_out,
            num_features * interaction_irreps,
            precision=jax.lax.Precision.HIGHEST,
            dtype=dtype,
            key=next(keys),
        )
        self.skip = (
            IrrepsIndexedLinear(
                input_irreps,
                num_features * hidden_out,
                num_species,
                name=f"{name}_skip",
                dtype=dtype,
                key=next(keys),
            )
            if has_skip
            else None
        )
        self.linZ_first = (
            IrrepsIndexedLinear(
                num_features * interaction_irreps,
                num_features * interaction_irreps,
                num_species,
                name=f"{name}_skip_first",
                dtype=dtype,
                key=next(keys),
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
            name=f"{name}_symmetric_contraction",
            dtype=dtype,
            key=next(keys),
        )
        self.linear_sc = IrrepsLinear(
            num_features * hidden_out,
            num_features * hidden_out,
            precision=jax.lax.Precision.HIGHEST,
            dtype=dtype,
            key=next(keys),
        )
        if is_last:
            readout_irreps = cue.Irreps(cue.O3, "16x0e")
            self.readout_mlp = IrrepsLinear(
                num_features * hidden_out,
                readout_irreps,
                precision=jax.lax.Precision.HIGHEST,
                dtype=dtype,
                key=next(keys),
            )
            self.readout = IrrepsLinear(
                readout_irreps,
                output_irreps,
                precision=jax.lax.Precision.HIGHEST,
                dtype=dtype,
                key=next(keys),
            )
        else:
            self.readout_mlp = None
            self.readout = IrrepsLinear(
                num_features * hidden_out,
                output_irreps,
                precision=jax.lax.Precision.HIGHEST,
                dtype=dtype,
                key=next(keys),
            )

    def __call__(
        self,
        vectors: Array,  # [num_edges, 3]
        node_feats: dict[Irrep, Array],  # [num_nodes, mul, ir.dim]
        species_counts: Array,  # [num_species]
        radial_embed: Array,  # [num_edges, radial_dim]
        senders: Array,  # [num_edges]
        receivers: Array,  # [num_edges]
        num_nodes: int,
    ) -> tuple[dict[Irrep, Array], dict[Irrep, Array]]:
        sph = self.sph(vectors)  # [num_edges, 1, ir.dim]
        skip = self.skip(node_feats, species_counts) if self.skip else None
        h = self.linear_up(node_feats)  # [num_nodes, mul, ir.dim]
        h = self.message(
            self.radial_mlp(radial_embed), h, sph, senders, receivers, num_nodes
        )  # [num_nodes, mul, ir.dim]
        h = self.linear_down(h)  # [num_nodes, mul, ir.dim]
        if self.linZ_first is not None:
            h = self.linZ_first(h, species_counts)  # [num_nodes, mul, ir.dim]
        h = self.linear_sc(self.sc(h, species_counts))  # [num_nodes, mul, ir.dim]
        if skip is not None:
            h = cuex.ir_dict.irreps_add(h, skip)  # [num_nodes, mul, ir.dim]
        out = scalar_activation(self.readout_mlp(h), jax.nn.silu) if self.is_last else h
        return self.readout(out), h  # [num_nodes, 1, 1], [num_nodes, mul, ir.dim]


class MACEModel(eqx.Module):
    embedding: Array
    layers: list[MACELayer]
    offsets: tuple[float, ...] = eqx.field(static=True)
    num_species: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    num_radial_basis: int = eqx.field(static=True)

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
        key: Array,
    ):
        self.offsets = tuple(float(x) for x in np.asarray(offsets).reshape(-1))
        self.num_species = num_species
        self.cutoff = cutoff
        self.num_radial_basis = num_radial_basis
        output_irreps = cue.Irreps(cue.O3, "1x0e")
        full_hidden = hidden_irreps.set_mul(num_features)
        keys = jax.random.split(key, num_layers + 1)
        self.embedding = jax.random.normal(keys[0], (num_species, num_features), dtype)
        self.layers = []
        for i in range(num_layers):
            is_first, is_last = i == 0, i == num_layers - 1
            input_irreps = full_hidden.filter(keep="0e") if is_first else full_hidden
            self.layers.append(
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
                    has_skip=not is_first or skip_connection_first_layer,
                    has_linZ_first=is_first and not skip_connection_first_layer,
                    is_last=is_last,
                    name=f"layer_{i}",
                    dtype=dtype,
                    key=keys[i + 1],
                )
            )

    def node_energies(
        self,
        edge_vectors: Array,  # [num_edges, 3]
        node_species: Array,  # [num_nodes]
        senders: Array,  # [num_edges]
        receivers: Array,  # [num_edges]
        species_counts: Array,  # [num_species]
        edge_mask: Array,  # [num_edges]
    ) -> Array:
        edge_mask = edge_mask[:, None]  # [num_edges, 1]
        num_nodes = node_species.shape[0]
        node_feats_array = jnp.repeat(
            self.embedding, species_counts, axis=0, total_repeat_length=num_nodes
        )  # [num_nodes, num_features]
        node_feats = {
            cue.O3(0, 1): (node_feats_array / jnp.sqrt(self.num_species))[:, :, None]
        }  # [num_nodes, num_features, 1]
        radial_embed = jax.vmap(
            lambda r: radial_basis(
                jnp.linalg.norm(r), self.cutoff, self.num_radial_basis
            )
        )(edge_vectors)  # [num_edges, num_radial_basis]
        atom_energies = jnp.zeros(num_nodes, edge_vectors.dtype)  # [num_nodes]
        for layer in self.layers:
            out, node_feats = layer(
                edge_vectors,
                node_feats,
                species_counts,
                radial_embed,
                senders,
                receivers,
                num_nodes,
            )
            atom_energies = atom_energies + jnp.squeeze(
                next(iter(out.values())), (-1, -2)
            )  # [num_nodes]
        atom_energies = atom_energies + jnp.repeat(
            jnp.asarray(self.offsets, atom_energies.dtype),
            species_counts,
            total_repeat_length=num_nodes,
        )  # [num_nodes]
        return atom_energies

    def __call__(
        self,
        batch: dict[str, Array],
        *,
        compute_virial: bool = False,
    ) -> tuple[Array, ...]:
        edge_vectors = batch["nn_vecs"]  # [num_edges, 3]
        node_species = batch["species"]  # [num_nodes]
        senders, receivers = batch["inda"], batch["indb"]  # [num_edges]
        node_graph_index = batch["inde"]  # [num_nodes]
        edge_mask = batch["mask"].astype(jnp.bool_)  # [num_edges]
        num_graphs = batch["nats"].shape[0]

        species_sort_perm = jnp.argsort(node_species)  # [num_nodes]
        node_species = node_species[species_sort_perm]  # [num_nodes]
        node_graph_index = node_graph_index[species_sort_perm]  # [num_nodes]
        inverse_species_sort_perm = (
            jnp.zeros_like(species_sort_perm)
            .at[species_sort_perm]
            .set(jnp.arange(species_sort_perm.shape[0]))
        )  # [num_nodes]
        senders, receivers = (
            inverse_species_sort_perm[senders],
            inverse_species_sort_perm[receivers],
        )  # [num_edges]
        species_counts = (
            jnp.zeros((self.num_species,), jnp.int32).at[node_species].add(1)
        )  # [num_species]

        def total_energy(edge_vectors: Array) -> tuple[Array, Array]:
            atom_energies = self.node_energies(
                edge_vectors,
                node_species,
                senders,
                receivers,
                species_counts,
                edge_mask,
            )
            return jnp.sum(atom_energies), atom_energies  # scalar, [num_nodes]

        forces, atom_energies = jax.grad(total_energy, has_aux=True)(edge_vectors)
        forces = jnp.where(edge_mask[:, None], forces, 0.0)  # [num_edges, 3]
        E = (
            jnp.zeros(num_graphs, atom_energies.dtype)
            .at[node_graph_index]
            .add(atom_energies)
        )  # [num_graphs]
        F = jnp.zeros(
            (atom_energies.shape[0], 3), atom_energies.dtype
        )  # [num_nodes, 3]
        F = (
            F.at[senders]
            .add(forces)
            .at[receivers]
            .add(-forces)[inverse_species_sort_perm]
        )  # [num_nodes, 3]
        if compute_virial:
            edge_virials = -jnp.einsum(
                "ei,ej->eij", forces, edge_vectors
            )  # [num_edges, 3, 3]
            edge_virials = jnp.where(
                edge_mask[:, None, None], edge_virials, 0.0
            )  # [num_edges, 3, 3]
            edge_graph_index = node_graph_index[senders]  # [num_edges]
            virials = (
                jnp.zeros((num_graphs, 3, 3), edge_virials.dtype)
                .at[edge_graph_index]
                .add(edge_virials)
            )  # [num_graphs, 3, 3]
            virials = 0.5 * (virials + jnp.swapaxes(virials, -1, -2))  # [G, 3, 3]
            return E, F, virials
        return E, F


def make_inference(model: MACEModel):
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    @jax.jit
    def inference_step(
        params: Any, batch_dict: dict[str, Array]
    ) -> tuple[Array, Array]:
        return params(batch_dict)

    return params, inference_step


def make_train_step(model: MACEModel, tx: optax.GradientTransformation):
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    opt_state = tx.init(params)

    @jax.jit
    def step(
        params: Any,
        opt_state: Any,
        batch_dict: dict[str, Array],
        target_E: Array,
        target_F: Array,
    ) -> tuple[Any, Any, Array]:
        def loss_from_params(params: Any) -> Array:
            E, F = params(batch_dict)
            return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_from_params)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), opt_state, loss

    return params, opt_state, step


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
    num_species, num_graphs, avg_num_neighbors = 50, 100, 20
    prefix = model_size.split("-")[0]
    num_features = {
        "MP-S": 128,
        "MP-M": 128,
        "MP-L": 128,
        "OFF-S": 64 + 32,
        "OFF-M": 128,
        "OFF-L": 128 + 64,
    }[model_size]
    hidden_irreps = {
        "MP-S": "0e",
        "MP-M": "0e+1o",
        "MP-L": "0e+1o+2e",
        "OFF-S": "0e",
        "OFF-M": "0e+1o",
        "OFF-L": "0e+1o+2e",
    }[model_size]
    model = MACEModel(
        num_layers=2,
        num_features=num_features,
        num_species=num_species,
        max_ell=3,
        correlation=3,
        num_radial_basis=8,
        interaction_irreps=cue.Irreps(cue.O3, "0e+1o+2e+3o"),
        hidden_irreps=cue.Irreps(cue.O3, hidden_irreps),
        offsets=np.zeros(num_species),
        cutoff=5.0,
        epsilon=1 / avg_num_neighbors,
        skip_connection_first_layer=prefix == "MP",
        dtype=dtype,
        key=jax.random.key(0),
    )
    edge_vectors = jax.random.normal(
        jax.random.key(0), (num_edges, 3), dtype
    )  # [num_edges, 3]
    species = jax.random.randint(
        jax.random.key(0), (num_atoms,), 0, num_species, jnp.int32
    )  # [num_atoms]
    senders, receivers = jax.random.randint(
        jax.random.key(0), (2, num_edges), 0, num_atoms, jnp.int32
    )  # [num_edges]
    graph_index = jnp.sort(
        jax.random.randint(jax.random.key(0), (num_atoms,), 0, num_graphs, jnp.int32)
    )  # [num_atoms]
    target_E = jax.random.normal(jax.random.key(0), (num_graphs,), dtype)  # [G]
    target_F = jax.random.normal(jax.random.key(0), (num_atoms, 3), dtype)  # [N, 3]
    batch_dict = dict(
        nn_vecs=edge_vectors,
        species=species,
        inda=senders,
        indb=receivers,
        inde=graph_index,
        nats=jnp.zeros((num_graphs,), dtype=jnp.int32).at[graph_index].add(1),
        mask=jnp.ones((num_edges,), dtype=jnp.int32),
    )  # flat graph batch

    num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(
        f"MACE-EQX {model_size}: {num_atoms} atoms, {num_edges} edges, "
        f"{dtype}, {num_params:,} params",
        flush=True,
    )

    train_time = inference_time = train_compile = inference_compile = 0.0
    tx = optax.adam(1e-2)
    params, opt_state, step = make_train_step(model, tx)
    infer_params, inference_step = make_inference(model)
    if mode in ["train", "both"]:
        t0 = time.perf_counter()
        params, opt_state, loss = step(
            params, opt_state, batch_dict, target_E, target_F
        )
        jax.block_until_ready(loss)
        train_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(10):
            params, opt_state, loss = step(
                params, opt_state, batch_dict, target_E, target_F
            )
        jax.block_until_ready(loss)
        train_time = 1e3 * (time.perf_counter() - t0) / 10
        infer_params = params
    if mode in ["inference", "both"]:
        t0 = time.perf_counter()
        out = inference_step(infer_params, batch_dict)
        jax.block_until_ready(out)
        inference_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(10):
            out = inference_step(infer_params, batch_dict)
        jax.block_until_ready(out)
        inference_time = 1e3 * (time.perf_counter() - t0) / 10
    if mode == "both":
        print(
            f"train: {train_time:.1f}ms, inference: {inference_time:.1f}ms, "
            f"compile: {train_compile:.1f}s + {inference_compile:.1f}s",
            flush=True,
        )
    elif mode == "train":
        print(f"train: {train_time:.1f}ms, compile: {train_compile:.1f}s", flush=True)
    else:
        print(
            f"inference: {inference_time:.1f}ms, compile: {inference_compile:.1f}s",
            flush=True,
        )
    try:
        import nvtx

        cuda = ctypes.CDLL("libcudart.so")
        cuda.cudaProfilerStart()
        if mode in ["train", "both"]:
            with nvtx.annotate("Train", color="green"):
                params, opt_state, loss = step(
                    params, opt_state, batch_dict, target_E, target_F
                )
                jax.block_until_ready(loss)
        if mode in ["inference", "both"]:
            with nvtx.annotate("Inference", color="blue"):
                jax.block_until_ready(inference_step(infer_params, batch_dict))
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
    parser.add_argument(
        "--larger",
        action="store_true",
        help="Use larger benchmark sizes (4x atoms and edges)",
    )
    args = parser.parse_args()

    defaults = {"MP": (3_000, 160_000), "OFF": (4_000, 70_000)}
    defaults_larger = {"MP": (12_000, 640_000), "OFF": (16_000, 280_000)}
    for dtype_str in args.dtype:
        for model_size in args.model:
            prefix = model_size.split("-")[0]
            size_defaults = defaults_larger if args.larger else defaults
            num_atoms = args.nodes or size_defaults[prefix][0]
            num_edges = args.edges or size_defaults[prefix][1]
            benchmark(
                model_size, num_atoms, num_edges, getattr(jnp, dtype_str), args.mode
            )


if __name__ == "__main__":
    main()
