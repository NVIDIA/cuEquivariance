# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Flax NNX layers and utilities."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import Array

import cuequivariance as cue
from cuequivariance import Irrep

from .activation import normalize_function
from .ir_dict import assert_mul_ir_dict, dict_to_irreps
from .rep_array.rep_array_ import RepArray
from .segmented_polynomials.segmented_polynomial import segmented_polynomial
from .segmented_polynomials.utils import Repeats
from .spherical_harmonics import spherical_harmonics

__all__ = [
    "IrrepsIndexedLinear",
    "IrrepsLinear",
    "IrrepsNormalize",
    "MLP",
    "SphericalHarmonics",
]


def _safe_norm(
    x: Array, eps: float = 0.0, axis: int = -1, keepdim: bool = False
) -> Array:
    sn = jnp.sum(jnp.conj(x) * x, axis=axis, keepdims=keepdim) + eps**2
    if eps == 0.0:
        sn_safe = jnp.where(sn == 0.0, 1.0, sn)
        rsn_safe = jnp.sqrt(sn_safe)
        return jnp.where(sn == 0.0, 0.0, rsn_safe)
    return jnp.sqrt(sn)


class IrrepsLinear(nnx.Module):
    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        scale: float = 1.0,
        *,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert irreps_in.regroup() == irreps_in
        assert irreps_out.regroup() == irreps_out

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.scale = scale

        irrep_cls = None
        for _, ir in irreps_in:
            irrep_cls = type(ir)
            break
        if irrep_cls is None:
            for _, ir in irreps_out:
                irrep_cls = type(ir)
                break
        if irrep_cls is None:
            raise ValueError("IrrepsLinear requires non-empty irreps.")
        self._irrep_cls = irrep_cls

        w = {}
        for mul_in, ir_in in irreps_in:
            for mul_out, ir_out in irreps_out:
                if ir_in == ir_out:
                    key = str(ir_in)
                    w[key] = nnx.Param(
                        jax.random.normal(rngs.params(), (mul_in, mul_out), dtype)
                    )
        self.w = nnx.Dict(w)

    def __call__(self, x: dict[Irrep, Array]) -> dict[Irrep, Array]:
        assert_mul_ir_dict(self.irreps_in, x)

        x0 = jax.tree.leaves(x)[0]
        shape = x0.shape[:-2]
        dtype = x0.dtype

        y = {ir: jnp.zeros(shape + (mul, ir.dim), dtype) for mul, ir in self.irreps_out}
        for key, w in self.w.items():
            ir = self._irrep_cls.from_string(key)
            y[ir] = (
                jnp.einsum("uv,...ui->...vi", w[...], x[ir])
                * self.scale
                / jnp.sqrt(w[...].shape[0])
            )

        assert_mul_ir_dict(self.irreps_out, y)
        return y


class SphericalHarmonics(nnx.Module):
    def __init__(self, max_degree: int, eps: float = 0.0):
        self.eps = eps
        self.max_degree = max_degree
        self.irreps_in = cue.Irreps(cue.O3, "1o")
        self.irreps_out = cue.Irreps(
            cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in range(max_degree + 1)]
        )

    def __call__(self, x: Array) -> dict[Irrep, Array]:
        assert x.shape[-1] == 3
        shape = x.shape[:-1]

        x = RepArray(self.irreps_in, x, cue.ir_mul)
        x = jax.tree.map(
            lambda v: v / _safe_norm(v, self.eps, axis=-1, keepdim=True), x
        )
        y = spherical_harmonics(range(self.max_degree + 1), x, normalize=False)

        y = {
            ir: rearrange(v, "... i m -> ... m i")
            for (_, ir), v in zip(y.irreps, y.segments)
        }
        actual = jax.tree.map(lambda x: x.shape, y)
        expected = {
            cue.O3(L, (-1) ** L): shape + (1, 2 * L + 1)
            for L in range(self.max_degree + 1)
        }
        assert actual == expected, f"y: {actual}, expected: {expected}"
        return y


class IrrepsNormalize(nnx.Module):
    def __init__(self, eps: float, scale: float = 1.0, skip_scalars: bool = False):
        assert eps >= 0.0
        self.eps = eps
        self.scale = scale
        self.skip_scalars = skip_scalars

    def __call__(self, x: dict[Irrep, Array]) -> dict[Irrep, Array]:
        def fn(ir: cue.Irrep, v: Array) -> Array:
            assert v.shape[-1] == ir.dim

            if self.skip_scalars and ir.is_scalar():
                return v * self.scale

            sn = jnp.conj(v) * v
            sn = jnp.sum(sn, axis=-1, keepdims=True)
            sn = jnp.mean(sn, axis=-2, keepdims=True)
            norm = jnp.sqrt(sn + self.eps**2)
            return v / norm * self.scale

        return {ir: fn(ir, v) for ir, v in x.items()}


class MLP(nnx.Module):
    def __init__(
        self,
        layer_sizes: list[int],
        activation: Callable,
        output_activation: bool = False,
        *,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.activation = normalize_function(activation)
        self.output_activation = output_activation
        self.num_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes

        linears = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = jax.random.normal(rngs.params(), (in_dim, out_dim), dtype)
            linears.append(nnx.Param(w))
        self.linears = nnx.List(linears)

    def __call__(self, x: Array) -> Array:
        for i, w in enumerate(self.linears):
            in_dim = self.layer_sizes[i]
            x = jnp.sqrt(1.0 / in_dim) * (x @ w[...])
            if i < self.num_layers - 1 or self.output_activation:
                x = self.activation(x)
        return x


class IrrepsIndexedLinear(nnx.Module):
    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        num_indices: int,
        scale: float = 1.0,
        *,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert irreps_in.regroup() == irreps_in
        assert irreps_out.regroup() == irreps_out

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.num_indices = num_indices
        self.scale = scale / jnp.sqrt(num_indices)

        self.e = cue.descriptors.linear(irreps_in, irreps_out) * self.scale
        self.w = nnx.Param(
            jax.random.normal(rngs.params(), (num_indices, self.e.inputs[0].dim), dtype)
        )

    def __call__(
        self, x: dict[Irrep, Array], num_index_counts: Array
    ) -> dict[Irrep, Array]:
        assert_mul_ir_dict(self.irreps_in, x)

        # Convert dict (batch, mul, ir.dim) -> ir_mul flat order
        x_ir_mul = jax.tree.map(lambda v: rearrange(v, "... m i -> ... i m"), x)
        x_flat = dict_to_irreps(self.irreps_in, x_ir_mul)
        num_elements = x_flat.shape[0]

        p = self.e.polynomial
        [y_flat] = segmented_polynomial(
            p,
            [self.w[...], x_flat],
            [jax.ShapeDtypeStruct((num_elements, p.outputs[0].size), x_flat.dtype)],
            [Repeats(num_index_counts), None, None],
            method="indexed_linear",
            name="indexed_linear",
        )

        # Convert ir_mul flat -> dict (batch, mul, ir.dim)
        y = {}
        offset = 0
        for mul, ir in self.irreps_out:
            size = mul * ir.dim
            segment = y_flat[..., offset : offset + size]
            y[ir] = rearrange(segment, "... (i m) -> ... m i", i=ir.dim, m=mul)
            offset += size
        assert_mul_ir_dict(self.irreps_out, y)
        return y
