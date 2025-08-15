# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cuequivariance as cue
import cuequivariance_jax as cuex


def test_special_double_backward():
    e = cue.descriptors.symmetric_contraction(
        32 * cue.Irreps("O3", "0e + 1o + 2e"), 32 * cue.Irreps("O3", "0e + 1o"), [1, 2]
    )
    rep_w, rep_x = e.inputs

    def h(*inputs):
        return cuex.equivariant_polynomial(e, inputs, method="naive")

    h0 = lambda w, x: h(w, x).array.sum() ** 2  # noqa
    h1 = lambda w, x: jax.grad(h0, 1)(w, x).array.sum() ** 2  # noqa

    w = jax.random.normal(jax.random.key(0), (1, rep_w.dim))
    x = cuex.randn(jax.random.key(1), rep_x, (3,))
    jax.grad(h1, 0)(w, x)


def make_uniform1d_descriptors():
    yield (
        cue.descriptors.channelwise_tensor_product(
            64 * cue.Irreps("SO3", "0 + 1 + 2"),
            cue.Irreps("SO3", "0 + 1 + 2 + 3"),
            cue.Irreps("SO3", "0 + 1 + 2"),
        )
        .squeeze_modes()
        .flatten_coefficient_modes()
    )
    yield (
        cue.descriptors.symmetric_contraction(
            64 * cue.Irreps("SO3", "0 + 1 + 2"),
            64 * cue.Irreps("SO3", "0 + 1"),
            [0, 1, 2, 3],
        )
    )


@pytest.mark.parametrize("e", make_uniform1d_descriptors())
def test_method_uniform_1d(e: cue.EquivariantPolynomial):
    if jax.default_backend() != "gpu":
        pytest.skip("test_custom_kernel requires CUDA")

    jax.config.update("jax_enable_x64", True)

    num_nodes, num_edges = 30, 100
    Zs = [num_edges] * e.num_inputs
    Zs[0] = num_nodes
    inputs = [
        cuex.randn(jax.random.key(0), rep, (Z,), jnp.float64)
        for Z, rep in zip(Zs, e.inputs)
    ]
    indices = [None] * e.num_operands
    indices[0] = jax.random.randint(
        jax.random.key(1), (num_edges,), 0, num_nodes, jnp.int32
    )
    indices[-1] = jax.random.randint(
        jax.random.key(1), (num_edges,), 0, num_nodes, jnp.int32
    )
    output_batch_shape = (num_nodes,)

    def fwd(inputs, indices, method: str):
        return cuex.equivariant_polynomial(
            e,
            inputs,
            jax.ShapeDtypeStruct(output_batch_shape + (e.outputs[0].dim,), jnp.float64),
            indices,
            method=method,
        ).array

    out0 = fwd(inputs, indices, method="naive")
    out1 = fwd(inputs, indices, method="uniform_1d")
    assert out0.shape == out1.shape
    assert out0.dtype == out1.dtype
    np.testing.assert_allclose(out0, out1, atol=1e-12, rtol=0)

    def bwd(inputs, indices, method: str):
        return jax.grad(lambda *inputs: fwd(inputs, indices, method).sum(), argnums=0)(
            *inputs
        ).array

    out0 = bwd(inputs, indices, method="naive")
    out1 = bwd(inputs, indices, method="uniform_1d")
    assert out0.shape == out1.shape
    assert out0.dtype == out1.dtype
    np.testing.assert_allclose(out0, out1, atol=1e-12, rtol=0)

    def bwd2(inputs, indices, method: str):
        return jax.grad(lambda *inputs: bwd(inputs, indices, method).sum(), argnums=1)(
            *inputs
        ).array

    out0 = bwd2(inputs, indices, method="naive")
    out1 = bwd2(inputs, indices, method="uniform_1d")
    assert out0.shape == out1.shape
    assert out0.dtype == out1.dtype
    np.testing.assert_allclose(out0, out1, atol=1e-12, rtol=0)
