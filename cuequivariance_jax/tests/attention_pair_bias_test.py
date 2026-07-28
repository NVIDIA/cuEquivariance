# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cuequivariance_jax as cuex


def _inputs(dtype=jnp.float32):
    B, M, N, D, H, DH, DZ = 2, 2, 3, 4, 2, 3, 5
    keys = jax.random.split(jax.random.key(123), 18)

    def normal(index, shape):
        return jax.random.normal(keys[index], shape, dtype=dtype)

    return {
        "single_repr": normal(0, (B * M, N, D)),
        "pair_repr": normal(1, (B, N, N, DZ)),
        "mask": jnp.asarray(
            [[1.0, 1.0, 0.25], [1.0, 0.5, 1.0]], dtype=dtype
        ),
        "num_heads": H,
        "w_ln_a": normal(2, (D,)),
        "b_ln_a": normal(3, (D,)),
        "w_proj_q": normal(4, (H * DH, D)),
        "b_proj_q": normal(5, (H * DH,)),
        "w_proj_k": normal(6, (H * DH, D)),
        "w_proj_v": normal(7, (H * DH, D)),
        "w_proj_g": normal(8, (H * DH, D)),
        "w_proj_o": normal(9, (D, H * DH)),
        "w_proj_z": normal(10, (H, DZ)),
        "b_proj_o": normal(11, (D,)),
        "b_proj_z": normal(12, (H,)),
        "w_ln_z": normal(13, (DZ,)),
        "b_ln_z": normal(14, (DZ,)),
        "b_proj_k": normal(15, (H * DH,)),
        "b_proj_v": normal(16, (H * DH,)),
        "b_proj_g": normal(17, (H * DH,)),
        "w_ln_q": jnp.linspace(0.5, 1.5, H * DH, dtype=dtype),
        "b_ln_q": jnp.linspace(-0.2, 0.2, H * DH, dtype=dtype),
        "w_ln_k": jnp.linspace(0.75, 1.25, H * DH, dtype=dtype),
        "b_ln_k": jnp.linspace(0.1, -0.1, H * DH, dtype=dtype),
    }


def _call(values, **kwargs):
    arguments = dict(values)
    arguments.update(kwargs)
    return cuex.attention_pair_bias(**arguments)


def test_attention_pair_bias_raw_and_cached_projection_match():
    values = _inputs()
    output, z_proj = _call(values, return_z_proj=True)

    cached = dict(values)
    cached["pair_repr"] = z_proj
    cached["w_proj_z"] = None
    output_cached, returned = _call(cached, is_cached_z_proj=True)

    assert output.shape == values["single_repr"].shape
    assert z_proj.shape == (2, 2, 3, 3)
    assert returned is None
    np.testing.assert_allclose(output_cached, output, rtol=2e-5, atol=2e-5)


def test_attention_pair_bias_jit_vjp_and_floating_mask_gradient():
    values = _inputs()

    def loss(single_repr, pair_repr, mask):
        local = dict(values)
        local.update(
            single_repr=single_repr,
            pair_repr=pair_repr,
            mask=mask,
        )
        output, z_proj = _call(local, return_z_proj=True)
        return jnp.sum(output.astype(jnp.float32) ** 2) + 0.01 * jnp.sum(z_proj)

    compiled = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))
    value, gradients = compiled(
        values["single_repr"], values["pair_repr"], values["mask"]
    )

    assert jnp.isfinite(value)
    for gradient, expected in zip(
        gradients,
        (values["single_repr"], values["pair_repr"], values["mask"]),
        strict=True,
    ):
        assert gradient.shape == expected.shape
        assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradients[2] != 0)


def test_attention_pair_bias_nested_vmap_reference_path():
    values = _inputs()
    single = jnp.stack([values["single_repr"], values["single_repr"] + 0.1])
    pair = jnp.stack([values["pair_repr"], values["pair_repr"] - 0.1])
    mask = jnp.stack([values["mask"], values["mask"]])
    single = jnp.stack([single, single + 0.2])
    pair = jnp.stack([pair, pair + 0.2])
    mask = jnp.stack([mask, mask])

    def operation(single_repr, pair_repr, pair_mask):
        local = dict(values)
        local.update(
            single_repr=single_repr,
            pair_repr=pair_repr,
            mask=pair_mask,
        )
        return _call(local)[0]

    output = jax.jit(jax.vmap(jax.vmap(operation)))(single, pair, mask)
    assert output.shape == (2, 2) + values["single_repr"].shape
    assert jnp.all(jnp.isfinite(output))
