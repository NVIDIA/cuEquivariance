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
import jax.random as random
import pytest
from cuequivariance_jax.triangle._layer_norm_transpose import layer_norm_transpose
from jax import test_util

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize(
    "layout_str,input_shape,expected_output_shape,feature_dim",
    [
        # 2D layouts
        ("nd->nd", (16, 64), (16, 64), 64),
        # ("nd->dn", (16, 64), (64, 16), 64),
        # 3D layouts
        ("bnd->bnd", (2, 16, 64), (2, 16, 64), 64),
        # ("bdn->bnd", (2, 64, 16), (2, 16, 64), 64),
        # ("bnd->bdn", (2, 16, 64), (2, 64, 16), 64),
        # ("dbn->bnd", (64, 2, 16), (2, 16, 64), 64),
        # ("bnd->dbn", (2, 16, 64), (64, 2, 16), 64),
        # 4D layouts
        ("bijd->bijd", (2, 8, 8, 64), (2, 8, 8, 64), 64),
        # ("bijd->bdij", (2, 8, 8, 64), (2, 64, 8, 8), 64),
        # ("bdij->bijd", (2, 64, 8, 8), (2, 8, 8, 64), 64),
        # ("dbij->bijd", (64, 2, 8, 8), (2, 8, 8, 64), 64),
        # ("bijd->dbij", (2, 8, 8, 64), (64, 2, 8, 8), 64),
    ],
)
def test_layer_norm_transpose(
    elementwise_affine, layout_str, input_shape, expected_output_shape, feature_dim
):
    """Test layer_norm_transpose across all layouts and elementwise_affine settings."""
    key = random.PRNGKey(42)
    eps = 1e-5
    D = feature_dim

    # Generate test data
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    x = random.normal(subkey1, input_shape, dtype=jnp.float32)
    w = random.normal(subkey2, (D,), dtype=jnp.float32) * 0.1 + 1.0
    b = random.normal(subkey3, (D,), dtype=jnp.float32) * 0.1

    # Test implementation on default device
    out = layer_norm_transpose(
        x, w, b, layout=layout_str, eps=eps, elementwise_affine=elementwise_affine
    )

    # Basic checks
    assert out.shape == expected_output_shape
    assert not jnp.any(jnp.isnan(out)) and not jnp.any(jnp.isinf(out))

    # Test implementation on CPU as reference
    cpu_device = jax.devices("cpu")[0]
    x_cpu = jax.device_put(x, cpu_device)
    w_cpu = jax.device_put(w, cpu_device)
    b_cpu = jax.device_put(b, cpu_device)

    ref_out = layer_norm_transpose(
        x_cpu,
        w_cpu,
        b_cpu,
        layout=layout_str,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )

    # Compare outputs
    assert jnp.allclose(out, ref_out, rtol=1e-5, atol=1e-6)

    # Test gradients on default device
    def loss_fn(x, w, b):
        return jnp.mean(
            layer_norm_transpose(
                x,
                w,
                b,
                layout=layout_str,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
            ** 2
        )

    test_util.check_grads(
        loss_fn, (x, w, b), order=1, modes=["rev"], atol=1e-2, rtol=1e-2
    )

    # Test gradients on CPU as well
    def loss_fn_cpu(x, w, b):
        return jnp.mean(
            layer_norm_transpose(
                x,
                w,
                b,
                layout=layout_str,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
            ** 2
        )

    test_util.check_grads(
        loss_fn_cpu, (x_cpu, w_cpu, b_cpu), order=1, modes=["rev"], atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("fallback", [True, False])
def test_layer_norm_vmap(fallback):
    V = 2  # vmap_size
    B = 3  # batch_size
    N = 4  # seq_len
    D = 64  # d_model

    key = jax.random.key(123)
    keys = jax.random.split(key, 5)
    dtype = jnp.float32

    x = jax.random.normal(keys[0], (V, B, N, D), dtype)
    w = jax.random.normal(keys[1], (D,), dtype)
    b = jax.random.normal(keys[2], (D,), dtype)

    def f(x, w, b):
        return layer_norm_transpose(
            x,
            w,
            b,
            layout="bnd->dbn",
            eps=1e-5,
            elementwise_affine=True,
            fallback=fallback,
        )

    out = jax.vmap(f, (0, None, None))(x, w, b)
    assert out.shape == (V, D, B, N)
