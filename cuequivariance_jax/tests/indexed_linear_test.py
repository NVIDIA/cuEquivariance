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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16, jnp.float64])
def test_indexed_linear(dtype):
    jax.config.update("jax_enable_x64", True)
    impl = "cuda" if "cuda" in jax.devices()[0].platform else "auto"

    num_species_total = 3
    batch_size = 10
    input_dim = 8
    output_dim = 16
    num_species = jnp.array([3, 4, 3], dtype=jnp.int32)
    input_array = jax.random.normal(jax.random.key(0), (batch_size, input_dim), dtype)
    input_irreps = cue.Irreps(cue.O3, f"{input_dim}x0e")
    output_irreps = cue.Irreps(cue.O3, f"{output_dim}x0e")
    e = cue.descriptors.linear(input_irreps, output_irreps)
    w = jax.random.normal(
        jax.random.key(1), (num_species_total, e.inputs[0].dim), dtype
    )

    result = cuex.experimental.indexed_linear(
        e.polynomial, num_species, w, input_array, impl=impl
    )
    assert result.shape == (batch_size, output_dim)

    [ref] = cuex.segmented_polynomial(
        e.polynomial,
        [w, input_array],
        [jax.ShapeDtypeStruct((batch_size, output_dim), dtype)],
        [jnp.repeat(jnp.arange(num_species_total), num_species), None, None],
    )

    result = np.asarray(result, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    match dtype:
        case jnp.float16 | jnp.bfloat16:
            atol, rtol = 1e-2, 1e-2
        case jnp.float32:
            atol, rtol = 1e-3, 1e-3
        case jnp.float64:
            atol, rtol = 1e-6, 1e-6
    np.testing.assert_allclose(result, ref, rtol=rtol, atol=atol)
