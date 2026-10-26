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

import builtins
import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import cuequivariance as cue
import cuequivariance_jax as cuex


def test_spherical_harmonics_uses_epsilon_in_vector_normalization():
    x = jnp.asarray([[3.0, 4.0, 0.0]])
    no_epsilon = cuex.nnx.SphericalHarmonics(max_degree=1, eps=0.0)
    with_epsilon = cuex.nnx.SphericalHarmonics(max_degree=1, eps=0.5)

    y0 = no_epsilon(x)
    y_eps = with_epsilon(x)
    vector_ir = next(ir for _, ir in no_epsilon.irreps_out if ir.dim == 3)
    scalar_ir = next(ir for _, ir in no_epsilon.irreps_out if ir.dim == 1)
    np.testing.assert_allclose(y0[scalar_ir], 1.0)
    np.testing.assert_allclose(y_eps[scalar_ir], 1.0)
    np.testing.assert_allclose(jnp.sum(y0[vector_ir] ** 2), 3.0, rtol=1e-6)
    zero = with_epsilon(jnp.zeros_like(x))
    np.testing.assert_array_equal(zero[vector_ir], jnp.zeros_like(zero[vector_ir]))

    np.testing.assert_allclose(
        y_eps[vector_ir], y0[vector_ir] * (5.0 / np.sqrt(25.25)), rtol=1e-6
    )


@pytest.mark.parametrize("skip_scalars", [True, False])
def test_irreps_normalize_scales_skipped_scalars_and_normalizes_vectors(skip_scalars):
    scalar_ir = cue.Irreps(cue.O3, "0e")[0].ir
    vector_ir = cue.Irreps(cue.O3, "1o")[0].ir
    x = {
        scalar_ir: jnp.asarray([[[1.0], [3.0]]]),
        vector_ir: jnp.asarray([[[3.0, 0.0, 4.0], [0.0, 0.0, 2.0]]]),
    }

    y = cuex.nnx.IrrepsNormalize(eps=0.5, scale=2.0, skip_scalars=skip_scalars)(x)

    scalar_factor = 2.0 if skip_scalars else 2.0 / np.sqrt(5.25)
    np.testing.assert_allclose(y[scalar_ir], scalar_factor * x[scalar_ir])
    np.testing.assert_allclose(
        y[vector_ir], x[vector_ir] * (2.0 / np.sqrt(14.75)), rtol=1e-6
    )


def test_irreps_linear_chooses_input_or_output_irrep_type():
    flax_nnx = pytest.importorskip("flax.nnx")
    empty = cue.Irreps(cue.O3, "")
    scalar = cue.Irreps(cue.O3, "0e")

    from_input = cuex.nnx.IrrepsLinear(scalar, empty, rngs=flax_nnx.Rngs(0))
    from_output = cuex.nnx.IrrepsLinear(empty, scalar, rngs=flax_nnx.Rngs(0))

    assert from_input._irrep_cls is type(scalar[0].ir)
    assert from_output._irrep_cls is type(scalar[0].ir)
    scalar_ir = scalar[0].ir
    assert from_input({scalar_ir: jnp.ones((2, 1, 1))}) == {}
    layer = cuex.nnx.IrrepsLinear(
        cue.Irreps(cue.O3, "2x0e"), scalar, scale=2.0, rngs=flax_nnx.Rngs(0)
    )
    layer.w[str(scalar_ir)][...] = jnp.array([[1.0], [-3.0]])
    output = layer({scalar_ir: jnp.array([[[2.0], [-0.5]]])})
    np.testing.assert_allclose(output[scalar_ir], [[[7.0 / np.sqrt(2)]]], rtol=1e-6)
    with pytest.raises(ValueError, match="requires non-empty irreps"):
        cuex.nnx.IrrepsLinear(empty, empty, rngs=flax_nnx.Rngs(0))


def test_nnx_import_without_flax_defines_fallback_classes(monkeypatch):
    module_name = "cuequivariance_jax._nnx_without_flax_test"
    source_path = Path(cuex.nnx.__file__)
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None

    real_import = builtins.__import__
    blocked_imports = []

    def import_without_flax(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flax" and globals and globals.get("__name__") == module_name:
            blocked_imports.append(name)
            raise ImportError("Flax disabled for fallback test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_flax)
    fallback_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = fallback_module
    try:
        spec.loader.exec_module(fallback_module)
    finally:
        sys.modules.pop(module_name, None)

    assert blocked_imports == ["flax"]
    assert all(
        isinstance(getattr(fallback_module.nnx, name), type)
        for name in ("Module", "Rngs", "Param", "Dict", "List")
    )

    scalar_ir = cue.Irreps(cue.O3, "0e")[0].ir
    y = fallback_module.IrrepsNormalize(eps=0.0, scale=3.0, skip_scalars=True)(
        {scalar_ir: jnp.ones((1, 1, 1))}
    )
    np.testing.assert_array_equal(y[scalar_ir], 3.0)
