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
import builtins
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cuequivariance_jax.flax_linen.layer_norm import LayerNorm

import cuequivariance as cue
import cuequivariance_jax as cuex


def _layer_norm_input(layout):
    irreps = cue.Irreps(cue.O3, "2x1o + 2x0e")
    vector = jnp.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0]],
        ]
    )
    scalar = jnp.array([[[0.0, 0.0]], [[6.0, 8.0]]])

    if layout == cue.mul_ir:
        vector = jnp.swapaxes(vector, -1, -2)
        scalar = jnp.swapaxes(scalar, -1, -2)

    return cuex.from_segments(irreps, [vector, scalar], shape=(2, -1), layout=layout)


def _apply_layer_norm(layer_norm_type, input_, epsilon):
    if hasattr(layer_norm_type, "init"):
        layer_norm = layer_norm_type(epsilon=epsilon)
        variables = layer_norm.init(jax.random.key(0), input_)
        return layer_norm.apply(variables, input_)
    layer_norm = layer_norm_type()
    layer_norm.epsilon = epsilon
    return layer_norm(input_)


def _assert_normalized(output, input_, layout, epsilon):
    denominator = jnp.sqrt(jnp.array(12.5, dtype=input_.dtype)) + epsilon
    scalar_denominator = jnp.sqrt(jnp.array(50.0, dtype=input_.dtype)) + epsilon
    expected_vector = (
        jnp.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0]],
            ]
        )
        / denominator
    )
    expected_scalar = jnp.array([[[0.0, 0.0]], [[6.0, 8.0]]]) / scalar_denominator
    if layout == cue.mul_ir:
        expected_vector = jnp.swapaxes(expected_vector, -1, -2)
        expected_scalar = jnp.swapaxes(expected_scalar, -1, -2)

    assert output.irreps == input_.irreps
    assert output.layout == layout
    assert output.shape == input_.shape
    assert output.dtype == input_.dtype
    np.testing.assert_allclose(output.segments[0], expected_vector, rtol=1e-6)
    np.testing.assert_allclose(output.segments[1], expected_scalar, rtol=1e-6)


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_layer_norm_normalizes_irrep_segments_in_both_layouts(layout):
    epsilon = 0.1
    input_ = _layer_norm_input(layout)

    output = _apply_layer_norm(LayerNorm, input_, epsilon)

    _assert_normalized(output, input_, layout, epsilon)


def test_layer_norm_rejects_a_rep_array_that_is_not_an_irreps_array():
    input_ = cuex.RepArray({0: cue.SO3(1)}, jnp.ones((3,)))

    with pytest.raises(AssertionError):
        _apply_layer_norm(LayerNorm, input_, 0.1)


def test_layer_norm_import_falls_back_when_flax_linen_is_unavailable(monkeypatch):
    from cuequivariance_jax.flax_linen import layer_norm as layer_norm_module

    spec = importlib.util.spec_from_file_location(
        "_layer_norm_without_flax", layer_norm_module.__file__
    )
    fallback_module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def reject_flax_linen(name, *args, **kwargs):
        if name == "flax.linen":
            raise ImportError("Flax is unavailable for this import")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", reject_flax_linen)
        spec.loader.exec_module(fallback_module)

    assert fallback_module.LayerNorm.__bases__ == (fallback_module.nn.Module,)
    layout = cue.ir_mul
    epsilon = 0.1
    input_ = _layer_norm_input(layout)
    output = _apply_layer_norm(fallback_module.LayerNorm, input_, epsilon)

    _assert_normalized(output, input_, layout, epsilon)


@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
def test_layer_norm_zero_input_has_finite_gradient(layout):
    input_ = _layer_norm_input(layout)
    zeros = jnp.zeros_like(input_.array)

    def forward(array):
        return _apply_layer_norm(
            LayerNorm, cuex.RepArray(input_.irreps, array, layout), 0.0
        ).array

    np.testing.assert_array_equal(forward(zeros), np.zeros(input_.shape))
    gradient = jax.grad(lambda array: jnp.sum(forward(array)))(zeros)
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(gradient, jnp.ones_like(gradient))
