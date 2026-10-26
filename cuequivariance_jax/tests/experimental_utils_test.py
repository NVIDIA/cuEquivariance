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

import builtins
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cuequivariance_jax.experimental import utils

import cuequivariance as cue
import cuequivariance_jax as cuex


@pytest.mark.parametrize(
    ("output_activation", "expected_last_activation"),
    [(True, jnp.tanh), (False, None), (jnp.square, jnp.square)],
)
def test_mlp_activation_dispatch(
    output_activation, expected_last_activation, monkeypatch, capsys
):
    pytest.importorskip("flax.linen")

    monkeypatch.setattr(utils.cuex, "normalize_function", lambda function: function)
    x = jnp.asarray([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]])
    mlp = utils.MultiLayerPerceptron(
        list_neurons=(4, 2),
        act=jnp.tanh,
        output_activation=output_activation,
        with_bias=False,
    )

    variables = mlp.init(jax.random.PRNGKey(0), x, verbose=True)
    params = variables["params"]
    hidden = jnp.sqrt(1.0 / x.shape[-1]) * (x @ params["Dense_0"]["kernel"])
    hidden = jnp.tanh(hidden)
    expected = jnp.sqrt(1.0 / hidden.shape[-1]) * (hidden @ params["Dense_1"]["kernel"])
    if expected_last_activation is not None:
        expected = expected_last_activation(expected)

    actual = mlp.apply(variables, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == (2, 2)
    assert "MLP matrices: in -> [(3, 4), (4, 2)] -> out" in capsys.readouterr().out


def test_optional_flax_import_fallback(monkeypatch):
    """Import-only fallback contract; an executable MLP still requires Flax."""
    real_import = builtins.__import__

    def import_without_flax(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flax.linen":
            raise ImportError("simulate the optional Flax dependency being absent")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_flax)
    spec = importlib.util.spec_from_file_location(
        "_experimental_utils_without_flax", utils.__file__
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert issubclass(module.MultiLayerPerceptron, module.nn.Module)

    def identity(value):
        return value

    assert module.nn.compact(identity) is identity


def test_smooth_scalar_utilities():
    x = jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        utils.sus(x),
        [0.0, 0.0, 0.0, np.exp(-1.0), np.exp(-0.5)],
        rtol=4 * np.finfo(np.float32).eps,
    )

    envelope_x = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        utils.soft_envelope(
            envelope_x, x_max=2.0, arg_multiplicator=2.0, value_at_origin=1.2
        ),
        [1.2, 1.2 * np.exp(-0.5), 0.0, 0.0],
        atol=1e-6,
    )

    np.testing.assert_allclose(
        utils.smooth_bump(x), [0.0, 0.0, 1.14136, 0.0, 0.0], atol=1e-6
    )


@pytest.mark.parametrize(
    ("indices", "indices_are_sorted", "expected"),
    [
        (
            [2, 0, 2],
            False,
            [
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0],
                [8.0, 10.0, 12.0],
                [0.0, 0.0, 0.0],
            ],
        ),
        (
            [0, 1, 2],
            True,
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
            ],
        ),
    ],
)
def test_gather_scatters_representation_rows(indices, indices_are_sorted, expected):
    irreps = cue.Irreps("SO3", "1")
    x = cuex.RepArray(
        {1: irreps},
        jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        layout=cue.ir_mul,
    )

    result = utils.gather(
        jnp.asarray(indices), x, n=4, indices_are_sorted=indices_are_sorted
    )

    assert result.reps == x.reps
    np.testing.assert_allclose(result.array, expected)

    invalid = cuex.RepArray({0: irreps}, x.array, layout=cue.ir_mul)
    with pytest.raises(AssertionError):
        utils.gather(jnp.asarray(indices), invalid, n=4)
