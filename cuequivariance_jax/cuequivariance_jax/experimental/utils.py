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
from typing import Callable

import jax
import jax.numpy as jnp

import cuequivariance_jax as cuex

try:
    import flax.linen as nn
except ImportError:

    class nn:
        class Module:
            pass

        @staticmethod
        def compact(f):
            return f


class MultiLayerPerceptron(nn.Module):
    r"""Just a simple MLP for scalars. No equivariance here."""

    list_neurons: tuple[int, ...]
    act: Callable | None = None
    output_activation: Callable | bool = True
    with_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, verbose: bool = False) -> jax.Array:
        """Evaluate the MLP

        Args:
            x: input of shape ``[..., input_size]``

        Returns:
            output of shape ``[..., list_neurons[-1]]``
        """
        output_activation = self.output_activation

        if output_activation is True:
            output_activation = self.act
        elif output_activation is False:
            output_activation = None
        else:
            assert callable(output_activation)

        act = None if self.act is None else cuex.normalize_function(self.act)
        last_act = (
            None
            if output_activation is None
            else cuex.normalize_function(output_activation)
        )

        _matrices = []
        for i, h in enumerate(self.list_neurons):
            _matrices.append((x.shape[-1], h))
            alpha = 1 / x.shape[-1]
            d = nn.Dense(
                features=h,
                use_bias=self.with_bias,
                kernel_init=nn.initializers.normal(stddev=1.0),
                bias_init=nn.initializers.zeros,
                param_dtype=x.dtype,
            )
            x = jnp.sqrt(alpha) * d(x)

            if i < len(self.list_neurons) - 1:
                if act is not None:
                    x = act(x)
            else:
                if last_act is not None:
                    x = last_act(x)

        if verbose and self.is_initializing():
            print(f"{self.name} MLP matrices: in -> {_matrices} -> out")

        return x


def bessel(x: jax.Array, n: int, x_max: float = 1.0) -> jax.Array:
    x = jnp.asarray(x)
    assert isinstance(n, int)

    x = x[..., None]
    n = jnp.arange(1, n + 1, dtype=x.dtype)
    return jnp.sqrt(2.0 / x_max) * jnp.pi * n / x_max * jnp.sinc(n * x / x_max)


def sus(x: jax.Array) -> jax.Array:
    r"""Smooth Unit Step function.

    ``-inf->0, 0->0, 2->0.6, +inf->1``
    """
    return jnp.where(x > 0.0, jnp.exp(-1.0 / jnp.where(x > 0.0, x, 1.0)), 0.0)


def soft_envelope(
    x: jax.Array,
    x_max: float = 1.0,
    arg_multiplicator: float = 2.0,  # controls how fast it goes to zero
    value_at_origin: float = 1.2,
) -> jax.Array:
    r""":math:`C^\infty` envelope function."""
    with jax.ensure_compile_time_eval():
        cste = value_at_origin / sus(arg_multiplicator)
    return cste * sus(arg_multiplicator * (1.0 - x / x_max))


def smooth_bump(x: jax.Array) -> jax.Array:
    """non-zero (positive) between -1 and 1"""
    return 1.14136 * jnp.exp(2.0) * sus(x + 1.0) * sus(1.0 - x)


def gather(
    i: jax.Array, x: cuex.RepArray, n: int, indices_are_sorted: bool = False
) -> cuex.RepArray:
    assert 0 not in x.reps
    y = jnp.zeros((n,) + x.shape[1:], dtype=x.dtype)
    y = y.at[i].add(x.array, indices_are_sorted=indices_are_sorted)
    return cuex.RepArray(x.reps, y)
