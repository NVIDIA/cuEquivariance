# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright 2023 Mario Geiger
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import *

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

import cuequivariance as cue
import cuequivariance_jax as cuex

ActFn = Callable[[float], float] | Callable[[jax.Array], jax.Array]


def soft_odd(x: jax.Array) -> jax.Array:
    """Smooth odd function that can be used as activation function for odd scalars.

    .. math::

        x (1 - e^{-x^2})

    Note:
        Odd scalars (l=0 and p=-1) has to be activated by functions with well defined parity:

        * even (:math:`f(-x)=f(x)`)
        * odd (:math:`f(-x)=-f(x)`).
    """
    return (1 - jnp.exp(-(x**2))) * x


def normalspace(n: int) -> jax.Array:
    r"""Sequence of normally distributed numbers :math:`x_i` for :math:`i=1, \ldots, n` such that

    .. math::

        \int_{-\infty}^{x_i} \phi(x) dx = \frac{i}{n+1}

    where :math:`\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}` is the normal distribution.

    Args:
        n (int): Number of points

    Returns:
        jax.Array: Sequence of normally distributed numbers
    """
    return jnp.sqrt(2) * jsp.erfinv(jnp.linspace(-1.0, 1.0, n + 2)[1:-1])


def normalize_function(phi: ActFn) -> ActFn:
    r"""Normalize a function, :math:`\psi(x)=\phi(x)/c` where :math:`c` is the normalization constant such that

    .. math::

        \int_{-\infty}^{\infty} \psi(x)^2 \frac{e^{-x^2/2}}{\sqrt{2\pi}} dx = 1
    """
    with jax.ensure_compile_time_eval():
        # k = jax.random.key(0)
        # x = jax.random.normal(k, (1_000_000,))
        x = normalspace(1_000_001)
        c = jnp.mean(jax.vmap(phi)(x) ** 2) ** 0.5
        c = c.item()

        if c < 1e-5:
            raise ValueError("Cannot normalize the zero function")

        if jnp.allclose(c, 1.0):
            return phi
        else:

            def rho(x):
                return phi(x) / c

            return rho


def parity_function(phi: ActFn) -> int:
    with jax.ensure_compile_time_eval():
        x = jnp.linspace(0.0, 10.0, 256)

        a1, a2 = jax.vmap(phi)(x), jax.vmap(phi)(-x)
        if jnp.max(jnp.abs(a1 - a2)) < 1e-5:
            return 1
        elif jnp.max(jnp.abs(a1 + a2)) < 1e-5:
            return -1
        else:
            return 0


def scalar_activation(
    input: cuex.IrrepsArray,
    acts: ActFn | list[ActFn | None] | dict[cue.Irrep, ActFn],
    *,
    normalize_act: bool = True,
) -> cuex.IrrepsArray:
    r"""Apply activation functions to the scalars of an `IrrepsArray`.
    The activation functions are by default normalized.
    """
    input = cuex.as_irreps_array(input)
    assert isinstance(input, cuex.IrrepsArray)
    assert input.is_simple

    if isinstance(acts, dict):
        acts = [acts.get(ir, None) for mul, ir in input.irreps()]
    if callable(acts):
        acts = [acts] * len(input.irreps())

    assert len(input.irreps()) == len(acts), (input.irreps(), acts)

    segments = []

    irreps_out = []
    for (mul, ir), x, act in zip(input.irreps(), input.segments(), acts):
        mul: int
        ir: cue.Irrep
        x: jax.Array

        if act is not None:
            assert np.all(np.imag(ir.H) == 0), "TODO: support complex scalars"
            assert ir.dim == 1, "Only scalars are supported"

            if normalize_act:
                act = normalize_function(act)

            p_act = parity_function(act)

            if np.allclose(ir.H, 1):
                # if the input is even, we can apply any activation function
                ir_out = ir
            else:
                if p_act == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
                    )
                elif p_act == 1:
                    ir_out = ir.trivial()
                elif p_act == -1:
                    ir_out = ir

            irreps_out.append((mul, ir_out))
            segments.append(act(x))
        else:
            irreps_out.append((mul, ir))
            segments.append(x)

    irreps_out = cue.Irreps(input.irreps(), irreps_out)
    return cuex.from_segments(
        irreps_out, segments, input.shape, input.layout, input.dtype
    )
