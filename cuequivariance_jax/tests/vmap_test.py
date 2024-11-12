# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex


@cue.assume("SO3", cue.ir_mul)
def test_vmap():
    def f(x):
        return x

    x = cuex.IrrepsArray({0: "1"}, jnp.zeros((3, 2)))
    y = jax.jit(cuex.vmap(f, 1, 0))(x)
    assert y.shape == (2, 3)
    assert y.dirreps == {1: cue.Irreps("1")}
