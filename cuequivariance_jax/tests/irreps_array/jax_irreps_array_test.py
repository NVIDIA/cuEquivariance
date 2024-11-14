# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex


@cue.assume("SO3", cue.ir_mul)
def test_segments():
    x = cuex.IrrepsArray("2x0 + 1", jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    x0, x1 = x.segments()
    assert x0.shape == (1, 2)
    assert x1.shape == (3, 1)
    y = cuex.from_segments("2x0 + 1", [x0, x1], x.shape)
    assert x.dirreps == y.dirreps
    assert x.layout == y.layout
    assert jnp.allclose(x.array, y.array)


@cue.assume("SO3", cue.ir_mul)
def test_slice_by_mul():
    x = cuex.IrrepsArray("2x0 + 1", jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    x = x.slice_by_mul()[1:]
    assert x.dirreps == {0: cue.Irreps("0 + 1")}
    assert x.layout == cue.ir_mul
    assert jnp.allclose(x.array, jnp.array([1.0, 0.0, 0.0, 0.0]))
