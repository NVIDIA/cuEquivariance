# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import pytest

import cuequivariance as cue
import cuequivariance_jax as cuex


@pytest.mark.parametrize("layout_in", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("layout_out", [cue.ir_mul, cue.mul_ir])
def test_explicit_linear(layout_in, layout_out):
    try:
        import flax
    except ImportError:
        pytest.skip("flax not installed")

    x = cuex.IrrepsArray(cue.Irreps("SO3", "3x0 + 2x1"), jnp.ones((16, 9)), layout_in)
    linear = cuex.flax_linen.Linear(cue.Irreps("SO3", "2x0 + 1"), layout_out)
    w = linear.init(jax.random.key(0), x)
    y: cuex.IrrepsArray = linear.apply(w, x)
    assert y.shape == (16, 5)
    assert y.irreps() == cue.Irreps("SO3", "2x0 + 1")
    assert y.layout == layout_out


@cue.assume("SO3", cue.ir_mul)
def test_implicit_linear():
    try:
        import flax
    except ImportError:
        pytest.skip("flax not installed")

    x = cuex.IrrepsArray("3x0 + 2x1", jnp.ones((16, 9)))
    linear = cuex.flax_linen.Linear("2x0 + 1")
    w = linear.init(jax.random.key(0), x)
    y = linear.apply(w, x)
    assert y.shape == (16, 5)
    assert y.irreps() == "2x0 + 1"
