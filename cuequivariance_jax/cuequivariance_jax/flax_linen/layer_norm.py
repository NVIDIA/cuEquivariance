# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from typing import *

import jax
import jax.numpy as jnp

import cuequivariance as cue
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


class LayerNorm(nn.Module):
    epsilon: float = 0.01

    @nn.compact
    def __call__(self, input: cuex.IrrepsArray) -> cuex.IrrepsArray:
        assert input.is_simple()

        def rms(v: jax.Array) -> jax.Array:
            # v [..., ir, mul] or [..., mul, ir]

            match input.layout:
                case cue.ir_mul:
                    axis_mul, axis_ir = -1, -2
                case cue.mul_ir:
                    axis_mul, axis_ir = -2, -1

            sn = jnp.sum(v**2, axis=axis_ir, keepdims=True)
            msn = jnp.mean(sn, axis=axis_mul, keepdims=True)  # [..., 1, 1]
            rmsn = jnp.sqrt(jnp.where(msn == 0.0, 1.0, msn))
            return rmsn

        return cuex.from_segments(
            input.irreps(),
            [x / (rms(x) + self.epsilon) for x in input.segments()],
            input.shape,
            input.layout,
            input.dtype,
        )
