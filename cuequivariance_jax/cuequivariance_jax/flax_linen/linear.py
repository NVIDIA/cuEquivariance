# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from typing import *

import jax

import cuequivariance as cue
from cuequivariance import descriptors
import cuequivariance_jax as cuex
from cuequivariance.irreps_array.misc_ui import assert_same_group

try:
    import flax.linen as nn
except ImportError:

    class nn:
        class Module:
            pass

        @staticmethod
        def compact(f):
            return f

        class initializers:
            class Initializer:
                pass


class Linear(nn.Module):
    irreps_out: cue.Irreps | str
    layout: cue.IrrepsLayout | None = None
    force: bool = False
    kernel_init: nn.initializers.Initializer = jax.random.normal

    @nn.compact
    def __call__(
        self, input: cuex.IrrepsArray, algorithm: str = "sliced"
    ) -> cuex.IrrepsArray:
        if not isinstance(input, cuex.IrrepsArray):
            raise ValueError(f"input must be of type IrrepsArray, got {type(input)}")

        assert input.is_simple()

        irreps_out = cue.Irreps(self.irreps_out)
        layout_out = cue.IrrepsLayout.as_layout(self.layout)

        assert_same_group(input.irreps(), irreps_out)
        if not self.force:
            irreps_out = irreps_out.filter(keep=input.irreps())

        e = descriptors.linear(input.irreps(), irreps_out)
        e = e.change_layout([cue.ir_mul, input.layout, layout_out])

        # Flattening mode i does slow down the computation a bit
        if algorithm != "sliced":
            e = e.flatten_modes("i")

        w = self.param("w", self.kernel_init, (e.operands[0].irreps.dim,), input.dtype)

        return cuex.equivariant_tensor_product(
            e, w, input, precision=jax.lax.Precision.HIGH, algorithm=algorithm
        )
