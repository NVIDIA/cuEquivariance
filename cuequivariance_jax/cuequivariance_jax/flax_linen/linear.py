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

import jax

import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance import descriptors
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
    """
    Equivariant linear layer.

    Args:
        irreps_out (Irreps): The output irreps. (The input irreps are inferred from the input.)
        layout (IrrepsLayout): The layout of the output irreps.
        force (bool): If False, the output irreps are filtered to contain only the reachable irreps from the input.
    """

    irreps_out: cue.Irreps | str
    layout: cue.IrrepsLayout | None = None
    force: bool = False
    kernel_init: nn.initializers.Initializer = jax.random.normal

    @nn.compact
    def __call__(
        self, input: cuex.RepArray, algorithm: str = "sliced"
    ) -> cuex.RepArray:
        assert input.is_irreps_array()

        irreps_out = cue.Irreps(self.irreps_out)
        layout_out = cue.IrrepsLayout.as_layout(self.layout)

        assert_same_group(input.irreps, irreps_out)
        if not self.force:
            irreps_out = irreps_out.filter(keep=input.irreps)

        e = descriptors.linear(input.irreps, irreps_out)
        e = e.change_layout([cue.ir_mul, input.layout, layout_out])

        # Flattening mode i does slow down the computation a bit
        if algorithm != "sliced":
            e = e.flatten_modes("i")

        w = self.param("w", self.kernel_init, (e.operands[0].dim,), input.dtype)

        return cuex.equivariant_tensor_product(
            e, w, input, precision=jax.lax.Precision.HIGH, algorithm=algorithm
        )
