# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import jax.numpy as jnp


def naive_batching_rule(
    primitive: jax.extend.core.Primitive,
    batched_inputs: tuple[jax.Array, ...],
    batch_axes: tuple[int | None, ...],
    **kwargs,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    """Generic naive batching rule for primitives that only support single batch dimensions.

    This batching rule handles vmap by:
    1. Moving all batch axes to position 0
    2. Expanding inputs with batch size 1 to the vmap batch size
    3. Fusing the vmap dimension with the native batch dimension
    4. Calling the primitive with fused batch dimension
    5. Unfusing the output dimensions back to separate vmap and native batch

    Args:
        primitive: The JAX primitive to call
        batched_inputs: Input arrays with vmap batch dimension
        batch_axes: Batch axis positions for each input (None if not batched)
        **kwargs: Additional keyword arguments to pass to the primitive

    Returns:
        Tuple of (outputs, output_batch_axes) where output_batch_axes are all 0

    Note:
        This is a "naive" implementation because it uses jnp.broadcast_to to expand
        inputs with batch size 1, which can be memory-inefficient. Ideally, the
        backend primitive should natively support mixed batch sizes.
    """

    # Move batch axes to position 0
    def prepare(input: jax.Array, axis: int | None) -> jax.Array:
        if axis is None:
            return jnp.expand_dims(input, 0)
        else:
            return jnp.moveaxis(input, axis, 0)

    batched_inputs = [
        prepare(input, axis) for input, axis in zip(batched_inputs, batch_axes)
    ]

    # Determine the new batch dimension and extract native batch sizes
    new_dim = 1
    old_dim = 1

    for x in batched_inputs:
        if x.shape[0] != 1:
            assert new_dim in (1, x.shape[0])
            new_dim = x.shape[0]
        if x.shape[1] != 1:
            assert old_dim in (1, x.shape[1])
            old_dim = x.shape[1]

    expanded_inputs = []
    for x in batched_inputs:
        x = jnp.broadcast_to(x, (new_dim, old_dim) + x.shape[2:])
        x = x.reshape((new_dim * old_dim,) + x.shape[2:])
        expanded_inputs.append(x)

    # Call the primitive
    outputs = primitive.bind(*expanded_inputs, **kwargs)

    # Unfuse batch dimensions
    unfused_outputs = []
    for out in outputs:
        unfused_out = out.reshape((new_dim, old_dim) + out.shape[1:])
        unfused_outputs.append(unfused_out)

    # All outputs have batch axis at position 0
    return tuple(unfused_outputs), (0,) * len(unfused_outputs)
