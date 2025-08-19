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
    input_batch_axes: tuple[int | None, ...],
    output_batch_axes: tuple[int, ...],
    batched_inputs: tuple[jax.Array, ...],
    vmapped_axes: tuple[int | None, ...],
    **kwargs,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    """Generic naive batching rule for primitives with flexible batch axis support."""
    assert len(vmapped_axes) == len(batched_inputs)
    assert len(input_batch_axes) == len(batched_inputs)

    vmap_size = batch_size = None
    prepared_inputs = []

    # Prepare inputs: handle batch/vmap axes
    for i, (arr, batch_axis, vmap_axis) in enumerate(
        zip(batched_inputs, input_batch_axes, vmapped_axes)
    ):
        if batch_axis is None:
            if vmap_axis is not None:
                raise ValueError(
                    f"Input {i} has vmap_axis={vmap_axis} but {primitive} has no batch axis for this input. "
                    "If you see this error, please consider opening an issue at https://github.com/NVIDIA/cuEquivariance."
                )
            prepared_inputs.append(arr)
        else:
            if vmap_axis is None:
                arr = jnp.expand_dims(arr, axis=batch_axis)
                prepared_inputs.append(arr)
                # (..., 1, batch_size, ...)
            else:
                vmap_size = vmap_size or arr.shape[vmap_axis]
                assert vmap_size == arr.shape[vmap_axis]

                arr = jnp.moveaxis(arr, vmap_axis, batch_axis)
                prepared_inputs.append(arr)
                # (..., vmap_size, batch_size, ...)
            batch_size = batch_size or arr.shape[batch_axis + 1]
            assert batch_size == arr.shape[batch_axis + 1]

    assert vmap_size is not None  # this should never happen

    # Fuse dimensions for primitive call
    final_inputs = []
    for arr, batch_axis in zip(prepared_inputs, input_batch_axes):
        if batch_axis is not None:
            shape = list(arr.shape)
            shape[batch_axis] = vmap_size
            arr = jnp.broadcast_to(arr, shape)  # (..., vmap_size, batch_size, ...)

            shape[batch_axis] = vmap_size * batch_size
            shape.pop(batch_axis + 1)
            arr = arr.reshape(shape)  # (..., vmap_size * batch_size, ...)
        final_inputs.append(arr)

    outputs = primitive.bind(*final_inputs, **kwargs)
    assert primitive.multiple_results and isinstance(outputs, (tuple, list))
    assert len(outputs) == len(output_batch_axes)

    # Unfuse output dimensions
    unfused_outputs = []
    for out, batch_axis in zip(outputs, output_batch_axes):
        # Unfuse and move vmap dimension to position 0
        shape = list(out.shape)
        shape[batch_axis : batch_axis + 1] = [vmap_size, batch_size]
        out = out.reshape(shape)
        out = jnp.moveaxis(out, batch_axis, 0)
        unfused_outputs.append(out)

    return tuple(unfused_outputs), (0,) * len(unfused_outputs)
