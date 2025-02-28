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
import logging
import re

import jax
import jax.numpy as jnp

import cuequivariance as cue
from cuequivariance_jax.primitives.primitives_utils import reshape

logger = logging.getLogger(__name__)


def sanitize_string(s):
    return re.sub(r"[^A-Za-z_]", "", s)


def segmented_polynomial_ops_impl(
    inputs: list[jax.Array],  # shape (batch_size, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    buffer_index: list[int],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
) -> tuple[list[jax.Array] | None, str]:
    def log(msg: str):
        logger.info(f"[{name}] {msg}")
        return None, name

    assert polynomial.num_inputs == len(buffer_index) - len(outputs_shape_dtype)
    assert polynomial.num_outputs == len(outputs_shape_dtype)

    buffers = list(inputs) + list(outputs_shape_dtype)
    for b in buffers:
        assert b.ndim == 2, f"Buffer {b.shape} must be 2D"

    # Reshape buffers to 3D by using the STP informations
    for ope, stp in polynomial.tensor_products:
        if len(stp.subscripts.modes()) != 1:
            return log(f"Unsupported STP: {stp}")
        if not stp.all_same_segment_shape():
            return log(f"Unsupported STP: {stp}")

        for i, operand in zip(ope.buffers, stp.operands):
            b = buffers[i]
            shape = (b.shape[0], operand.num_segments, operand.segment_size)
            if b.ndim == 2:
                b = buffers[i] = reshape(b, shape)
            if b.shape != shape:
                return log(f"Shape mismatch: {b.shape} != {shape} for {i} {stp} {ope}")

    for b in buffers:
        if b.dtype.type not in {jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16}:
            return log(f"Unsupported buffer type: {b.dtype}")

    for i in indices:
        if i.dtype.type != jnp.int32:
            return log(f"Unsupported index type: {i.dtype}")

    if not all(b.ndim == 3 for b in buffers):
        return log("All buffers must be used")

    if len({b.shape[2] for b in buffers}.union({1})) != 2:
        return log(f"Buffer shapes not compatible {[b.shape for b in buffers]}")

    if max(b.shape[2] for b in buffers) % 32 != 0:
        return log(f"Extend must be a multiple of 32, got {[b.shape for b in buffers]}")

    math_dtype = jnp.dtype(math_dtype)
    if math_dtype.type not in {jnp.float32, jnp.float64}:
        return log(f"Unsupported math_dtype: {math_dtype}")

    batch_size = 1
    for i, b in zip(buffer_index, buffers):
        if i >= 0:
            batch_size = indices[i].shape[0]
        elif b.shape[0] != 1:
            batch_size = b.shape[0]

    # TODO: remove if the backend supports atomic operations for float16/bfloat16
    for i, b in zip(
        buffer_index[polynomial.num_inputs :], buffers[polynomial.num_inputs :]
    ):
        if b.dtype.type not in {jnp.float32, jnp.float64}:
            if i >= 0 or b.shape[0] != batch_size:
                return log(
                    f"Output buffer {b.shape} of type {b.dtype} and buffer index {i} is not supported"
                )

    try:
        from cuequivariance_ops_jax import (
            Operation,
            Path,
            tensor_product_uniform_1d_jit,
        )
    except ImportError:
        return log("cuequivariance_ops_jax is not installed")

    operations = []
    paths = []
    for ope, stp in polynomial.tensor_products:
        operations.append(Operation(ope.buffers, len(paths), stp.num_paths))
        for path in stp.paths:
            paths.append(Path(path.indices, path.coefficients.item()))

    log("Using the uniform 1d kernel of cuequivariance_ops_jax 🚀\n" + str(polynomial))
    outputs = tensor_product_uniform_1d_jit(
        buffers[: polynomial.num_inputs],
        buffers[polynomial.num_inputs :],
        indices,
        buffer_index,
        operations=operations,
        paths=paths,
        math_dtype=math_dtype,
        name=sanitize_string(name),
    )
    return [jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])) for x in outputs], ""
