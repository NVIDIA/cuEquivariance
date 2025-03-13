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
import logging
import re

import jax
import jax.numpy as jnp
import numpy as np

import cuequivariance as cue
from cuequivariance_jax.segmented_polynomials.utils import reshape

logger = logging.getLogger(__name__)


def sanitize_string(s):
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    if s == "" or s[0].isdigit():
        s = "_" + s
    return s


def segmented_polynomial_ops_impl(
    inputs: list[jax.Array],  # shape (*batch_sizes, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    buffer_index: tuple[tuple[int, ...], ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
) -> tuple[list[jax.Array] | None, str]:
    def log(msg: str):
        msg = f"[{name}] {msg}"
        logger.info(msg)
        return None, msg

    buffer_index = np.array(buffer_index)
    num_batch_axes = buffer_index.shape[1]
    assert polynomial.num_inputs + len(outputs_shape_dtype) == buffer_index.shape[0]
    assert polynomial.num_outputs == len(outputs_shape_dtype)

    # We don't use the feature that indices can index themselves
    buffer_index = np.concatenate(
        [buffer_index, np.full((len(indices), num_batch_axes), -1, np.int32)]
    )

    buffers = list(inputs) + list(outputs_shape_dtype)
    for b in buffers:
        assert b.ndim == num_batch_axes + 1, (
            f"Buffer {b.shape} must have {num_batch_axes} batch axes"
        )

    # Reshape buffers to 3D by using the STP informations
    for ope, stp in polynomial.tensor_products:
        if len(stp.subscripts.modes()) != 1:
            return log(f"Unsupported STP: {stp}")
        if not stp.all_same_segment_shape():
            return log(f"Unsupported STP: {stp}")

        for i, operand in zip(ope.buffers, stp.operands):
            b = buffers[i]
            shape = b.shape[:-1] + (operand.num_segments, operand.segment_size)
            if b.ndim == num_batch_axes + 1:
                b = buffers[i] = reshape(b, shape)
            if b.shape != shape:
                return log(f"Shape mismatch: {b.shape} != {shape} for {i} {stp} {ope}")

    if not all(b.ndim == num_batch_axes + 2 for b in buffers):
        return log("All buffers must be used")

    for b in buffers:
        if b.dtype.type not in {jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16}:
            return log(f"Unsupported buffer type: {b.dtype}")

    for i in indices:
        if i.dtype.type not in {jnp.int32, jnp.int64}:
            return log(f"Unsupported index type: {i.dtype}")

    if len({b.shape[-1] for b in buffers}.union({1})) != 2:
        return log(f"Buffer shapes not compatible {[b.shape for b in buffers]}")

    math_dtype = jnp.dtype(math_dtype)
    if math_dtype.type not in {jnp.float32, jnp.float64}:
        return log(f"Unsupported math_dtype: {math_dtype}")

    try:
        from cuequivariance_ops_jax import (
            Operation,
            Path,
            tensor_product_uniform_1d_jit,
        )
    except ImportError as e:
        return log(f"cuequivariance_ops_jax is not installed: {e}")

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
        list(indices),
        buffer_index,
        operations=operations,
        paths=paths,
        math_dtype=math_dtype,
        name=sanitize_string(name),
    )
    return [
        jnp.reshape(x, x.shape[:-2] + (x.shape[-2] * x.shape[-1],)) for x in outputs
    ], ""
