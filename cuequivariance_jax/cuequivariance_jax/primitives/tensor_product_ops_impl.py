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


def tensor_product_ops_impl(
    inputs: list[jax.Array],  # shape (batch_size, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    buffer_index: list[int],
    descriptors: frozenset[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    math_dtype: jnp.dtype,
    name: str,
) -> list[jax.Array] | None:
    num_inputs = len(buffer_index) - len(outputs_shape_dtype)

    buffers = list(inputs) + list(outputs_shape_dtype)
    for ope, stp in descriptors:
        if len(stp.subscripts.modes()) != 1:
            logger.info(f"Unsupported STP: {stp} for {name}")
            return None
        if not stp.all_same_segment_shape():
            logger.info(f"Unsupported STP: {stp} for {name}")
            return None

        for i, operand in zip(ope.buffers, stp.operands):
            b = buffers[i]
            shape = (b.shape[0], operand.num_segments, operand.segment_size)
            if b.ndim == 2:
                b = buffers[i] = reshape(b, shape)
            if b.shape != shape:
                logger.info(
                    f"Shape mismatch: {b.shape} != {shape} for {i} {stp} {ope} for {name}"
                )
                return None

    if not all(b.ndim == 3 for b in buffers):
        logger.info(f"All buffers must be used, for {name}")
        return None
    if len({b.shape[2] for b in buffers}.union({1})) != 2:
        logger.info(
            f"Buffer shapes not compatible {[b.shape for b in buffers]}, for {name}"
        )
        return None
    if max(b.shape[2] for b in buffers) % 32 != 0:
        logger.info(
            f"Buffer shapes not compatible {[b.shape for b in buffers]}, for {name}"
        )
        return None

    try:
        from cuequivariance_ops_jax import (
            Operation,
            Path,
            tensor_product_uniform_1d_jit,
        )
    except ImportError:
        logger.info("cuequivariance_ops_jax is not installed")
        return None

    operations = []
    paths = []
    for ope, stp in descriptors:
        operations.append(Operation(ope.buffers, len(paths), stp.num_paths))
        for path in stp.paths:
            paths.append(Path(path.indices, path.coefficients.item()))

    logger.info(f"Using cuequivariance_ops_jax for {name}")
    outputs = tensor_product_uniform_1d_jit(
        buffers[:num_inputs],
        buffers[num_inputs:],
        indices,
        buffer_index,
        operations=operations,
        paths=paths,
        math_dtype=math_dtype,
        name=sanitize_string(name),
    )
    return [jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])) for x in outputs]
