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
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import cuequivariance as cue
from cuequivariance_jax.segmented_polynomials.utils import reshape

logger = logging.getLogger(__name__)


@dataclass
class Ok:
    outputs: list[jax.Array]

    def unwrap(self) -> list[jax.Array]:
        return self.outputs

    def unwrap_or(self, default):
        return self.outputs

    def is_ok(self) -> bool:
        return True


@dataclass
class Err:
    msg: str

    def unwrap(self):
        raise ValueError(self.msg)

    def unwrap_or(self, default):
        return default

    def is_ok(self) -> bool:
        return False


Result = Ok | Err


def sanitize_string(s):
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    if s == "" or s[0].isdigit():
        s = "_" + s
    return s


def segmented_polynomial_ops_impl(
    inputs: list[jax.Array],  # shape (batch_size, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    buffer_index: list[int],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
) -> Result:
    def make_error(msg: str) -> Err:
        logger.info(f"[{name}] {msg}")
        return Err(msg)

    assert polynomial.num_inputs == len(buffer_index) - len(outputs_shape_dtype)
    assert polynomial.num_outputs == len(outputs_shape_dtype)

    buffers = list(inputs) + list(outputs_shape_dtype)
    for b in buffers:
        assert b.ndim == 2, f"Buffer {b.shape} must be 2D"

    # Reshape buffers to 3D by using the STP informations
    for ope, stp in polynomial.operations:
        if len(stp.subscripts.modes()) != 1:
            return make_error(f"Unsupported STP: {stp}")
        if not stp.all_same_segment_shape():
            return make_error(f"Unsupported STP: {stp}")

        for i, operand in zip(ope.buffers, stp.operands):
            b = buffers[i]
            shape = (b.shape[0], operand.num_segments, operand.segment_size)
            if b.ndim == 2:
                b = buffers[i] = reshape(b, shape)
            if b.shape != shape:
                return make_error(
                    f"Shape mismatch: {b.shape} != {shape} for {i} {stp} {ope}"
                )

    for b in buffers:
        if b.dtype.type not in {jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16}:
            return make_error(f"Unsupported buffer type: {b.dtype}")

    for i in indices:
        # TODO: this restriction will be removed by MR109
        if i.dtype.type != jnp.int32:
            return make_error(f"Unsupported index type: {i.dtype}")

    if not all(b.ndim == 3 for b in buffers):
        return make_error("All buffers must be used")

    if len({b.shape[2] for b in buffers}.union({1})) != 2:
        return make_error(f"Buffer shapes not compatible {[b.shape for b in buffers]}")

    # TODO: this restriction will be removed by MR109
    if max(b.shape[2] for b in buffers) % 32 != 0:
        return make_error(
            f"Extend must be a multiple of 32, got {[b.shape for b in buffers]}"
        )

    math_dtype = jnp.dtype(math_dtype)
    if math_dtype.type not in {jnp.float32, jnp.float64}:
        return make_error(f"Unsupported math_dtype: {math_dtype}")

    batch_size = 1
    for i, b in zip(buffer_index, buffers):
        if i >= 0:
            batch_size = indices[i].shape[0]
        elif b.shape[0] != 1:
            batch_size = b.shape[0]

    # TODO: this restriction will be removed by MR109
    for i, b in zip(
        buffer_index[polynomial.num_inputs :], buffers[polynomial.num_inputs :]
    ):
        if b.dtype.type not in {jnp.float32, jnp.float64}:
            if i >= 0 or b.shape[0] != batch_size:
                return make_error(
                    f"Output buffer {b.shape} of type {b.dtype} and buffer index {i} is not supported"
                )

    try:
        from cuequivariance_ops_jax import (
            Operation,
            Path,
            tensor_product_uniform_1d_jit,
        )
    except ImportError as e:
        return make_error(f"cuequivariance_ops_jax is not installed: {e}")

    operations = []
    paths = []
    for ope, stp in polynomial.operations:
        operations.append(Operation(ope.buffers, len(paths), stp.num_paths))
        for path in stp.paths:
            paths.append(Path(path.indices, path.coefficients.item()))

    logger.info(f"Using the uniform 1d kernel for '{name}' 🚀\n" + str(polynomial))
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
    return Ok([jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])) for x in outputs])
