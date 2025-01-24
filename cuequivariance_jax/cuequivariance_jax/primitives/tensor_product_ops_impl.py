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
import math

import jax
import jax.numpy as jnp

import cuequivariance as cue
from cuequivariance.tensor_product_execution import InBuffer

logger = logging.getLogger(__name__)


def tensor_product_ops_impl(
    *inputs: jax.Array,  # input buffers
    output_shapes: tuple[tuple[int, ...] | None, ...],  # shapes of the operands
    d: cue.SegmentedTensorProduct,
    exe: cue.TensorProductExecution,
    **options,
) -> tuple[jax.Array, ...]:  # output buffers
    assert exe.max_out_buffer + 1 == len(exe.out_buffers)

    detail_str = f"\n{d}\n{exe}".replace("\n", "\n  | ")

    if not d.all_same_segment_shape():
        logger.info("🛶 can't use tensor_product_uniform_1d for" + detail_str)
        raise NotImplementedError()

    try:
        from cuequivariance_ops_jax import tensor_product_uniform_1d
    except ImportError:
        logger.info("🛶 can't import cuequivariance_ops_jax")
        raise NotImplementedError()

    modes = d.subscripts.modes()
    if len(modes) > 1:
        logger.info("🛶 can't use tensor_product_uniform_1d for" + detail_str)
        raise NotImplementedError()

    if len(modes) == 1:
        dims: set[int] = d.get_dims(modes[0])
        if len(dims) != 1:
            logger.info("🛶 can't use tensor_product_uniform_1d for" + detail_str)
            raise NotImplementedError()

    batch_size = 1
    for shape in [input.shape[:-1] for input in inputs] + [
        shape for shape in output_shapes if shape is not None
    ]:
        n = math.prod(shape)
        if n > 1:
            if n != batch_size and batch_size != 1:
                logger.info("🛶 can't use tensor_product_uniform_1d for" + detail_str)
                raise NotImplementedError()
            batch_size = n

    reshaped_inputs = []
    for index, input in enumerate(inputs):
        operands = {
            (d.operands[op].size, d.operands[op].num_segments)
            for op in exe.get_in_buffer_operands(index)
        }
        if len(operands) != 1:
            logger.info("🛶 can't use tensor_product_uniform_1d for" + detail_str)
            raise NotImplementedError()
        size, num_segments = operands.pop()
        reshaped_inputs.append(
            input.reshape(
                (math.prod(input.shape[:-1]), num_segments, size // num_segments)
            )
        )

    output_operands = []
    outputs = []
    for index in exe.out_buffers:
        operands = exe.get_out_buffer_operands(index)
        if len(operands) != 1:
            logger.info("🛶 can't use tensor_product_uniform_1d for" + detail_str)
            raise NotImplementedError()
        ope = operands.pop()
        size, num_segments = d.operands[ope].size, d.operands[ope].num_segments

        output_operands.append(ope)
        outputs.append(
            jnp.zeros(
                (math.prod(output_shapes[ope]), num_segments, size // num_segments),
                dtype=options["dtype_output"],
            )
        )

    logger.info("🎉 use tensor_product_uniform_1d for" + detail_str)

    outputs = tensor_product_uniform_1d(
        options["dtype_math"],
        [ope.num_segments for ope in d.operands],
        [path.indices for path in d.paths],
        [path.coefficients.item() for path in d.paths],
        reshaped_inputs,
        outputs,
        [
            tuple(
                int(b) if isinstance(b, InBuffer) else -1 - int(b) for b in computation
            )
            for computation in exe.computations
        ],
    )

    outputs = [
        output.reshape(output_shapes[ope] + (-1,))
        for ope, output in zip(output_operands, outputs)
    ]
    return tuple(outputs)
