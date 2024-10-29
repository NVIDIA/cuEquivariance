# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import logging
import math
import warnings
from typing import *

import jax
import jax.lax
import jax.numpy as jnp

from cuequivariance import segmented_tensor_product as stp
from cuequivariance.tensor_product_execution import (
    InBuffer,
    TensorProductExecution,
)

logger = logging.getLogger(__name__)


def tensor_product_ops_impl(
    *inputs: jax.Array,
    shapes: tuple[tuple[int, ...], ...],
    d: stp.SegmentedTensorProduct,
    exe: TensorProductExecution,
    dtype_output: jnp.dtype,
    dtype_math: jnp.dtype,
    **_options,
) -> tuple[jax.Array, ...] | None:
    assert exe.max_out_buffer + 1 == len(exe.out_buffers)

    if not dtype_output in [jnp.float32, jnp.float64]:
        return None
    if not all(x.dtype == dtype_output for x in inputs):
        return None

    num_batch = math.prod(jnp.broadcast_shapes(*shapes))
    if not all(math.prod(shape) in [1, num_batch] for shape in shapes):
        return None

    if not (2 <= d.num_operands <= 7):
        return None

    for b in exe.out_buffers:
        if len({c.out_operand for c in exe.computations if c.out_buffer == b}) != 1:
            return None

    d = d.squeeze_modes()
    d = d.consolidate_paths()

    if len(d.subscripts.modes()) != 1:
        return None

    (m,) = d.subscripts.modes()
    uu = d.get_dims(m)
    u = math.gcd(*uu)
    d = d.split_mode(m, u)
    assert d.all_same_segment_shape()

    if u % 32 != 0:
        return None

    try:
        from cuequivariance_ops_jax import tensor_product_uniform_1d
    except ImportError:
        warnings.warn(
            "Unable to import cuequivariance_ops_jax.tensor_product_uniform_1d. "
            "Falling back to pure JAX implementation."
        )
        return None

    finputs = [None] * len(inputs)
    foutputs = [None] * len(exe.out_buffers)
    for c in exe.computations:
        for oid, (b, x) in zip(c.in_operands, c.map_inputs(enumerate(inputs))):
            op: stp.Operand = d.operands[oid]
            finputs[b] = jnp.reshape(
                x,
                (
                    math.prod(shapes[oid]),
                    op.num_segments,
                    u if op.subscripts == m else 1,
                ),
            )

        op = d.operands[c.out_operand]
        foutputs[c.out_buffer] = jnp.zeros(
            (
                math.prod(shapes[c.out_operand]),
                op.num_segments,
                u if op.subscripts == m else 1,
            ),
            dtype=dtype_output,
        )

    foutputs = tensor_product_uniform_1d(
        dtype_math,
        [ope.num_segments for ope in d.operands],
        [path.indices for path in d.paths],
        [float(path.coefficients) for path in d.paths],
        finputs,
        foutputs,
        [
            tuple(int(b) if isinstance(b, InBuffer) else -1 - int(b) for b in c)
            for c in exe.computations
        ],
    )

    outputs = [None] * len(exe.out_buffers)
    for c in exe.computations:
        outputs[c.out_buffer] = jnp.reshape(
            foutputs[c.out_buffer],
            shapes[c.out_operand] + (d.operands[c.out_operand].size,),
        )

    return tuple(outputs)
