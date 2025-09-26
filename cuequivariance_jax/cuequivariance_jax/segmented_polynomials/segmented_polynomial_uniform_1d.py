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
import math
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from cuequivariance_jax.segmented_polynomials.utils import reshape
from packaging import version

import cuequivariance as cue


def sanitize_string(s):
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    if s == "" or s[0].isdigit():
        s = "_" + s
    return s


def execute_uniform_1d(
    inputs: list[jax.Array],  # shape (*batch_sizes, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    index_configuration: tuple[tuple[int, ...], ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
) -> list[jax.Array]:
    error_message = f"Failed to execute 'uniform_1d' method for the following polynomial:\n{polynomial}\n"

    index_configuration = np.array(index_configuration)
    num_batch_axes = index_configuration.shape[1]
    assert (
        polynomial.num_inputs + len(outputs_shape_dtype) == index_configuration.shape[0]
    )
    assert polynomial.num_outputs == len(outputs_shape_dtype)

    try:
        polynomial = polynomial.flatten_coefficient_modes()
    except ValueError as e:
        raise ValueError(
            error_message
            + f"This method does not support coefficient modes. Flattening them failed:\n{e}"
        ) from e
    assert all(d.coefficient_subscripts == "" for _, d in polynomial.operations)

    polynomial = polynomial.squeeze_modes()
    polynomial = polynomial.canonicalize_subscripts()

    def fn(op, d: cue.SegmentedTensorProduct):
        if d.subscripts.modes() == []:
            d = d.append_modes_to_all_operands("u", dict(u=1))
        return op, d

    polynomial = polynomial.apply_fn(fn)

    # We don't use the feature that indices can index themselves
    index_configuration = np.concatenate(
        [index_configuration, np.full((len(indices), num_batch_axes), -1, np.int32)]
    )

    buffers = list(inputs) + list(outputs_shape_dtype)
    for b in buffers:
        assert b.ndim == num_batch_axes + 1, (
            f"Buffer {b.shape} must have {num_batch_axes} batch axes"
        )
    for i in indices:
        assert i.ndim == num_batch_axes, (
            f"Index {i.shape} must have {num_batch_axes} batch axes"
        )

    # Special case where num_batch_axes == 0
    if num_batch_axes == 0:
        num_batch_axes = 1
        buffers = [reshape(b, (1, *b.shape)) for b in buffers]
        indices = [reshape(i, (1, *i.shape)) for i in indices]
        index_configuration = np.full((index_configuration.shape[0], 1), -1, np.int32)

    # Reshape buffers to 3D by using the STP informations
    extents = set()
    for ope, stp in polynomial.operations:
        if len(stp.subscripts.modes()) != 1:
            raise ValueError(
                error_message
                + f"The 'uniform_1d' method requires exactly one mode, but {len(stp.subscripts.modes())} modes were found in subscripts: {stp.subscripts}.\n"
                + "Resolution: Consider applying 'flatten_modes()' to the polynomial to eliminate a mode by increasing the number of segments and paths. "
                + "Please note that flattening modes with large extents may negatively impact performance."
            )
        assert stp.subscripts.modes() == ["u"], (
            "Should be the case after canonicalization"
        )
        if not stp.all_same_segment_shape():
            dims = stp.get_dims("u")
            gcd = math.gcd(*stp.get_dims("u"))
            suggestion = stp.split_mode("u", gcd)
            raise ValueError(
                error_message
                + "The 'uniform_1d' method requires all segments to have uniform shapes within each operand.\n"
                + f"Current configuration: {stp}\n"
                + "Resolution: If your mode extents share a common divisor, consider applying 'split_mode()' to create uniform segment extents. "
                + f"For mode u={dims}, the greatest common divisor is {gcd}. Applying 'split_mode()' would result in: {suggestion}"
            )

        for i, operand in zip(ope.buffers, stp.operands):
            if operand.ndim == 1:
                extents.add(operand.segment_shape[0])

            b = buffers[i]
            shape = b.shape[:num_batch_axes] + (
                operand.num_segments,
                operand.segment_size,
            )
            if b.ndim == num_batch_axes + 1:
                b = buffers[i] = reshape(b, shape)
            if b.shape != shape:
                raise ValueError(
                    f"Shape mismatch: {b.shape} != {shape} for {i} {stp} {ope}"
                )

    if len(extents) != 1:
        raise ValueError(
            f"The 'uniform_1d' method requires a single uniform mode among all the STPs of the polynomial, got u={extents}."
        )

    if not all(b.ndim == num_batch_axes + 2 for b in buffers):
        raise ValueError("All buffers must be used")

    for b in buffers:
        if b.dtype.type not in {jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16}:
            raise ValueError(f"Unsupported buffer type: {b.dtype}")

    for i in indices:
        if i.dtype.type not in {jnp.int32, jnp.int64}:
            raise ValueError(f"Unsupported index type: {i.dtype}")

    if len({b.shape[-1] for b in buffers}.union({1})) > 2:
        raise ValueError(f"Buffer shapes not compatible {[b.shape for b in buffers]}")

    if math_dtype is not None:
        supported_dtypes = {"float32", "float64", "float16", "bfloat16"}
        if math_dtype not in supported_dtypes:
            raise ValueError(
                f"method='uniform_1d' only supports math_dtype equal to {supported_dtypes}, got '{math_dtype}'."
            )
        compute_dtype = getattr(jnp, math_dtype)
    else:
        if jnp.result_type(*buffers) == jnp.float64:
            compute_dtype = jnp.float64
        else:
            compute_dtype = jnp.float32

    try:
        from cuequivariance_ops_jax import (
            Operation,
            Path,
            __version__,
            tensor_product_uniform_1d_jit,
        )
    except ImportError as e:
        raise ValueError(f"cuequivariance_ops_jax is not installed: {e}")

    if version.parse(__version__) < version.parse("0.4.0.dev"):
        message = f"cuequivariance_ops_jax version {__version__} is too old, need at least 0.4.0"
        warnings.warn(message)
        raise ValueError(message)

    operations = []
    paths = []
    for ope, stp in polynomial.operations:
        operations.append(Operation(ope.buffers, len(paths), stp.num_paths))
        for path in stp.paths:
            paths.append(Path(path.indices, path.coefficients.item()))

    outputs = tensor_product_uniform_1d_jit(
        buffers[: polynomial.num_inputs],
        buffers[polynomial.num_inputs :],
        list(indices),
        index_configuration,
        operations=operations,
        paths=paths,
        math_dtype=compute_dtype,
        name=sanitize_string(name),
    )
    return [jnp.reshape(x, y.shape) for x, y in zip(outputs, outputs_shape_dtype)]
