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
import os
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from cuequivariance_jax.segmented_polynomials.utils import group_by_index, reshape
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
    options: dict,
    name: str,
) -> list[jax.Array]:
    error_message = f"Failed to execute 'uniform_1d' method for the following polynomial:\n{polynomial}\n"

    index_configuration = np.array(index_configuration)
    num_batch_axes = index_configuration.shape[1]
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

    if polynomial.num_inputs + len(outputs_shape_dtype) == index_configuration.shape[0]:
        index_configuration = np.concatenate(
            [index_configuration, np.full((len(indices), num_batch_axes), -1, np.int32)]
        )

    assert (
        polynomial.num_inputs + len(outputs_shape_dtype) + len(indices)
        == index_configuration.shape[0]
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

    if "math_dtype" in options:
        supported_dtypes = {"float32", "float64", "float16", "bfloat16"}
        math_dtype = options["math_dtype"]
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

    if options.get("auto_deterministic_indexing"):
        outputs = deterministic_indexing_grouped(
            buffers[: polynomial.num_inputs],
            buffers[polynomial.num_inputs :],
            list(indices),
            index_configuration,
            operations=operations,
            paths=paths,
            math_dtype=compute_dtype,
            name=sanitize_string(name),
        )
    else:
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


def deterministic_indexing_grouped(
    inputs: list[jax.Array],
    outputs: list[jax.Array],
    indices: list[jax.Array],
    index_configuration: np.ndarray,
    operations: list,
    paths: list,
    math_dtype: jnp.dtype,
    name: str,
):
    from collections import defaultdict

    from cuequivariance_ops_jax import Operation

    ni, no = len(inputs), len(outputs)
    assert index_configuration.shape[0] == ni + no + len(indices)
    assert np.all(index_configuration[ni + no :] == -1)

    output_configs = index_configuration[ni : ni + no]
    unique_configs, group_assignments = np.unique(
        output_configs, axis=0, return_inverse=True
    )

    groups = defaultdict(list)
    for output_idx, group_idx in enumerate(group_assignments):
        groups[group_idx].append(output_idx)

    if os.environ.get("CUEQUIVARIANCE_DEBUG_UNIFORM_1D"):
        print(f"\n{'=' * 80}")
        print(f"🎯 deterministic_indexing_grouped: {name}")
        print(f"{'=' * 80}")
        print(
            f"📊 {ni} inputs, {no} outputs, {len(indices)} indices, {len(operations)} ops, {len(paths)} paths"
        )
        print(f"🔢 Input shapes:  {[tuple(x.shape) for x in inputs]}")
        print(f"📦 Output shapes: {[tuple(x.shape) for x in outputs]}")
        print(f"🎲 Index shapes:  {[tuple(x.shape) for x in indices]}")
        print(f"\n📋 Index Configuration ({index_configuration.shape}):")
        print(f"   Inputs  [{0:2d}:{ni:2d}]: {index_configuration[:ni].tolist()}")
        print(
            f"   Outputs [{ni:2d}:{ni + no:2d}]: {index_configuration[ni : ni + no].tolist()}"
        )
        print(
            f"   Indices [{ni + no:2d}:{len(index_configuration):2d}]: {index_configuration[ni + no :].tolist()}"
        )
        print(f"🎨 Found {len(groups)} unique output groups:")
        for group_idx, output_indices in groups.items():
            print(
                f"   Group {group_idx}: {unique_configs[group_idx].tolist()} → outputs {output_indices} ({len(output_indices)} outputs)"
            )
        print(f"{'=' * 80}\n")

    result_outputs = [None] * no

    for group_idx, output_indices in groups.items():
        group_outputs = [outputs[i] for i in output_indices]
        output_buffer_map = {
            ni + old_idx: ni + new_idx for new_idx, old_idx in enumerate(output_indices)
        }

        group_operations = [
            Operation(
                tuple(
                    output_buffer_map.get(b, b) if ni <= b < ni + no else b
                    for b in op.buffers
                ),
                op.start_path,
                op.num_paths,
            )
            for op in operations
            if any(ni <= b < ni + no and (b - ni) in output_indices for b in op.buffers)
        ]

        group_index_config = np.concatenate(
            [
                index_configuration[:ni],
                index_configuration[ni : ni + no][output_indices],
                index_configuration[ni + no :],
            ]
        )

        used_indices = sorted(set(group_index_config.flatten()) - {-1})
        if len(used_indices) < len(indices):
            index_map = {
                old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)
            }
            group_indices = [indices[i] for i in used_indices]
            group_index_config_remapped = group_index_config.copy()
            for i in range(group_index_config_remapped.shape[0]):
                for j in range(group_index_config_remapped.shape[1]):
                    val = group_index_config_remapped[i, j]
                    if val != -1:
                        group_index_config_remapped[i, j] = index_map[val]
            no_group = len(group_outputs)
            rows_to_keep = list(range(ni + no_group)) + [
                ni + no_group + i for i in used_indices
            ]
            group_index_config_remapped = group_index_config_remapped[rows_to_keep]
        else:
            group_indices = indices
            group_index_config_remapped = group_index_config

        assert group_index_config_remapped.shape[0] == ni + len(group_outputs) + len(
            group_indices
        )

        group_result = deterministic_indexing(
            inputs,
            group_outputs,
            group_indices,
            group_index_config_remapped,
            operations=group_operations,
            paths=paths,
            math_dtype=math_dtype,
            name=f"{name}_group{group_idx}",
        )

        for new_idx, old_idx in enumerate(output_indices):
            result_outputs[old_idx] = group_result[new_idx]

    return result_outputs


def deterministic_indexing(
    inputs: list[jax.Array],
    outputs: list[jax.Array],
    indices: list[jax.Array],
    index_configuration: np.ndarray,
    operations: list,
    paths: list,
    math_dtype: jnp.dtype,
    name: str,
):
    from cuequivariance_ops_jax import tensor_product_uniform_1d_jit

    ni, no = len(inputs), len(outputs)
    assert index_configuration.shape[0] == ni + no + len(indices)

    first_output_config = index_configuration[ni]
    if np.all(first_output_config == -1):
        if os.environ.get("CUEQUIVARIANCE_DEBUG_UNIFORM_1D"):
            print(f"\n{'=' * 80}")
            print(
                f"🎯 deterministic_indexing: {name} (early return - all outputs unindexed)"
            )
            print(f"{'=' * 80}")
            print(
                f"📊 {ni} inputs, {no} outputs, {len(indices)} indices, {len(operations)} ops, {len(paths)} paths"
            )
            print("⏭️  Skipping deterministic indexing (all outputs have -1 indices)")
            print(f"{'=' * 80}\n")
        return tensor_product_uniform_1d_jit(
            inputs,
            outputs,
            indices,
            index_configuration,
            operations=operations,
            paths=paths,
            math_dtype=math_dtype,
            name=name,
        )

    axis = np.argmax(first_output_config != -1)
    primary_index = int(first_output_config[axis])

    buffers = list(inputs) + list(outputs)
    extents = {
        buffers[i].shape[j]
        for i in range(ni + no)
        for j in range(index_configuration.shape[1])
        if index_configuration[i, j] == primary_index
    }
    assert len(extents) == 1, (
        f"Expected unique extent for primary_index={primary_index}, got {extents}"
    )
    extent = extents.pop()

    primary_index_array = indices[primary_index]
    new_indices = []
    for idx, old_index in enumerate(indices):
        indptr, grouped_index = group_by_index(
            primary_index_array, old_index, extent, axis
        )
        new_indices.append(indptr if idx == primary_index else grouped_index)

    index_configuration = index_configuration.copy()
    index_configuration[ni + no + primary_index, axis] = -2

    unindexed_mask = index_configuration[:ni, axis] == -1
    if np.any(unindexed_mask):
        first_unindexed_input = np.argmax(unindexed_mask)
        batch_size = inputs[first_unindexed_input].shape[axis]
        target_shape = list(primary_index_array.shape)
        target_shape[axis] = batch_size

        shape = [1] * len(target_shape)
        shape[axis] = batch_size
        sequential_index = jnp.broadcast_to(
            jnp.arange(batch_size).reshape(shape),
            target_shape,
        )
        _, grouped_sequential = group_by_index(
            primary_index_array, sequential_index, extent, axis
        )
        new_indices.append(grouped_sequential)

        new_index_id = len(new_indices) - 1
        index_configuration[:ni, axis] = np.where(
            unindexed_mask, new_index_id, index_configuration[:ni, axis]
        )
        index_configuration = np.concatenate(
            [
                index_configuration,
                np.full((1, index_configuration.shape[1]), -1, dtype=np.int32),
            ],
            axis=0,
        )

    if os.environ.get("CUEQUIVARIANCE_DEBUG_UNIFORM_1D"):
        print(f"\n{'=' * 80}")
        print(f"🎯 deterministic_indexing: {name}")
        print(f"{'=' * 80}")
        print(
            f"📊 {ni} inputs, {no} outputs, {len(new_indices)} indices, {len(operations)} ops, {len(paths)} paths"
        )
        print(f"🔢 Input shapes:  {[tuple(x.shape) for x in inputs]}")
        print(f"📦 Output shapes: {[tuple(x.shape) for x in outputs]}")
        print(f"🎲 Index shapes:  {[tuple(x.shape) for x in new_indices]}")
        print(f"\n📋 Index Configuration ({index_configuration.shape}):")
        print(f"   Inputs  [{0:2d}:{ni:2d}]: {index_configuration[:ni].tolist()}")
        print(
            f"   Outputs [{ni:2d}:{ni + no:2d}]: {index_configuration[ni : ni + no].tolist()}"
        )
        print(
            f"   Indices [{ni + no:2d}:{len(index_configuration):2d}]: {index_configuration[ni + no :].tolist()}"
        )
        print(f"{'=' * 80}\n")

    return tensor_product_uniform_1d_jit(
        inputs,
        outputs,
        new_indices,
        index_configuration,
        operations=operations,
        paths=paths,
        math_dtype=math_dtype,
        name=name,
    )
