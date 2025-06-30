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
from einops import rearrange as rea
from packaging import version

import cuequivariance as cue
from cuequivariance_jax.segmented_polynomials.utils import reshape


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
    math_dtype: jnp.dtype,
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
            + f"This method do not support coefficient modes. Flattening them failed:\n{e}"
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

    math_dtype = jnp.dtype(math_dtype)
    if math_dtype.type not in {jnp.float32, jnp.float64}:
        raise ValueError(f"Unsupported math_dtype: {math_dtype}")

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
        math_dtype=math_dtype,
        name=sanitize_string(name),
    )
    return [jnp.reshape(x, y.shape) for x, y in zip(outputs, outputs_shape_dtype)]


def execute_gemm_grouped(
    inputs: list[jax.Array],  # shape (*batch_sizes, operand_size)
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    indices: list[jax.Array],
    index_configuration: tuple[tuple[int, ...], ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: jnp.dtype,
    name: str,
) -> list[jax.Array]:
    index_configuration = np.array(index_configuration)
    num_batch_axes = index_configuration.shape[1]
    assert (
        polynomial.num_inputs + len(outputs_shape_dtype) == index_configuration.shape[0]
    )
    assert polynomial.num_outputs == len(outputs_shape_dtype)

    math_dtype = jnp.dtype(math_dtype)
    if math_dtype.type not in {jnp.float32, jnp.float64}:
        raise ValueError(f"Unsupported math_dtype: {math_dtype}")

    if not all(x.dtype == math_dtype for x in inputs):
        raise ValueError("All inputs must have the same dtype as math_dtype")

    if not all(x.dtype == math_dtype for x in outputs_shape_dtype):
        raise ValueError("All outputs must have the same dtype as math_dtype")

    if not all(x.dtype in {jnp.int32, jnp.int64} for x in indices):
        raise ValueError("All indices must have dtype int32 or int64")

    from cuequivariance_ops_jax import gemm_grouped

    # index_configuration = np.concatenate(
    #     [index_configuration, np.full((len(indices), num_batch_axes), -1, np.int32)]
    # )

    if not np.all(index_configuration == -1):
        raise ValueError("GEMM grouped: indexing not supported")
    if len(indices) != 0:
        raise ValueError("GEMM grouped: indices not supported")

    gemms = []

    nin = polynomial.num_inputs
    for ope, stp in polynomial.operations:
        assert stp.num_operands == 3, f"Unsupported STP: {stp}"
        assert stp.coefficient_subscripts == ""
        oid, i = ope.output_operand_buffer(nin)
        [AA, BB] = [inputs[i] for i in ope.input_buffers(nin)]
        CC = outputs_shape_dtype[i - nin]
        stp = stp.move_operand_last(oid)

        Aslices = stp.operands[0].segment_slices()
        Bslices = stp.operands[1].segment_slices()

        for path in stp.paths:
            A = AA[..., Aslices[path.indices[0]]]
            B = BB[..., Bslices[path.indices[1]]]

            A = jnp.reshape(A, A.shape[:-1] + stp.operands[0].segments[path.indices[0]])
            B = jnp.reshape(B, B.shape[:-1] + stp.operands[1].segments[path.indices[1]])
            C_shape = CC.shape[:-1] + stp.operands[2].segments[path.indices[2]]

            assert "z" not in stp.subscripts
            sa, sb, sc = stp.subscripts.operands
            assert (
                A.ndim == 1 + len(sa)
                and B.ndim == 1 + len(sb)
                and len(C_shape) == 1 + len(sc)
            )
            if A.shape[0] == 1:
                A = rea(A, "1 ... -> ...")
            else:
                sa = "z" + sa
            if B.shape[0] == 1:
                B = rea(B, "1 ... -> ...")
            else:
                sb = "z" + sb
            if C_shape[0] == 1:
                C_shape = C_shape[1:]
            else:
                sc = "z" + sc

            print(
                f"GEMM: {A.shape} @ {B.shape} -> {C_shape} with contraction {sa},{sb}->{sc}"
            )

            gemm = None

            if (sa, sb, sc) == ("uv", "zwu", "zwv"):
                gemm = (rea(B, "z w u -> (z w) u"), A, False, False)
            if (sa, sb, sc) == ("zuv", "zwu", "zwv"):
                gemm = (B, A, False, False)
            if (sa, sb, sc) == ("uv", "zwv", "zwu"):
                gemm = (rea(B, "z w v -> (z w) v"), A, False, True)
            if (sa, sb, sc) == ("zuv", "zuw", "vw"):
                gemm = (
                    rea(A, "z u v -> (z u) v"),
                    rea(B, "z u w -> (z u) w"),
                    True,
                    False,
                )
            if (sa, sb, sc) == ("zuv", "zuw", "wv"):
                gemm = (
                    rea(B, "z u w -> (z u) w"),
                    rea(A, "z u v -> (z u) v"),
                    True,
                    False,
                )

            if (sa, sb, sc) == ("zu", "zv", "vu"):
                gemm = (B, A, True, False)
            if (sa, sb, sc) == ("zu", "zv", "uv"):
                gemm = (A, B, True, False)
            if (sa, sb, sc) == ("uv", "zu", "zv"):
                gemm = (B, A, False, False)
            if (sa, sb, sc) == ("uv", "zv", "zu"):
                gemm = (B, A, False, True)
            if (sa, sb, sc) == ("u", "zvu", "zv"):
                gemm = (rea(B, "z v u -> (z v) u"), A[:, None], False, False)
            if (sa, sb, sc) == ("u", "zv", "zvu"):
                gemm = (rea(B, "z v -> (z v) 1"), A[None, :], False, False)
            if (sa, sb, sc) == ("zu", "zuv", "v"):
                gemm = (
                    rea(A, "z u -> (z u) 1"),
                    rea(B, "z u v -> (z u) v"),
                    True,
                    False,
                )
            if (sa, sb, sc) == ("zuv", "zu", "v"):
                gemm = (
                    rea(A, "z u v -> (z u) v"),
                    rea(B, "z u -> (z u) 1"),
                    True,
                    False,
                )

            if (sa, sb, sc) == ("u", "zu", "z"):
                gemm = (B, A[:, None], False, False)
            if (sa, sb, sc) == ("u", "z", "zu"):
                gemm = (B[:, None], A[None, :], False, False)
            if (sa, sb, sc) == ("z", "zu", "u"):
                gemm = (A[None, :], B, False, False)
            if (sa, sb, sc) == ("zu", "z", "u"):
                gemm = (B[None, :], A, False, False)

            if gemm is None:
                raise ValueError(
                    f"gemm_grouped does not support: {A.shape} @ {B.shape} -> {C_shape} with contraction {sa},{sb}->{sc}"
                )
            gemms.append(gemm + (path.coefficients.item(),))

    num_batch_axes = {A.ndim - 2 for A, _, _, _, _ in gemms}
    assert len(num_batch_axes) == 1
    num_batch_axes = num_batch_axes.pop()
    gemm_outs = gemm_grouped(
        gemms, [], np.full((2 * len(gemms), num_batch_axes), -1, np.int32)
    )
    outputs = [jnp.zeros(x.shape, dtype=x.dtype) for x in outputs_shape_dtype]

    for ope, stp in polynomial.operations:
        oid, i = ope.output_operand_buffer(nin)
        slices = stp.operands[oid].segment_slices()
        segments = stp.operands[oid].segments

        for path in stp.paths:
            sid = path.indices[oid]
            acc = outputs[i - nin]
            outputs[i - nin] = acc.at[..., slices[sid]].add(
                jnp.reshape(
                    gemm_outs.pop(0), acc.shape[:-1] + (math.prod(segments[sid]),)
                )
            )
    return outputs
