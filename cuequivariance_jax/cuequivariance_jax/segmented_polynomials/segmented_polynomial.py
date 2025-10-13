# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import warnings
from functools import partial

import jax
import jax.core
import jax.extend
import jax.lax
import jax.numpy as jnp
import numpy as np  # noqa: F401
from cuequivariance_jax.segmented_polynomials.indexing_mode import IndexingMode
from cuequivariance_jax.segmented_polynomials.segmented_polynomial_indexed_linear import (
    execute_indexed_linear,
)
from cuequivariance_jax.segmented_polynomials.segmented_polynomial_naive import (
    execute_naive,
)
from cuequivariance_jax.segmented_polynomials.segmented_polynomial_uniform_1d import (
    execute_uniform_1d,
)
from cuequivariance_jax.segmented_polynomials.utils import (
    batch_size,
    reshape,
    sanitize_multi_index,
)
from jax.interpreters import ad, batching, mlir, partial_eval, xla

import cuequivariance as cue
import cuequivariance_jax as cuex  # noqa: F401

logger = logging.getLogger(__name__)


def segmented_polynomial(
    polynomial: cue.SegmentedPolynomial,
    inputs: list[jax.Array],
    outputs_shape_dtype: list[jax.ShapeDtypeStruct],
    indices: None | list[None | jax.Array | tuple[jax.Array | slice]] = None,
    *,
    method: str = "",
    math_dtype: str | None = None,
    name: str | None = None,
    precision: jax.lax.Precision = "undefined",
) -> list[jax.Array]:
    """Compute a segmented polynomial.

    Evaluates a segmented polynomial, which represents a mathematical operation composed of
    several tensor products. The function supports both JAX and CUDA implementations for
    maximum performance and flexibility.

    Args:
        polynomial: The segmented polynomial to compute.
        inputs: List of input buffers as JAX arrays.
        outputs_shape_dtype: List of output shapes and dtypes specifications.
            The last shape dimension can be set to -1 to infer the size from the polynomial descriptor.
        indices: Optional list of indices for inputs and outputs. If None, no indexing
            is applied. Defaults to None. Note that indices are not supported for all methods.
        method: Specifies the implementation method to use. Options are:

            - ``"naive"``: Uses a naive JAX implementation. It always works but is not optimized.
            - ``"uniform_1d"``: Uses a CUDA implementation for polynomials with a single uniform mode.
            - ``"indexed_linear"``: Uses a CUDA implementation for linear layers with indexed weights.

            .. note::
               The ``"fused_tp"`` method is only available in the PyTorch implementation.
        math_dtype: Data type for computational operations. If None, automatically determined from input types. Defaults to None.

            Supported options vary by method:

            - ``"naive"``: String dtype names (e.g., ``"float32"``, ``"float64"``, ``"float16"``, ``"bfloat16"``).
              Also supports ``"tensor_float32"`` for TensorFloat-32 mode.
            - ``"uniform_1d"``: String dtype names (e.g., ``"float32"``, ``"float64"``, ``"float16"``, ``"bfloat16"``).
            - ``"indexed_linear"``: CUBLAS compute type strings such as ``"CUBLAS_COMPUTE_32F"``, ``"CUBLAS_COMPUTE_32F_FAST_TF32"``,
              ``"CUBLAS_COMPUTE_32F_PEDANTIC"``, ``"CUBLAS_COMPUTE_64F"``, etc.

        name: Optional name for the operation.

    Returns:
        List of JAX arrays containing the computed polynomial outputs.

    Notes:
        **JAX Transformations Support:**
            - Supports JAX transformations: jit, grad, jvp, vmap
            - Supports infinite derivatives through JVP and transpose rules
            - Full batching support

    Examples:
        Simple example computing the spherical harmonics:

        >>> p = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2]).polynomial
        >>> cuex.segmented_polynomial(
        ...     p, [jnp.array([0.0, 1.0, 0.0])], [jax.ShapeDtypeStruct((-1,), jnp.float32)], method="naive"
        ... )
        [Array([1.       , 0.       , 1.7320508, 0.       , 0.       , 0.       ,
               2.236068 , 0.       , 0.       ], dtype=float32)]

        Example computing a tensor product with indexing using the "uniform_1d" method:

        >>> poly: cue.SegmentedPolynomial = cue.descriptors.channelwise_tensor_product(
        ...     cue.Irreps(cue.O3, "32x0e + 32x1o + 32x1e + 32x2o"),
        ...     cue.Irreps(cue.O3, "0e + 1o + 1e"),
        ...     cue.Irreps(cue.O3, "32x0e + 32x1o + 32x1e"),
        ... ).polynomial
        >>> a = np.random.randn(1, 50, poly.inputs[0].size)
        >>> b = np.random.randn(10, 50, poly.inputs[1].size)
        >>> c = np.random.randn(100, 1, poly.inputs[2].size)
        >>> i = np.random.randint(0, 10, (100, 50))
        >>> D = jax.ShapeDtypeStruct(shape=(11, 12, poly.outputs[0].size), dtype=jnp.float32)
        >>> j1 = np.random.randint(0, 11, (100, 50))
        >>> j2 = np.random.randint(0, 12, (100, 1))
        >>> [D] = cuex.segmented_polynomial(
        ...     poly, [a, b, c], [D], [None, np.s_[i, :], None, np.s_[j1, j2]], method="uniform_1d"
        ... )
        >>> D.shape
        (11, 12, 1056)

        Example computing a linear layer with indexed weights using the "indexed_linear" method:

        >>> input_irreps = cue.Irreps(cue.O3, "10x0e + 10x1o")
        >>> output_irreps = cue.Irreps(cue.O3, "20x0e + 20x1o")
        >>> poly = cue.descriptors.linear(input_irreps, output_irreps).polynomial
        >>> counts = jnp.array([3, 4, 3], dtype=jnp.int32)  # Number of elements in each partition
        >>> w = jax.random.normal(jax.random.key(0), (3, poly.inputs[0].size), dtype=jnp.float32)
        >>> x = jax.random.normal(jax.random.key(1), (10, poly.inputs[1].size), dtype=jnp.float32)
        >>> y = jax.ShapeDtypeStruct((10, poly.outputs[0].size), jnp.float32)
        >>> [y] = cuex.segmented_polynomial(
        ...     poly,
        ...     [w, x], [y], [cuex.Repeats(counts), None, None],
        ...     method="indexed_linear"
        ... )
        >>> y.shape
        (10, 80)


    .. note::
        This operation uses a custom CUDA kernel for performance. When using this function
        on multiple devices, manual sharding is required to achieve proper performance.
        Without explicit sharding, performance will be significantly degraded. See
        `JAX shard_map documentation <https://docs.jax.dev/en/latest/notebooks/shard_map.html>`_
        for details on manual parallelism.
    """

    if method == "":
        raise ValueError(
            "Hello! It looks like you're using code that was written for an older version of this library. "
            "Starting in v0.6.0, the `method` argument is now required for `segmented_polynomial()`. "
            "This change helps ensure you get optimal performance by explicitly choosing the computation method.\n\n"
            "To fix this, simply add a `method` parameter to your function call. Here are the available options:\n"
            "• 'naive' - Works everywhere but not optimized (good for testing)\n"
            "• 'uniform_1d' - Fast CUDA implementation for single uniform mode polynomials\n"
            "• 'indexed_linear' - Fast CUDA implementation for linear layers with indexed weights\n\n"
            "Example: outputs = segmented_polynomial(poly, inputs, outputs, method='naive')"
        )

    if name is None:
        name = "segmented_polynomial"

    if math_dtype is not None and not isinstance(math_dtype, str):
        math_dtype = jnp.dtype(math_dtype).name
    assert isinstance(math_dtype, str) or math_dtype is None

    if precision != "undefined":
        raise ValueError(
            "precision is not anymore supported. Please use math_dtype instead."
        )

    assert len(inputs) == polynomial.num_inputs
    assert len(outputs_shape_dtype) == polynomial.num_outputs

    for i, x, ope in zip(range(polynomial.num_inputs), inputs, polynomial.inputs):
        if x.ndim == 0:
            raise ValueError(f"Input {i} has no dimensions")
        if x.shape[-1] != ope.size:
            raise ValueError(
                f"Input {i} has shape {x.shape} but expected shape {ope.size} for polynomial:\n{polynomial}"
            )

    # Sanitize the inputs, outputs and indices
    inputs = [jnp.asarray(x) for x in inputs]
    for out, ope in zip(outputs_shape_dtype, polynomial.outputs):
        if len(out.shape) == 0:
            raise ValueError(f"Output has no dimensions: {out}")
        if out.shape[-1] != ope.size and out.shape[-1] != -1:
            warnings.warn(
                f"Output has shape {out.shape} but expected the last dimension to be {ope.size} for polynomial:\n{polynomial}",
                stacklevel=2,
            )
    outputs_shape_dtype = [
        jax.ShapeDtypeStruct(x.shape[:-1] + (ope.size,), x.dtype)
        for x, ope in zip(outputs_shape_dtype, polynomial.outputs)
    ]
    indices = jax.tree.map(
        lambda x: jnp.asarray(x) if hasattr(x, "shape") else x, indices
    )
    io_buffers = list(inputs) + list(outputs_shape_dtype)
    del inputs

    # Determine number of batch axes
    shapes = [x.shape[:-1] for x in io_buffers]
    shapes += [x.shape for x in jax.tree.leaves(indices) if isinstance(x, jax.Array)]
    num_batch_axes: int = max(len(s) for s in shapes)
    del shapes

    # sanitize the indices
    if indices is None:
        indices = [None] * len(io_buffers)

    if len(indices) != len(io_buffers):
        raise ValueError(
            f"Expected {len(io_buffers)} indices, got {len(indices)}. "
            "Please provide an index for each buffer. "
            "If a buffer does not have an index, please set it to None."
        )

    indices: list[None | tuple[jax.Array | slice | cuex.Repeats]] = [
        sanitize_multi_index(idx, num_batch_axes) if idx is not None else idx
        for idx in indices
    ]

    # Expand the buffers to have the same number of batch axes
    def fn(x, n: int):
        if hasattr(x, "shape"):
            return reshape(x, (1,) * (n - x.ndim) + x.shape)
        return x

    io_buffers = [fn(x, num_batch_axes + 1) for x in io_buffers]
    indices = [
        None if multi is None else tuple(fn(x, num_batch_axes) for x in multi)
        for multi in indices
    ]

    # indices --> (unique_indices: list[jax.Array], index_configuration: list[list[int]], index_mode)
    index_mode: list[list[IndexingMode]] = []
    index_configuration: list[list[int]] = []
    unique_indices: list[jax.Array] = []
    for multi in indices:
        if multi is None:
            index_configuration.append([-1] * num_batch_axes)
            index_mode.append([IndexingMode.BATCHED_OR_SHARED] * num_batch_axes)
        else:
            if not all(
                isinstance(i, jax.Array)
                or (isinstance(i, slice) and i == slice(None))
                or isinstance(i, cuex.Repeats)
                for i in multi
            ):
                raise ValueError(
                    f"Expected index to be a jax.Array, cuex.Repeats or a slice, got {multi}"
                )
            im = []
            bi = []
            for a in multi:
                if isinstance(a, slice):
                    assert a == slice(None)
                    bi.append(-1)
                    im.append(IndexingMode.BATCHED_OR_SHARED)
                else:
                    is_repeats = isinstance(a, cuex.Repeats)
                    if is_repeats:
                        a = a.repeats
                    found = False
                    for i, b in enumerate(unique_indices):
                        if a is b:
                            bi.append(i)
                            found = True
                            break
                    if not found:
                        bi.append(len(unique_indices))
                        unique_indices.append(a)
                    im.append(
                        IndexingMode.REPEATED if is_repeats else IndexingMode.INDEXED
                    )
            index_configuration.append(bi)
            index_mode.append(im)

    # Execute the polynomial
    kwargs = dict(
        inputs=io_buffers[: polynomial.num_inputs],
        outputs_shape_dtype=io_buffers[polynomial.num_inputs :],
        indices=unique_indices,
        index_configuration=index_configuration,
        index_mode=index_mode,
        polynomial=polynomial,
        math_dtype=math_dtype,
        name=name,
    )

    outputs = segmented_polynomial_prim(**kwargs, method=method)

    # Reshape the outputs to the original requested shapes
    def fn(x: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jnp.reshape(x, shape)

    return list(map(fn, outputs, [out.shape for out in outputs_shape_dtype]))


segmented_polynomial_p = jax.extend.core.Primitive("segmented_polynomial")
segmented_polynomial_p.multiple_results = True


def _dce_helper(
    used_inputs: list[bool],
    used_outputs: list[bool],
    index_configuration: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    num_indices: int,
) -> tuple[
    list[bool], tuple[tuple[int, ...], ...], tuple[tuple[IndexingMode, ...], ...]
]:
    # Determine which indices are used
    used_indices_id: list[int] = sorted(
        {
            j
            for i, used in enumerate(used_inputs + used_outputs)
            if used
            for j in index_configuration[i]
            if j >= 0
        }
    )
    used_indices: list[bool] = [i in used_indices_id for i in range(num_indices)]

    # Remap the index_configuration to the used indices
    index_configuration = tuple(
        tuple(
            used_indices_id.index(j) if j >= 0 else -1 for j in index_configuration[i]
        )
        for i, used in enumerate(used_inputs + used_outputs)
        if used
    )

    index_mode = tuple(
        tuple(modes)
        for modes, used in zip(index_mode, used_inputs + used_outputs)
        if used
    )

    return used_indices, index_configuration, index_mode


def segmented_polynomial_prim(
    inputs: list[jax.Array],  # input buffers
    outputs_shape_dtype: list[jax.ShapeDtypeStruct],  # output shapes and dtypes
    indices: list[jax.Array],  # index buffers
    index_configuration: list[list[int]],  # maps: buffer index -> unique indices index
    index_mode: list[list[IndexingMode]],  # shared, batched, indexed, repeated
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
    method: str,
    return_none_if_empty: bool = False,
) -> tuple[jax.Array, ...]:  # output buffers
    """
    - Filters out unused buffers and indices
    - Calls the tensor product primitive
    - Maps the outputs back to the original output buffers
    """
    assert len(inputs) + len(outputs_shape_dtype) == len(index_configuration)
    assert max(max(bi, default=-1) for bi in index_configuration) < len(indices)

    # fuse STPs, consolidate modes, squeeze modes, remove empty segments, consolidate paths, sort paths
    polynomial = polynomial.consolidate()

    used_inputs, used_outputs = polynomial.used_inputs(), polynomial.used_outputs()

    used_indices, index_configuration, index_mode = _dce_helper(
        used_inputs, used_outputs, index_configuration, index_mode, len(indices)
    )

    new_outputs = segmented_polynomial_p.bind(
        *[v for v, used in zip(inputs, used_inputs) if used],
        *[v for v, used in zip(indices, used_indices) if used],
        index_configuration=index_configuration,
        index_mode=index_mode,
        outputs_shape_dtype=tuple(
            x for x, used in zip(outputs_shape_dtype, used_outputs) if used
        ),
        polynomial=polynomial.filter_keep_operands(used_inputs + used_outputs),
        math_dtype=math_dtype,
        name=str(name),
        method=method,
    )

    if return_none_if_empty:
        old_outputs = [None] * len(outputs_shape_dtype)
    else:
        old_outputs = [jnp.zeros(out.shape, out.dtype) for out in outputs_shape_dtype]

    i_new = 0
    for i_old, used in enumerate(used_outputs):
        if used:
            old_outputs[i_old] = new_outputs[i_new]
            i_new += 1

    return tuple(old_outputs)


def _remap_indices_and_config(
    old_indices: list[jax.Array],
    old_index_configuration: tuple[tuple[int, ...], ...],
    old_index_mode: tuple[tuple[IndexingMode, ...], ...],
    mapping: list[int],
) -> tuple[
    list[jax.Array], tuple[tuple[int, ...], ...], tuple[tuple[IndexingMode, ...], ...]
]:
    new_indices = []
    new_index_configuration = []
    new_index_mode = []

    for old_i in mapping:  # len = new_num_inputs + new_num_outputs
        new_bi = []
        for a in old_index_configuration[old_i]:
            if a >= 0:
                b: jax.Array = old_indices[a]
                found = False
                for i, c in enumerate(new_indices):
                    if b is c:
                        new_bi.append(i)
                        found = True
                        break
                if not found:
                    new_bi.append(len(new_indices))
                    new_indices.append(b)
            else:
                new_bi.append(-1)
        new_index_configuration.append(tuple(new_bi))
        new_index_mode.append(old_index_mode[old_i])
    return new_indices, tuple(new_index_configuration), tuple(new_index_mode)


def segmented_polynomial_abstract_eval(
    *inputs_and_indices: jax.core.ShapedArray,
    index_configuration: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
    method: str,
) -> tuple[jax.core.ShapedArray, ...]:
    return tuple(
        jax.core.ShapedArray(out.shape, out.dtype) for out in outputs_shape_dtype
    )


def segmented_polynomial_impl(
    platform: str | None,
    *inputs_and_indices: jax.Array,
    index_configuration: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
    method: str,
) -> tuple[jax.Array, ...]:
    num_inputs = len(index_configuration) - len(outputs_shape_dtype)
    inputs, indices = inputs_and_indices[:num_inputs], inputs_and_indices[num_inputs:]
    del inputs_and_indices
    assert polynomial.num_inputs == num_inputs

    assert all(polynomial.used_operands())

    try:  # TODO: remove this try-except block
        polynomial = polynomial.unsymmetrize_for_identical_operands()
    except NotImplementedError:
        pass

    if os.environ.get("CUE_PRINT_STATS"):
        bi = np.array(index_configuration, dtype=np.int32)
        io = list(inputs) + list(outputs_shape_dtype)
        batch_sizes = [
            batch_size([x.shape[i] for x, idx in zip(io, bi[:, i]) if idx < 0])
            for i in range(bi.shape[1])
        ]
        fl = polynomial.flop(math.prod(batch_sizes))
        mem = sum(x.size * x.dtype.itemsize for x in io + list(indices))
        print(
            f"{name}: {fl / 1e9:.2f} GFLOP, {mem / 1e9:.2f} GB, arithmetic intensity: {fl / mem:.2f} FLOP/byte"
        )

    assert method in ("naive", "uniform_1d", "indexed_linear")
    if platform != "cuda" and method != "naive":
        warnings.warn(
            f"Method '{method}' requires CUDA, but platform is '{platform}'. "
            "Falling back to 'naive' implementation.",
            stacklevel=2,
        )
        method = "naive"

    kwargs = dict(
        inputs=inputs,
        outputs_shape_dtype=outputs_shape_dtype,
        indices=indices,
        index_configuration=index_configuration,
        polynomial=polynomial,
        math_dtype=math_dtype,
        name=name,
    )

    if any(mode == IndexingMode.REPEATED for modes in index_mode for mode in modes):
        if method not in ("naive", "indexed_linear"):
            raise ValueError(
                "IndexingMode.REPEATED is only supported with 'naive' or 'indexed_linear' methods."
            )

    match method:
        case "naive":
            return execute_naive(**kwargs, index_mode=index_mode)
        case "uniform_1d":
            return execute_uniform_1d(**kwargs)
        case "indexed_linear":
            return execute_indexed_linear(**kwargs, index_mode=index_mode)


def segmented_polynomial_jvp(
    primals_and_indices: tuple[jax.Array, ...],
    tangents_and_zeros: tuple[jax.Array | ad.Zero, ...],
    *,
    index_configuration: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
    method: str,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array | ad.Zero, ...]]:
    num_inputs = len(index_configuration) - len(outputs_shape_dtype)

    primals, tangents = (
        primals_and_indices[:num_inputs],
        tangents_and_zeros[:num_inputs],
    )
    indices = primals_and_indices[num_inputs:]
    assert all(isinstance(t, ad.Zero) for t in tangents_and_zeros[num_inputs:])
    del primals_and_indices, tangents_and_zeros

    out_primals = segmented_polynomial_prim(
        primals,
        outputs_shape_dtype,
        indices,
        index_configuration,
        index_mode,
        polynomial,
        math_dtype,
        name,
        method=method,
    )

    jvp_poly, _ = polynomial.jvp([not isinstance(t, ad.Zero) for t in tangents])
    jvp_indices, jvp_index_configuration, jvp_index_mode = _remap_indices_and_config(
        indices,
        index_configuration,
        index_mode,
        [i for i, x in enumerate(primals)]
        + [i for i, x in enumerate(tangents) if not isinstance(x, ad.Zero)]
        + [num_inputs + i for i, x in enumerate(outputs_shape_dtype)],
    )

    out_tangents = segmented_polynomial_prim(
        list(primals) + [t for t in tangents if not isinstance(t, ad.Zero)],
        outputs_shape_dtype,
        jvp_indices,
        jvp_index_configuration,
        jvp_index_mode,
        jvp_poly,
        math_dtype,
        name
        + "_jvp"
        + "".join("0" if isinstance(t, ad.Zero) else "1" for t in tangents),
        method=method,
    )

    return out_primals, out_tangents


def segmented_polynomial_transpose(
    cotangents: tuple[jax.Array | ad.Zero, ...],
    *inputs_and_indices: jax.Array | ad.UndefinedPrimal,
    index_configuration: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
    method: str,
) -> tuple[jax.Array | ad.Zero | None, ...]:
    num_inputs = len(index_configuration) - len(outputs_shape_dtype)
    inputs, indices = inputs_and_indices[:num_inputs], inputs_and_indices[num_inputs:]
    assert all(not ad.is_undefined_primal(idx) for idx in indices)
    del inputs_and_indices

    # The cotangents replace the outputs as inputs
    # The undefined primal inputs become outputs

    tr_poly, _ = polynomial.transpose(
        [ad.is_undefined_primal(x) for x in inputs],
        [not isinstance(x, ad.Zero) for x in cotangents],
    )
    tr_indices, tr_index_configuration, tr_index_mode = _remap_indices_and_config(
        indices,
        index_configuration,
        index_mode,
        [i for i, x in enumerate(inputs) if not ad.is_undefined_primal(x)]
        + [
            num_inputs + i
            for i, x in enumerate(cotangents)
            if not isinstance(x, ad.Zero)
        ]
        + [i for i, x in enumerate(inputs) if ad.is_undefined_primal(x)],
    )

    tmp = segmented_polynomial_prim(
        [x for x in inputs if not ad.is_undefined_primal(x)]
        + [x for x in cotangents if not isinstance(x, ad.Zero)],  # inputs
        [
            jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype)
            for x in inputs
            if ad.is_undefined_primal(x)
        ],
        tr_indices,
        tr_index_configuration,
        tr_index_mode,
        tr_poly,
        math_dtype,
        name + "_T",
        method=method,
        return_none_if_empty=True,
    )

    outputs = [None] * (len(inputs) + len(indices))
    i = 0
    for b, input in enumerate(inputs):
        if ad.is_undefined_primal(input):
            outputs[b] = tmp[i] if tmp[i] is not None else ad.Zero(input.aval)
            i += 1
    return tuple(outputs)


def segmented_polynomial_batching(
    batched_inputs_and_indices: tuple[jax.Array, ...],
    batch_axes_of_inputs_and_indices: tuple[int | None, ...],
    *,
    index_configuration: tuple[tuple[int, ...], ...],
    index_mode: tuple[tuple[IndexingMode, ...], ...],
    outputs_shape_dtype: tuple[jax.ShapeDtypeStruct, ...],
    polynomial: cue.SegmentedPolynomial,
    math_dtype: str | None,
    name: str,
    method: str,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    # Add a new batch axis in the first dimension
    def prepare(input: jax.Array, axis: int | None) -> jax.Array:
        if axis is None:
            return jnp.expand_dims(input, 0)
        else:
            return jnp.moveaxis(input, axis, 0)

    batched_inputs_and_indices = [
        prepare(input, axis)
        for input, axis in zip(
            batched_inputs_and_indices, batch_axes_of_inputs_and_indices
        )
    ]

    # Determine the new batch dimension
    new_dim = 1
    for x in batched_inputs_and_indices:
        if x.shape[0] != 1:
            assert new_dim in (1, x.shape[0])
            new_dim = x.shape[0]

    outputs_shape_dtype = tuple(
        jax.ShapeDtypeStruct((new_dim,) + out.shape, out.dtype)
        for out in outputs_shape_dtype
    )

    # The new batch axis is not indexed
    index_configuration = tuple((-1,) + bi for bi in index_configuration)
    index_mode = tuple((IndexingMode.BATCHED_OR_SHARED,) + im for im in index_mode)

    outputs = segmented_polynomial_p.bind(
        *batched_inputs_and_indices,
        index_configuration=index_configuration,
        index_mode=index_mode,
        outputs_shape_dtype=outputs_shape_dtype,
        polynomial=polynomial,
        math_dtype=math_dtype,
        name=name + "_batching",
        method=method,
    )
    return outputs, (0,) * len(outputs)


def segmented_polynomial_dce(
    used_outputs: list[bool],
    eqn: jax.extend.core.JaxprEqn,
) -> tuple[list[bool], jax.extend.core.JaxprEqn | None]:
    assert len(used_outputs) == len(eqn.outvars)

    polynomial: cue.SegmentedPolynomial = eqn.params["polynomial"]
    index_configuration = eqn.params["index_configuration"]
    index_mode = eqn.params["index_mode"]
    outputs_shape_dtype = eqn.params["outputs_shape_dtype"]

    # If no outputs are used, we can eliminate the operation entirely
    if not any(used_outputs) and not eqn.effects:
        return [False] * len(eqn.invars), None

    num_inputs = polynomial.num_inputs

    polynomial = polynomial.compute_only(used_outputs)
    used_inputs: list[bool] = polynomial.used_inputs()

    used_indices, index_configuration, index_mode = _dce_helper(
        used_inputs,
        used_outputs,
        index_configuration,
        index_mode,
        len(eqn.invars) - num_inputs,
    )

    new_eqn = jax.extend.core.JaxprEqn(
        [v for v, used in zip(eqn.invars, used_inputs + used_indices) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive,
        dict(
            eqn.params,
            polynomial=polynomial.filter_keep_operands(used_inputs + used_outputs),
            index_configuration=index_configuration,
            index_mode=index_mode,
            outputs_shape_dtype=tuple(
                x for x, used in zip(outputs_shape_dtype, used_outputs) if used
            ),
        ),
        eqn.effects,
        eqn.source_info,
        eqn.ctx,
    )

    return used_inputs + used_indices, new_eqn


segmented_polynomial_p.def_abstract_eval(segmented_polynomial_abstract_eval)
segmented_polynomial_p.def_impl(partial(xla.apply_primitive, segmented_polynomial_p))
mlir.register_lowering(
    segmented_polynomial_p,
    mlir.lower_fun(
        partial(segmented_polynomial_impl, "cuda"),
        segmented_polynomial_p.multiple_results,
    ),
    "cuda",
)
mlir.register_lowering(
    segmented_polynomial_p,
    mlir.lower_fun(
        partial(segmented_polynomial_impl, None),
        segmented_polynomial_p.multiple_results,
    ),
    None,
)
ad.primitive_jvps[segmented_polynomial_p] = segmented_polynomial_jvp
ad.primitive_transposes[segmented_polynomial_p] = segmented_polynomial_transpose
batching.primitive_batchers[segmented_polynomial_p] = segmented_polynomial_batching
partial_eval.dce_rules[segmented_polynomial_p] = segmented_polynomial_dce
