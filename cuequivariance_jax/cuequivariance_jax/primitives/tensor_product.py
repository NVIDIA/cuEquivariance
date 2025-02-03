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
from functools import partial

import jax
import jax.core
import jax.extend
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mosaic.gpu import profiler
from jax.interpreters import ad, batching, mlir, xla

import cuequivariance as cue
from cuequivariance.tensor_product_execution import (
    Computation,
    InBuffer,
    OutBuffer,
    TensorProductExecution,
)
from cuequivariance_jax.primitives.tensor_product_ops_impl import (
    tensor_product_ops_impl,
)
from cuequivariance_jax.primitives.tensor_product_vanilla_impl import (
    tensor_product_vanilla_impl,
)

logger = logging.getLogger(__name__)


def tensor_product(
    d: cue.SegmentedTensorProduct,
    *inputs: jax.Array,
    dtype_output: jnp.dtype | None = None,
    dtype_math: jnp.dtype | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    algorithm: str = "sliced",
    use_custom_primitive: bool = True,
    use_custom_kernels: bool | None = False,
    name: str | None = None,
    **options,
) -> jax.Array:
    """
    Compute the last operand of a `SegmentedTensorProduct`.

    Args:
        d (SegmentedTensorProduct): The descriptor of the operation.
        *inputs (jax.Array): The input arrays for each operand except the last one.
        dtype_output (jnp.dtype, optional): The data type for the output.
        dtype_math (jnp.dtype, optional): The data type for mathematical operations.
        precision (jax.lax.Precision, optional): The precision for the computation. Defaults to ``jax.lax.Precision.HIGHEST``.
        algorithm (str, optional): The algorithm to use for the computation. Defaults to "sliced". See table below for available algorithms.
        use_custom_primitive (bool, optional): Whether to use custom JVP/transpose rules.
        use_custom_kernels (bool, optional): Whether to use custom kernels.

    Returns:
        jax.Array: The result of the tensor product. The last operand of the `SegmentedTensorProduct`.

    .. table:: Available algorithms for the tensor product
        :align: center
        :class: longtable

        +---------------------+--------------------------+------------------+----------------------------+
        | Algorithms          | Needs Identical Segments | Compilation Time | Execution Time             |
        +=====================+==========================+==================+============================+
        |``sliced``           | No                       | Several minutes  | It depends                 |
        +---------------------+--------------------------+------------------+----------------------------+
        |``stacked``          | Yes                      | Several minutes  | It depends                 |
        +---------------------+--------------------------+------------------+----------------------------+
        |``compact_stacked``  | Yes                      | Few seconds      | It depends                 |
        +---------------------+--------------------------+------------------+----------------------------+
        |``indexed_compact``  | Yes                      | Few seconds      | It depends                 |
        +---------------------+--------------------------+------------------+----------------------------+
        |``indexed_vmap``     | Yes                      | Few seconds      | Probably the second slowest|
        +---------------------+--------------------------+------------------+----------------------------+
        |``indexed_for_loop`` | Yes                      | Few seconds      | Probably the slowest       |
        +---------------------+--------------------------+------------------+----------------------------+
    """
    if isinstance(precision, str):
        precision = jax.lax.Precision[precision]

    if name is None:
        name = "tensor_product"

    kwargs = dict(
        dtype_output=dtype_output,
        dtype_math=dtype_math,
        precision=precision,
        algorithm=algorithm,
        use_custom_primitive=use_custom_primitive,
        use_custom_kernels=use_custom_kernels,
        name=name,
        **options,
    )

    if len(inputs) > d.num_operands - 1:
        raise ValueError(
            f"cuex.tensor_product: expected {d.num_operands - 1} inputs, got {len(inputs)}"
        )

    # currying
    if len(inputs) < d.num_operands - 1:

        def _partial(*remaining_inputs: jax.Array) -> jax.Array:
            return tensor_product(d, *inputs, *remaining_inputs, **kwargs)

        return _partial

    for oid, input in enumerate(inputs):
        if input.ndim == 0:
            raise ValueError(f"cuex.tensor_product: input {oid} no dimensions")
        if input.shape[-1] != d.operands[oid].size:
            raise ValueError(
                f"cuex.tensor_product: expected operand {oid} to have size {d.operands[oid].size}, got {input.shape[-1]}"
            )

    d = d.remove_zero_paths()
    d = d.remove_empty_segments()

    if len(inputs) > 0:
        output_shape = jnp.broadcast_shapes(*[input.shape[:-1] for input in inputs])
        if dtype_output is None:
            dtype_output = jnp.result_type(*inputs)
    else:
        output_shape = ()
        if dtype_output is None:
            dtype_output = jnp.empty(0).dtype

    if dtype_math is None:
        if dtype_output.itemsize <= jnp.dtype(jnp.float32).itemsize:
            dtype_math = jnp.empty((), jnp.float32).dtype
        else:
            dtype_math = dtype_output

    kwargs = dict(
        dtype_output=dtype_output,
        dtype_math=dtype_math,
        precision=precision,
        algorithm=algorithm,
        use_custom_primitive=use_custom_primitive,
        use_custom_kernels=use_custom_kernels,
        name=name,
        **options,
    )

    # inputs of shape (..., ope.size) with identical ndim
    inputs = [
        jnp.reshape(input, (1,) * (len(output_shape) + 1 - input.ndim) + input.shape)
        for input in inputs
    ]
    output_shapes = tuple(None for _ in inputs) + (output_shape,)
    exe = TensorProductExecution(
        [
            Computation(
                [InBuffer(oid) for oid in range(d.num_operands - 1)] + [OutBuffer(0)]
            )
        ]
    )
    (output,) = tensor_product_prim(
        *inputs, output_shapes=output_shapes, d=d, exe=exe, **kwargs
    )
    return output


################################################################################

tensor_product_p = jax.extend.core.Primitive("tensor_product")
tensor_product_p.multiple_results = True


def clean_inputs(
    inputs: list[jax.Array], exe: TensorProductExecution
) -> tuple[list[jax.Array], TensorProductExecution]:
    # remove unused inputs
    inputs = [inputs[i] for i in exe.in_buffers]
    exe = exe.map_buffers(lambda i: exe.in_buffers.index(i))

    # remove duplicate inputs
    unique_inputs = []
    for x in inputs:
        if id(x) not in map(id, unique_inputs):
            unique_inputs.append(x)
    exe = exe.map_buffers(lambda i: [id(x) for x in unique_inputs].index(id(inputs[i])))

    return unique_inputs, exe


def tensor_product_prim(
    *inputs: jax.Array,  # input buffers
    output_shapes: tuple[tuple[int, ...] | None, ...],  # shapes of the operands
    d: cue.SegmentedTensorProduct,
    exe: TensorProductExecution,
    use_custom_primitive: bool = True,
    **options,
) -> tuple[jax.Array, ...]:  # output buffers
    if exe.is_trivial:
        return ()

    assert exe.max_out_buffer + 1 == len(exe.out_buffers)

    unique_inputs, exe = clean_inputs(list(inputs), exe)

    if use_custom_primitive:
        return tensor_product_p.bind(
            *unique_inputs, output_shapes=output_shapes, d=d, exe=exe, **options
        )
    else:
        return tensor_product_vanilla_impl(
            *unique_inputs, output_shapes=output_shapes, d=d, exe=exe, **options
        )


def produce_minimal_code(
    *inputs: jax.Array,
    output_shapes: tuple[tuple[int, ...] | None, ...],
    d: cue.SegmentedTensorProduct,
    exe: TensorProductExecution,
    dtype_output: jnp.dtype,
    dtype_math: jnp.dtype,
    **options,
) -> str:
    def format_dtype(dtype: jnp.dtype) -> str:
        dtype = jnp.dtype(dtype)
        return f"jnp.{dtype.name}"

    mincode = """import jax.numpy as jnp
import cuequivariance as cue
from cuequivariance.tensor_product_execution import InBuffer, OutBuffer
from cuequivariance_jax.primitives.tensor_product import tensor_product_prim
"""
    mincode += (
        "inputs = ["
        + ", ".join([f"jnp.zeros({x.shape}, {format_dtype(x.dtype)})" for x in inputs])
        + "]\n"
    )
    mincode += f"output_shapes = {output_shapes}\n"
    mincode += f'd = cue.SegmentedTensorProduct.from_base64("{d.to_base64()}")\n'
    mincode += f"exe = cue.TensorProductExecution({exe.computations})\n"
    mincode += f"dtype_output = {format_dtype(dtype_output)}\n"
    mincode += f"dtype_math = {format_dtype(dtype_math)}\n"
    # tensor_product_prim(
    #     *inputs,
    #     output_shapes=output_shapes,
    #     d=d,
    #     exe=exe,
    #     dtype_output=dtype_output,
    #     dtype_math=dtype_math,
    #     use_custom_kernels=True,
    # )
    mincode += "# " + ", ".join([f"{k}={v}" for k, v in options.items()])
    return mincode


def profile_and_select_implementation(
    name: str, impls: list[tuple[str, callable]], *inputs: jax.Array
):
    # import time
    # t0 = time.perf_counter()

    with jax.ensure_compile_time_eval():
        dummy_inputs = [np.random.normal(size=x.shape).astype(x.dtype) for x in inputs]
        dummy_inputs = [jax.device_put(x) for x in dummy_inputs]
        ref = None
        first_runtime: float | None = None
        best: tuple[str, float, callable] | None = None
        for impl_name, impl in impls:
            try:
                out, runtime = profiler.measure(impl, mode="cupti")(*dummy_inputs)
            except NotImplementedError:
                continue
            else:
                logger.info(f"🚀 {name}: {impl_name} runtime {runtime:.2f} ms")
                if ref is None:
                    ref = out
                    best = impl_name, runtime, impl
                    first_runtime = runtime
                else:
                    dtype = jax.tree.leaves(out)[0].dtype
                    diff = max(
                        [
                            np.max(np.abs(a - b))
                            for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref))
                        ]
                    )
                    if diff > 50.0 * jnp.finfo(dtype).eps:
                        raise ValueError(
                            f"cuex.tensor_product: {name} implementation {impl_name} produced different results, diff={diff}"
                        )
                    if runtime < best[1]:
                        best = impl_name, runtime, impl
        assert best is not None
    impl_name, runtime, impl = best

    # dt = time.perf_counter() - t0
    speedup = first_runtime / runtime
    print(
        f"{name:<50}: {impl_name:<10} with runtime {runtime:.2f} ms, speedup {speedup:.2f}x wrt {first_runtime:.2f} ms"
    )

    return impl(*inputs)


def tensor_product_impl(
    platform: str | None,
    *inputs: jax.Array,
    output_shapes: tuple[tuple[int, ...] | None, ...],
    d: cue.SegmentedTensorProduct,
    exe: TensorProductExecution,
    name: str = "tensor_product",
    use_custom_kernels: bool | None = True,
    block_u: int | None = None,
    elements_per_thread: int | None = None,
    **options,
) -> tuple[jax.Array, ...]:
    assert exe.max_in_buffer + 1 == len(exe.in_buffers) == len(inputs)
    assert exe.max_out_buffer + 1 == len(exe.out_buffers)

    def dispatch(
        inputs: list[jax.Array],
        d: cue.SegmentedTensorProduct,
        exe: TensorProductExecution,
    ) -> list[jax.Array]:
        kwargs = dict(output_shapes=output_shapes, d=d, exe=exe, **options)

        if platform == "cuda":
            # print(produce_minimal_code(*inputs, **kwargs))
            # print()
            if use_custom_kernels is None:
                outputs = profile_and_select_implementation(
                    name,
                    [
                        ("vanilla", partial(tensor_product_vanilla_impl, **kwargs)),
                    ]
                    + [
                        (
                            f"ops_{bu}_{ept}",
                            partial(
                                tensor_product_ops_impl,
                                **kwargs,
                                block_u=bu,
                                elements_per_thread=ept,
                            ),
                        )
                        for bu in ([block_u] if block_u else [16, 32, 64, 128])
                        for ept in (
                            [elements_per_thread]
                            if elements_per_thread
                            else [1, 2, 4, 8]
                        )
                    ],
                    *inputs,
                )
                return outputs
            if use_custom_kernels is True:
                return tensor_product_ops_impl(
                    *inputs,
                    **kwargs,
                    block_u=block_u or 16,
                    elements_per_thread=elements_per_thread or 1,
                )

        return tensor_product_vanilla_impl(*inputs, **kwargs)

    outputs = [0] * len(exe.out_buffers)
    for partition, ex in exe.group_by_identical_buffers():
        d_sorted = d
        for x in partition:
            d_sorted = d_sorted.sort_indices_for_identical_operands(x)

        used_inputs, ex = clean_inputs(inputs, ex)
        tmp = dispatch(
            used_inputs,
            d_sorted,
            ex.map_buffers(None, lambda b: ex.out_buffers.index(b)),
        )
        for b, t in zip(ex.out_buffers, tmp):
            outputs[b] += t

    return tuple(outputs)


def tensor_product_abstract_eval(
    *inputs: jax.core.ShapedArray,
    output_shapes: tuple[tuple[int, ...] | None, ...],
    d: cue.SegmentedTensorProduct,
    exe: TensorProductExecution,
    dtype_output: jnp.dtype,
    **options,
) -> tuple[jax.core.ShapedArray, ...]:
    # assert that all input/output are used
    assert exe.max_in_buffer + 1 == len(exe.in_buffers) == len(inputs)
    assert exe.max_out_buffer + 1 == len(exe.out_buffers)

    for c in exe.computations:
        for oid, x in zip(c.in_operands, c.map_inputs(inputs)):
            if x.shape[-1] != d.operands[oid].size:
                raise ValueError(
                    f"cuex.tensor_product: expected input to have size {d.operands[oid].size}, got {x.shape[-1]}"
                )

    outputs = [None] * len(exe.out_buffers)
    for c in exe.computations:
        out = jax.core.ShapedArray(
            shape=output_shapes[c.out_operand] + (d.operands[c.out_operand].size,),
            dtype=dtype_output,
        )
        assert outputs[c.out_buffer] is None or outputs[c.out_buffer] == out
        outputs[c.out_buffer] = out
    return tuple(outputs)


def tensor_product_jvp(
    primals: tuple[jax.Array, ...],
    tangents: tuple[jax.Array | ad.Zero, ...],
    *,
    output_shapes: tuple[tuple[int, ...] | None, ...],
    d: cue.SegmentedTensorProduct,
    exe: TensorProductExecution,
    name: str,
    **options,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array | ad.Zero, ...]]:
    out_primals = tensor_product_prim(
        *primals, output_shapes=output_shapes, d=d, exe=exe, name=name, **options
    )
    out_tangents = [ad.Zero(p.aval) for p in out_primals]

    jvp = exe.jvp([not isinstance(t, ad.Zero) for t in tangents])
    del exe

    permutations: list[tuple[int, ...]] = d.symmetries()
    for multiplicator, exe in jvp.group_by_symmetries(permutations):
        # tensor_product_prim can remove unused inputs
        tmp = tensor_product_prim(
            *primals,
            *[t for t in tangents if not isinstance(t, ad.Zero)],
            output_shapes=output_shapes,
            d=multiplicator * d,
            exe=exe.map_buffers(None, lambda b: exe.out_buffers.index(b)),
            name=name + "->jvp",
            **options,
        )
        for i, t in zip(exe.out_buffers, tmp):
            out_tangents[i] = ad.add_tangents(out_tangents[i], t)

    return out_primals, tuple(out_tangents)


def tensor_product_transpose(
    cotangents: tuple[jax.Array | ad.Zero, ...],
    *inputs: jax.Array | ad.UndefinedPrimal,
    output_shapes: tuple[tuple[int, ...] | None, ...],
    d: cue.SegmentedTensorProduct,
    exe: TensorProductExecution,
    name: str,
    **options,
) -> tuple[jax.Array | ad.Zero | None, ...]:
    # The cotangents replace the outputs as inputs
    # The undefined primal inputs become outputs
    del output_shapes
    output_shapes = [None] * d.num_operands
    for comp in exe.computations:
        for oid, x in zip(comp.in_operands, comp.map_inputs(inputs)):
            if ad.is_undefined_primal(x):
                undefined_primal_shape = x.aval.shape[:-1]
                # if the following assert fails, we need to change the internal API of the primitive
                assert (
                    output_shapes[oid] is None
                    or output_shapes[oid] == undefined_primal_shape
                )
                output_shapes[oid] = undefined_primal_shape
    output_shapes = tuple(output_shapes)

    tr = exe.transpose(
        [ad.is_undefined_primal(x) for x in inputs],
        [not isinstance(x, ad.Zero) for x in cotangents],
    )
    tmp = tensor_product_prim(
        *[x for x in inputs if not ad.is_undefined_primal(x)],
        *[x for x in cotangents if not isinstance(x, ad.Zero)],
        output_shapes=output_shapes,
        d=d,
        exe=tr.map_buffers(None, lambda b: tr.out_buffers.index(b)),
        name=name + "->transpose",
        **options,
    )
    outputs = [None] * len(inputs)

    i = 0
    for b, input in enumerate(inputs):
        if ad.is_undefined_primal(input):
            if i in tr.out_buffers:
                outputs[b] = tmp[tr.out_buffers.index(i)]
            else:
                outputs[b] = ad.Zero(input.aval)
            i += 1
    return tuple(outputs)


def tensor_product_batching(
    batched_inputs: tuple[jax.Array, ...],
    batch_axes: tuple[int | None, ...],
    *,
    output_shapes: tuple[tuple[int, ...] | None, ...],
    d: cue.SegmentedTensorProduct,
    exe: TensorProductExecution,
    name: str,
    **options,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    def prepare(input: jax.Array, axis: int | None) -> jax.Array:
        if axis is None:
            return jnp.expand_dims(input, 0)
        else:
            return jnp.moveaxis(input, axis, 0)

    assert len(batched_inputs) == len(batch_axes)
    batched_inputs = [
        prepare(input, axis) for input, axis in zip(batched_inputs, batch_axes)
    ]
    new_dim = max(input.shape[0] for input in batched_inputs)

    new_output_shapes = [None] * d.num_operands
    for comp in exe.computations:
        oid = comp.out_operand
        expected = (new_dim,) + output_shapes[oid]
        if new_output_shapes[oid] is None:
            new_output_shapes[oid] = expected
        assert new_output_shapes[oid] == expected
    new_output_shapes = tuple(new_output_shapes)

    outputs = tensor_product_prim(
        *batched_inputs,
        output_shapes=new_output_shapes,
        d=d,
        exe=exe,
        name=name + "->batching",
        **options,
    )

    return outputs, (0,) * len(outputs)


tensor_product_p.def_abstract_eval(tensor_product_abstract_eval)
tensor_product_p.def_impl(partial(xla.apply_primitive, tensor_product_p))
mlir.register_lowering(
    tensor_product_p,
    mlir.lower_fun(
        partial(tensor_product_impl, "cuda"), tensor_product_p.multiple_results
    ),
    "cuda",
)
mlir.register_lowering(
    tensor_product_p,
    mlir.lower_fun(
        partial(tensor_product_impl, None), tensor_product_p.multiple_results
    ),
    None,
)
ad.primitive_jvps[tensor_product_p] = tensor_product_jvp
ad.primitive_transposes[tensor_product_p] = tensor_product_transpose
batching.primitive_batchers[tensor_product_p] = tensor_product_batching
