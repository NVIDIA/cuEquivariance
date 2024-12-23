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
import warnings
from functools import partial
from typing import List, Optional, OrderedDict, Tuple

import torch
import torch.fx

from cuequivariance import segmented_tensor_product as stp

logger = logging.getLogger(__name__)


def prod(numbers: List[int]):
    """
    This method is a workaround for script() not recognizing math.prod()
    """
    if torch.jit.is_scripting():
        product = 1
        for num in numbers:
            product *= num
        return product
    else:
        return math.prod(numbers)


class TensorProduct(torch.nn.Module):
    """
    PyTorch module that computes the last operand of the segmented tensor product defined by the descriptor.

    Args:
        descriptor (SegmentedTensorProduct): The descriptor of the segmented tensor product.
        math_dtype (torch.dtype, optional): The data type of the coefficients and calculations.
        device (torch.device, optional): The device on which the calculations are performed.
        use_fallback (bool, optional):  Determines the computation method. If `None` (default), a CUDA kernel will be used if available. If `False`, a CUDA kernel will be used, and an exception is raised if it's not available. If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Raises:
            RuntimeError: If `use_fallback` is `False` and no CUDA kernel is available.

    """

    def __init__(
        self,
        descriptor: stp.SegmentedTensorProduct,
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()
        self.descriptor = descriptor
        if math_dtype is None:
            math_dtype = torch.get_default_dtype()

        self.has_cuda = False
        self.num_operands = descriptor.num_operands

        if use_fallback is False:
            self.f = _tensor_product_cuda(descriptor, device, math_dtype)
            self.has_cuda = True
        elif use_fallback is None:
            try:
                self.f = _tensor_product_cuda(descriptor, device, math_dtype)
                self.has_cuda = True
            except NotImplementedError as e:
                logger.info(f"CUDA implementation not available: {e}")
            except ImportError as e:
                logger.warning(f"CUDA implementation not available: {e}")
                logger.warning(
                    "Did you forget to install the CUDA version of cuequivariance-ops-torch?\n"
                    "Install it with one of the following commands:\n"
                    "pip install cuequivariance-ops-torch-cu11\n"
                    "pip install cuequivariance-ops-torch-cu12"
                )

        if not self.has_cuda:
            self.f = _tensor_product_fx(descriptor, device, math_dtype, True)

    @torch.jit.ignore
    def __repr__(self):
        has_cuda_kernel = (
            "(with CUDA kernel)" if self.has_cuda else "(without CUDA kernel)"
        )
        return f"TensorProduct({self.descriptor} {has_cuda_kernel})"

    def forward(self, inputs: List[torch.Tensor]):
        r"""
        Perform the tensor product based on the specified descriptor.

        Args:
            inputs (list of torch.Tensor): The input tensors. The number of input tensors should match the number of operands in the descriptor minus one.
                Each input tensor should have a shape of (batch, operand_size) or (1, operand_size)
                where `operand_size` corresponds to the size of each operand as defined in
                the tensor product descriptor.

        Returns:
            torch.Tensor:
                The output tensor resulting from the tensor product.
                It has a shape of (batch, last_operand_size), where
                `last_operand_size` is the size of the last operand in the descriptor.

        """
        return self.f(inputs)


def to_notypeconv(t, *args, **kwargs):
    new_kwargs = kwargs.copy()
    new_kwargs.pop("dtype", None)
    new_args = [None if isinstance(a, torch.dtype) else a for a in args]
    result = t.__original_to(*new_args, **new_kwargs)
    return result


def disable_type_conv(t):
    """
    This modifier can be used on Tensors or whole Modules
    to prevent them from being modified during to(dtype=x) calls
    """
    t.__original_to = t.to
    t.to = partial(to_notypeconv, t)
    return t


def _tensor_product_fx(
    descriptor: stp.SegmentedTensorProduct,
    device: Optional[torch.device],
    math_dtype: torch.dtype,
    optimize_einsums: bool,
) -> torch.nn.Module:
    """
    batch support of this function:
    - at least one input operand should have a batch dimension (ndim=2)
    - the output operand will have a batch dimension (ndim=2)
    """
    descriptor = descriptor.remove_zero_paths()
    descriptor = descriptor.remove_empty_segments()

    num_inputs = descriptor.num_operands - 1

    if num_inputs > 0 and descriptor.num_paths > 0:
        graph = torch.fx.Graph()
        tracer = torch.fx.proxy.GraphAppendingTracer(graph)
        constants = OrderedDict()

        inputs = [
            torch.fx.Proxy(graph.placeholder(f"input_{i}"), tracer)
            for i in range(num_inputs)
        ]
        for input in inputs:
            torch._assert(input.ndim == 2, "input should have ndim=2")
        operand_subscripts = [
            f"Z{operand.subscripts}" for operand in descriptor.operands
        ]

        formula = (
            ",".join([descriptor.coefficient_subscripts] + operand_subscripts[:-1])
            + "->"
            + operand_subscripts[-1]
        )
        slices = [ope.segment_slices() for ope in descriptor.operands]

        outputs = []
        for path_idx, path in enumerate(descriptor.paths):
            segments = []
            for oid in range(num_inputs):
                seg_shape = descriptor.get_segment_shape(oid, path)
                inp = inputs[oid][..., slices[oid][path.indices[oid]]]
                if len(seg_shape) > 0:
                    inp = inp.reshape(inputs[oid].shape[:-1] + seg_shape)
                else:
                    inp = inp.reshape(inputs[oid].shape[:-1])

                segments.append(inp.to(dtype=math_dtype))

            c_tensor = disable_type_conv(
                torch.tensor(path.coefficients, dtype=math_dtype, device=device)
            )
            constants[f"c{path_idx}"] = c_tensor

            c = torch.fx.Proxy(graph.get_attr(f"c{path_idx}"), tracer=tracer).clone()
            out = torch.einsum(formula, c, *segments)
            out = out.to(dtype=inputs[0].dtype)

            seg_shape = descriptor.get_segment_shape(-1, path)
            outputs += [
                out.reshape(out.shape[: out.ndim - len(seg_shape)] + (prod(seg_shape),))
            ]

        if len(outputs) == 0:
            raise NotImplementedError("No FX implementation for empty paths")

        def _sum(tensors, *, shape=None, like=None):
            if len(tensors) == 0:
                return like.new_zeros(shape)
            out = tensors[0]
            for t in tensors[1:]:
                out = torch.add(out, t)
            return out

        batch_shape = outputs[0].shape[:-1]
        output = torch.cat(
            [
                _sum(
                    [
                        out
                        for out, path in zip(outputs, descriptor.paths)
                        if path.indices[-1] == i
                    ],
                    shape=batch_shape + (prod(descriptor.operands[-1][i]),),
                    like=outputs[0],
                )
                for i in range(descriptor.operands[-1].num_segments)
            ],
            dim=-1,
        )

        graph.output(output.node)

        graph.lint()
        constants_root = torch.nn.Module()
        for key, value in constants.items():
            constants_root.register_buffer(key, value)
        graphmod = torch.fx.GraphModule(constants_root, graph)

        if optimize_einsums:
            try:
                import opt_einsum_fx
            except ImportError:
                logger.warning(
                    "opt_einsum_fx not available.\n"
                    "To use the optimization, please install opt_einsum_fx.\n"
                    "pip install opt_einsum_fx"
                )
            else:
                example_inputs = [
                    torch.zeros((10, operand.size))
                    for operand in descriptor.operands[:num_inputs]
                ]
                graphmod = opt_einsum_fx.optimize_einsums_full(graphmod, example_inputs)
    elif num_inputs == 0:

        class _no_input(torch.nn.Module):
            def __init__(self, descriptor: stp.SegmentedTensorProduct):
                super().__init__()

                for pid, path in enumerate(descriptor.paths):
                    self.register_buffer(
                        f"c{pid}",
                        torch.tensor(
                            path.coefficients, dtype=math_dtype, device=device
                        ),
                    )

            def forward(self):
                output = torch.zeros(
                    (descriptor.operands[-1].size,), device=device, dtype=math_dtype
                )
                for pid in range(descriptor.num_paths):
                    output += torch.einsum(
                        descriptor.coefficient_subscripts
                        + "->"
                        + descriptor.operands[0].subscripts,
                        getattr(self, f"c{pid}"),
                    )
                return output

        graphmod = _no_input(descriptor)

    else:
        raise NotImplementedError(
            "No FX implementation for empty paths and non-empty inputs"
        )

    return _Wrapper(graphmod, descriptor)


class _Caller(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module


class _NoArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module()


class _OneArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0])


class _TwoArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1])


class _ThreeArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1], args[2])


class _FourArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1], args[2], args[3])


class _FiveArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1], args[2], args[3], args[4])


class _SixArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1], args[2], args[3], args[4], args[5])


class _SevenArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )


CALL_DISPATCHERS = [
    _NoArgCaller,
    _OneArgCaller,
    _TwoArgCaller,
    _ThreeArgCaller,
    _FourArgCaller,
    _FiveArgCaller,
    _SixArgCaller,
    _SevenArgCaller,
]


class _Wrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, descriptor: stp.SegmentedTensorProduct):
        super().__init__()
        self.module = CALL_DISPATCHERS[descriptor.num_operands - 1](module)
        self.descriptor = descriptor

    def forward(self, args: List[torch.Tensor]):
        if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
            for oid, arg in enumerate(args):
                torch._assert(
                    arg.ndim == 2,
                    f"input {oid} should have ndim=2",
                )
                torch._assert(
                    arg.shape[1] == self.descriptor.operands[oid].size,
                    f"input {oid} should have shape (batch, {self.descriptor.operands[oid].size})",
                )

        return self.module(args)


def _tensor_product_cuda(
    descriptor: stp.SegmentedTensorProduct,
    device: Optional[torch.device],
    math_dtype: torch.dtype,
) -> torch.nn.Module:
    logger.debug(f"Starting search for a cuda kernel for {descriptor}")

    if descriptor.num_paths == 0:
        raise NotImplementedError("No cuda kernel for empty paths.")

    if descriptor.num_operands not in (3, 4):
        raise NotImplementedError(
            "Only descriptors with 3 or 4 operands are supported."
            f" Got {descriptor.subscripts}."
        )

    if not torch.cuda.is_available():
        raise NotImplementedError("CUDA is not available.")

    # Dispatch strategy:
    # 1. try to use TensorProductUniform4x1d
    # 2. try to use FusedTensorProductOp3 or FusedTensorProductOp4

    if math_dtype in [torch.float32, torch.float64]:
        d = descriptor
        d = d.flatten_coefficient_modes(force=True)
        d = d.squeeze_modes()
        if len(d.subscripts.modes()) == 1:
            d = d.canonicalize_subscripts()
            dims = d.get_dims("u")
            d = d.split_mode("u", math.gcd(*dims))
            u = next(iter(d.get_dims("u")))

            import cuequivariance_ops_torch as ops

            if ops.TensorProductUniform1d.is_supported(
                operand_dim=[o.ndim for o in d.operands],
                operand_extent=u,
                operand_num_segments=[o.num_segments for o in d.operands],
            ):
                if descriptor.num_operands == 3:
                    return TensorProductUniform3x1d(d, device, math_dtype)
                else:
                    return TensorProductUniform4x1d(d, device, math_dtype)

    supported_targets = [
        stp.Subscripts(subscripts)
        for subscripts in [
            "u__uw_w",
            "_v_vw_w",
            "u_v_uv_u",
            "u_v_uv_v",
            "u_u_uw_w",
            "u_v_uvw_w",
            "u_u_u",
            "u_v_uv",
            "u_uv_v",
            "u__u",
            "_v_v",
        ]
    ]

    try:
        descriptor, perm = next(
            stp.dispatch(descriptor, supported_targets, "permute_all_but_last")
        )
    except StopIteration:
        raise NotImplementedError(
            f"No cuda kernel found for {descriptor}."
            " Supported targets are: " + ", ".join(str(t) for t in supported_targets)
        )

    if descriptor.num_operands == 3:
        return FusedTensorProductOp3(descriptor, perm[:2], device, math_dtype)
    elif descriptor.num_operands == 4:
        return FusedTensorProductOp4(descriptor, perm[:3], device, math_dtype)


class FusedTensorProductOp3(torch.nn.Module):
    def __init__(
        self,
        descriptor: stp.SegmentedTensorProduct,
        perm: Tuple[int, int],
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()

        self._perm = _permutation_module(perm)
        self.descriptor = descriptor.permute_operands(
            [perm.index(i) for i in range(2)] + [2]
        )

        if math_dtype not in [torch.float32, torch.float64]:
            warnings.warn(
                "cuequivariance_ops_torch.FusedTensorProductOp3 only supports math_dtype==float32 or math_dtype==float64"
            )

        import cuequivariance_ops_torch as ops

        self._f = ops.FusedTensorProductOp3(
            operand_segment_modes=[ope.subscripts for ope in descriptor.operands],
            operand_segment_offsets=[
                [s.start for s in ope.segment_slices()] for ope in descriptor.operands
            ],
            operand_segment_shapes=[ope.segments for ope in descriptor.operands],
            path_indices=descriptor.indices,
            path_coefficients=descriptor.stacked_coefficients,
            math_dtype=math_dtype,
        ).to(device=device)

    @torch.jit.ignore
    def __repr__(self) -> str:
        return f"FusedTensorProductOp3({self.descriptor} (output last operand))"

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x0, x1 = self._perm(inputs[0], inputs[1])

        if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
            logger.debug(
                f"Calling FusedTensorProductOp3: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}"
            )

        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")

        batch = max(x0.shape[0], x1.shape[0])

        if batch > 1:
            if x0.shape[0] == 1:
                x0 = x0.squeeze(0)
            if x1.shape[0] == 1:
                x1 = x1.squeeze(0)

        # ops.FusedTensorProductOp3 expects inputs
        # of shape (Z, dim) or (dim,)
        return self._f(x0, x1)


class FusedTensorProductOp4(torch.nn.Module):
    def __init__(
        self,
        descriptor: stp.SegmentedTensorProduct,
        perm: Tuple[int, int, int],
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()

        self._perm = _permutation_module(perm)
        self.descriptor = descriptor.permute_operands(
            [perm.index(i) for i in range(3)] + [3]
        )

        if math_dtype not in [torch.float32, torch.float64]:
            warnings.warn(
                "cuequivariance_ops_torch.FusedTensorProductOp4 only supports math_dtype==float32 or math_dtype==float64"
            )

        import cuequivariance_ops_torch as ops

        self._f = ops.FusedTensorProductOp4(
            operand_segment_modes=[ope.subscripts for ope in descriptor.operands],
            operand_segment_offsets=[
                [s.start for s in ope.segment_slices()] for ope in descriptor.operands
            ],
            operand_segment_shapes=[ope.segments for ope in descriptor.operands],
            path_indices=descriptor.indices,
            path_coefficients=descriptor.stacked_coefficients,
            math_dtype=math_dtype,
        ).to(device=device)

    @torch.jit.ignore
    def __repr__(self) -> str:
        return f"FusedTensorProductOp4({self.descriptor} (output last operand))"

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x0, x1, x2 = self._perm(inputs[0], inputs[1], inputs[2])

        if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
            logger.debug(
                f"Calling FusedTensorProductOp4: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}, {x2.shape}"
            )

        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x2.ndim == 2, "input should be (batch, dim) or (1, dim)")

        batch = max(x0.shape[0], x1.shape[0], x2.shape[0])

        if batch > 1:
            if x0.shape[0] == 1:
                x0 = x0.squeeze(0)
            if x1.shape[0] == 1:
                x1 = x1.squeeze(0)
            if x2.shape[0] == 1:
                x2 = x2.squeeze(0)

        # ops.FusedTensorProductOp4 expects inputs
        # of shape (Z, dim) or (dim,)
        return self._f(x0, x1, x2)


class TensorProductUniform1d(torch.nn.Module):
    def __init__(
        self,
        descriptor: stp.SegmentedTensorProduct,
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()
        import cuequivariance_ops_torch as ops

        self.descriptor = descriptor

        assert len(descriptor.subscripts.modes()) == 1
        assert descriptor.all_same_segment_shape()
        assert descriptor.coefficient_subscripts == ""
        u = next(iter(descriptor.get_dims(descriptor.subscripts.modes()[0])))

        self._f = ops.TensorProductUniform1d(
            operand_dim=[ope.ndim for ope in descriptor.operands],
            operand_extent=u,
            operand_num_segments=[ope.num_segments for ope in descriptor.operands],
            path_indices=[path.indices for path in descriptor.paths],
            path_coefficients=[float(path.coefficients) for path in descriptor.paths],
            math_dtype=math_dtype,
        ).to(device=device)


class TensorProductUniform3x1d(TensorProductUniform1d):
    @torch.jit.ignore
    def __repr__(self):
        return f"TensorProductUniform3x1d({self.descriptor} (output last operand))"

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x0, x1 = inputs

        if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
            logger.debug(
                f"Calling TensorProductUniform3x1d: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}"
            )

        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")

        # ops.TensorProductUniform1d expects inputs
        # of shape (Z, dim) or (1, dim)
        return self._f(x0, x1)


class TensorProductUniform4x1d(TensorProductUniform1d):
    @torch.jit.ignore
    def __repr__(self):
        return f"TensorProductUniform4x1d({self.descriptor} (output last operand))"

    def forward(self, inputs: List[torch.Tensor]):
        x0, x1, x2 = inputs

        if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
            logger.debug(
                f"Calling TensorProductUniform4x1d: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}, {x2.shape}"
            )

        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x2.ndim == 2, "input should be (batch, dim) or (1, dim)")

        # ops.TensorProductUniform1d expects inputs
        # of shape (Z, dim) or (1, dim)
        return self._f(x0, x1, x2)


def _permutation_module(permutation: Tuple[int, ...]):
    graph = torch.fx.Graph()
    inputs = [graph.placeholder(f"input_{i}") for i in range(len(permutation))]
    graph.output([inputs[i] for i in permutation])
    return torch.fx.GraphModule(dict(), graph, class_name="perm")
