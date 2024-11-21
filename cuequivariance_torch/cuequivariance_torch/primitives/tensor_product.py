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
from typing import List, Optional, OrderedDict, Tuple

import torch
import torch.fx

from cuequivariance import segmented_tensor_product as stp

logger = logging.getLogger(__name__)


class TensorProduct(torch.nn.Module):
    """
    PyTorch module that computes the last operand of the segmented tensor product defined by the descriptor.

    Args:
        descriptor (SegmentedTensorProduct): The descriptor of the segmented tensor product.
        math_dtype (torch.dtype, optional): The data type of the coefficients and calculations.
        device (torch.device, optional): The device on which the calculations are performed.
        optimize_fallback (bool, optional): If `True`, the fallback method is optimized. If `False`, the fallback method is used without optimization.
    """

    def __init__(
        self,
        descriptor: stp.SegmentedTensorProduct,
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        optimize_fallback: Optional[bool] = None,
    ):
        super().__init__()
        self.descriptor = descriptor

        if math_dtype is None:
            math_dtype = torch.get_default_dtype()

        try:
            self.f_cuda = _tensor_product_cuda(descriptor, device, math_dtype)
        except NotImplementedError as e:
            logger.info(f"CUDA implementation not available: {e}")
            self.f_cuda = None
        except ImportError as e:
            logger.warning(f"CUDA implementation not available: {e}")
            logger.warning(
                "Did you forget to install the CUDA version of cuequivariance-ops-torch?\n"
                "Install it with one of the following commands:\n"
                "pip install cuequivariance-ops-torch-cu11\n"
                "pip install cuequivariance-ops-torch-cu12"
            )
            self.f_cuda = None

        self.f_fx = _tensor_product_fx(
            descriptor, device, math_dtype, optimize_fallback is True
        )
        self._optimize_fallback = optimize_fallback

    def __repr__(self):
        has_cuda_kernel = (
            "(with CUDA kernel)" if self.f_cuda is not None else "(without CUDA kernel)"
        )
        return f"TensorProduct({self.descriptor} {has_cuda_kernel})"

    def forward(self, inputs: List[torch.Tensor], use_fallback: Optional[bool] = None):
        r"""
        Perform the tensor product based on the specified descriptor.

        Args:
            inputs (list of torch.Tensor): The input tensors. The number of input tensors should match the number of operands in the descriptor minus one.
                Each input tensor should have a shape of ((batch,) operand_size), where `operand_size` corresponds to the size
                of each operand as defined in the tensor product descriptor.
            use_fallback (bool, optional):  Determines the computation method. If `None` (default), a CUDA kernel will be used if available and the input
                is on CUDA. If `False`, a CUDA kernel will be used, and an exception is raised if it's not available or the
                input is not on CUDA. If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Returns:
            torch.Tensor:
                The output tensor resulting from the tensor product.
                It has a shape of (batch, last_operand_size), where
                `last_operand_size` is the size of the last operand in the descriptor.

        Raises:
            RuntimeError: If `use_fallback` is `False` and either no CUDA kernel is available or the input tensor is not on CUDA.
        """
        if any(x.numel() == 0 for x in inputs):
            use_fallback = True  # Empty tensors are not supported by the CUDA kernel

        if (
            inputs
            and inputs[0].device.type == "cuda"
            and self.f_cuda is not None
            and (use_fallback is not True)
        ):
            return self.f_cuda(*inputs)

        if use_fallback is False:
            if self.f_cuda is not None:
                raise RuntimeError("CUDA kernel available but input is not on CUDA")
            else:
                raise RuntimeError("No CUDA kernel available")

        if self._optimize_fallback is None:
            warnings.warn(
                "The fallback method is used but it has not been optimized. "
                "Consider setting optimize_fallback=True when creating the TensorProduct module."
            )
        return self.f_fx(inputs)


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
            segments = [
                inputs[oid][..., slices[oid][path.indices[oid]]]
                .reshape(
                    inputs[oid].shape[:-1] + descriptor.get_segment_shape(oid, path)
                )
                .to(dtype=math_dtype)
                for oid in range(num_inputs)
            ]
            constants[f"c{path_idx}"] = torch.tensor(
                path.coefficients, dtype=math_dtype, device=device
            ).view(
                {
                    2: torch.int16,
                    4: torch.int32,
                    8: torch.int64,
                }[math_dtype.itemsize]
            )
            c = (
                torch.fx.Proxy(graph.get_attr(f"c{path_idx}"), tracer=tracer)
                .view(math_dtype)
                .clone()
            )
            out = torch.einsum(formula, c, *segments)
            out = out.to(dtype=inputs[0].dtype)

            seg_shape = descriptor.get_segment_shape(-1, path)
            outputs += [
                out.reshape(
                    out.shape[: out.ndim - len(seg_shape)] + (math.prod(seg_shape),)
                )
            ]

        if len(outputs) == 0:
            raise NotImplementedError("No FX implementation for empty paths")

        batch_shape = outputs[0].shape[:-1]
        output = torch.cat(
            [
                _sum(
                    [
                        out
                        for out, path in zip(outputs, descriptor.paths)
                        if path.indices[-1] == i
                    ],
                    shape=batch_shape + (math.prod(descriptor.operands[-1][i]),),
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
    else:

        class _no_input_or_no_paths(torch.nn.Module):
            def __init__(self, descriptor: stp.SegmentedTensorProduct):
                super().__init__()

                for pid, path in enumerate(descriptor.paths):
                    self.register_buffer(
                        f"c{pid}",
                        torch.tensor(
                            path.coefficients, dtype=math_dtype, device=device
                        ),
                    )

            def forward(self, *args):
                shape = torch.broadcast_shapes(*[arg.shape[:-1] for arg in args])
                output = torch.zeros(
                    shape + (descriptor.operands[-1].size,),
                    device=device,
                    dtype=math_dtype,
                )
                for pid in range(descriptor.num_paths):
                    output += torch.einsum(
                        descriptor.coefficient_subscripts
                        + "->"
                        + descriptor.operands[0].subscripts,
                        getattr(self, f"c{pid}"),
                    )
                return output

        graphmod = _no_input_or_no_paths(descriptor)

    return _Wrapper(graphmod, descriptor)


class _Wrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, descriptor: stp.SegmentedTensorProduct):
        super().__init__()
        self.module = module
        self.descriptor = descriptor

    def forward(self, args):
        for oid, arg in enumerate(args):
            torch._assert(
                arg.shape[-1] == self.descriptor.operands[oid].size,
                "input shape[-1] does not match operand size",
            )

        shape = torch.broadcast_shapes(*[arg.shape[:-1] for arg in args])

        args = [
            (
                arg.expand(shape + (arg.shape[-1],)).reshape(
                    (math.prod(shape), arg.shape[-1])
                )
                if math.prod(arg.shape[:-1]) > 1
                else arg.reshape((math.prod(arg.shape[:-1]), arg.shape[-1]))
            )
            for arg in args
        ]

        logger.debug(
            f"Calling torch.fx tensor product: {self.descriptor}, input shapes: {', '.join(str(arg.shape) for arg in args)}"
        )
        out = self.module(*args)

        return out.reshape(shape + (out.shape[-1],))


def _sum(tensors, *, shape=None, like=None):
    if len(tensors) == 0:
        return like.new_zeros(shape)
    out = tensors[0]
    for t in tensors[1:]:
        out += t
    return out


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


def _reshape(x: torch.Tensor, leading_shape: tuple[int, ...]) -> torch.Tensor:
    # Make x have shape (Z, x.shape[-1]) or (x.shape[-1],)
    if math.prod(leading_shape) > 1 and math.prod(x.shape[:-1]) == 1:
        return x.reshape((x.shape[-1],))
    else:
        return x.expand(leading_shape + (x.shape[-1],)).reshape(
            (math.prod(leading_shape), x.shape[-1])
        )


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

    def __repr__(self) -> str:
        return f"TensorProductCUDA({self.descriptor} (output last operand))"

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        x0, x1 = self._perm(x0, x1)
        assert x0.ndim >= 1, x0.ndim
        assert x1.ndim >= 1, x1.ndim

        shape = torch.broadcast_shapes(x0.shape[:-1], x1.shape[:-1])
        x0 = _reshape(x0, shape)
        x1 = _reshape(x1, shape)

        logger.debug(
            f"Calling FusedTensorProductOp3: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}"
        )

        out = self._f(x0, x1)

        return out.reshape(shape + (out.shape[-1],))


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

    def __repr__(self) -> str:
        return f"TensorProductCUDA({self.descriptor} (output last operand))"

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        x0, x1, x2 = self._perm(x0, x1, x2)
        assert x0.ndim >= 1, x0.ndim
        assert x1.ndim >= 1, x1.ndim
        assert x2.ndim >= 1, x2.ndim

        shape = torch.broadcast_shapes(x0.shape[:-1], x1.shape[:-1], x2.shape[:-1])
        x0 = _reshape(x0, shape)
        x1 = _reshape(x1, shape)
        x2 = _reshape(x2, shape)

        logger.debug(
            f"Calling FusedTensorProductOp4: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}, {x2.shape}"
        )

        out = self._f(x0, x1, x2)

        return out.reshape(shape + (out.shape[-1],))


class TensorProductUniform3x1d(torch.nn.Module):
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

    def __repr__(self):
        return f"TensorProductCUDA({self.descriptor} (output last operand))"

    def forward(self, x0, x1):
        assert x0.ndim >= 1, x0.ndim
        assert x1.ndim >= 1, x1.ndim

        shape = torch.broadcast_shapes(x0.shape[:-1], x1.shape[:-1])
        x0 = _reshape(x0, shape)
        x1 = _reshape(x1, shape)

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)
        if x1.ndim == 1:
            x1 = x1.unsqueeze(0)

        logger.debug(
            f"Calling TensorProductUniform3x1d: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}"
        )

        out = self._f(x0, x1)

        return out.reshape(shape + (out.shape[-1],))


class TensorProductUniform4x1d(torch.nn.Module):
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

    def __repr__(self):
        return f"TensorProductCUDA({self.descriptor} (output last operand))"

    def forward(self, x0, x1, x2):
        assert x0.ndim >= 1, x0.ndim
        assert x1.ndim >= 1, x1.ndim
        assert x2.ndim >= 1, x2.ndim

        shape = torch.broadcast_shapes(x0.shape[:-1], x1.shape[:-1], x2.shape[:-1])
        x0 = _reshape(x0, shape)
        x1 = _reshape(x1, shape)
        x2 = _reshape(x2, shape)

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)
        if x1.ndim == 1:
            x1 = x1.unsqueeze(0)
        if x2.ndim == 1:
            x2 = x2.unsqueeze(0)

        logger.debug(
            f"Calling TensorProductUniform4x1d: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}, {x2.shape}"
        )

        out = self._f(x0, x1, x2)

        return out.reshape(shape + (out.shape[-1],))


def _permutation_module(permutation: Tuple[int, ...]):
    graph = torch.fx.Graph()
    inputs = [graph.placeholder(f"input_{i}") for i in range(len(permutation))]
    graph.output([inputs[i] for i in permutation])
    return torch.fx.GraphModule(dict(), graph, class_name="perm")
