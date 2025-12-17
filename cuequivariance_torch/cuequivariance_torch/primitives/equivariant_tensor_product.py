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
from typing import List, Optional, Union

import torch
from cuequivariance.group_theory.irreps_array.misc_ui import default_layout

import cuequivariance as cue
import cuequivariance_torch as cuet


class Dispatcher(torch.nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.tp = tp


class Transpose1Dispatcher(Dispatcher):
    def forward(self, inputs: List[torch.Tensor]):
        ret = inputs.copy()
        ret[0] = self.tp[0](ret[0])
        return ret


class Transpose2Dispatcher(Dispatcher):
    def forward(self, inputs: List[torch.Tensor]):
        ret = inputs.copy()
        ret[0] = self.tp[0](ret[0])
        ret[1] = self.tp[1](ret[1])
        return ret


class Transpose3Dispatcher(Dispatcher):
    def forward(self, inputs: List[torch.Tensor]):
        ret = inputs.copy()
        ret[0] = self.tp[0](ret[0])
        ret[1] = self.tp[1](ret[1])
        ret[2] = self.tp[2](ret[2])
        return ret


class Transpose4Dispatcher(Dispatcher):
    def forward(self, inputs: List[torch.Tensor]):
        ret = inputs.copy()
        ret[0] = self.tp[0](ret[0])
        ret[1] = self.tp[1](ret[1])
        ret[2] = self.tp[2](ret[2])
        ret[3] = self.tp[3](ret[3])
        return ret


TRANSPOSE_DISPATCHERS = [
    Transpose1Dispatcher,
    Transpose2Dispatcher,
    Transpose3Dispatcher,
    Transpose4Dispatcher,
]


class TPDispatcher(cuet._Wrapper):
    def forward(
        self,
        inputs: List[torch.Tensor],
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if indices is not None:
            # TODO: at some point we will have kernel for this
            inputs[0] = inputs[0][indices]
        return self.module(inputs)


class SymmetricTPDispatcher(Dispatcher):
    def forward(
        self,
        inputs: List[torch.Tensor],
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert indices is None
        return self.tp(inputs[0])


class IWeightedSymmetricTPDispatcher(Dispatcher):
    def forward(
        self,
        inputs: List[torch.Tensor],
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x0, x1 = inputs
        if indices is None:
            torch._assert(
                x0.ndim == 2,
                f"Expected x0 to have shape (batch, dim), got {x0.shape}",
            )
            indices = torch.arange(x1.shape[0], dtype=torch.int32, device=x1.device)
            indices = indices % x0.shape[0]
        return self.tp(x0, indices, x1)


class EquivariantTensorProduct(torch.nn.Module):
    r"""Equivariant tensor product.

    Args:
        e (cuequivariance.EquivariantTensorProduct): Equivariant tensor product.
        layout (IrrepsLayout): layout for inputs and output.
        layout_in (IrrepsLayout): layout for inputs.
        layout_out (IrrepsLayout): layout for output.
        device (torch.device): device of the Module.
        math_dtype (torch.dtype): dtype for internal computations.
        use_fallback (bool, optional):  Determines the computation method. If `None` (default), a CUDA kernel will be used if available. If `False`, a CUDA kernel will be used, and an exception is raised if it's not available. If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.
    Raises:
        RuntimeError: If `use_fallback` is `False` and no CUDA kernel is available.

    Examples:
        >>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        >>> e = cue.descriptors.fully_connected_tensor_product(
        ...    cue.Irreps("SO3", "2x1"), cue.Irreps("SO3", "2x1"), cue.Irreps("SO3", "2x1")
        ... )
        >>> w = torch.ones(1, e.inputs[0].dim, device=device)
        >>> x1 = torch.ones(17, e.inputs[1].dim, device=device)
        >>> x2 = torch.ones(17, e.inputs[2].dim, device=device)
        >>> tp = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul, device=device)
        >>> tp(w, x1, x2)
        tensor([[0., 0., 0., 0., 0., 0.],...)

        You can optionally index the first input tensor:

        >>> w = torch.ones(3, e.inputs[0].dim, device=device)
        >>> indices = torch.randint(3, (17,))
        >>> tp(w, x1, x2, indices=indices)
        tensor([[0., 0., 0., 0., 0., 0.],...)
    """

    def __init__(
        self,
        e: cue.EquivariantTensorProduct,
        *,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[
            Union[cue.IrrepsLayout, tuple[Optional[cue.IrrepsLayout], ...]]
        ] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()

        # TODO: remove this when re-design
        if isinstance(e, cue.EquivariantPolynomial):
            assert e.num_outputs == 1
            for ope, stp in e.polynomial.operations:
                inputs = list(range(e.num_inputs))
                output = e.num_inputs
                expected = tuple(
                    inputs[: stp.num_operands - 1]
                    + [inputs[-1]] * max(0, stp.num_operands - e.num_operands)
                    + [output]
                )
                assert ope.buffers == expected, f"{ope.buffers} != {expected}"
            e = cue.EquivariantTensorProduct(
                [stp for _, stp in e.polynomial.operations], e.inputs + e.outputs
            )

        if not isinstance(layout_in, tuple):
            layout_in = (layout_in,) * e.num_inputs
        if len(layout_in) != e.num_inputs:
            raise ValueError(
                f"Expected {e.num_inputs} input layouts, got {len(layout_in)}"
            )
        layout_in = tuple(lay or layout for lay in layout_in)
        layout_out = layout_out or layout
        del layout

        self.etp = e

        transpose_in = torch.nn.ModuleList()
        for layout_used, input_expected in zip(layout_in, e.inputs):
            if isinstance(input_expected, cue.IrrepsAndLayout):
                layout_used = default_layout(layout_used)
                transpose_in.append(
                    cuet.TransposeIrrepsLayout(
                        input_expected.irreps,
                        source=layout_used,
                        target=input_expected.layout,
                        device=device,
                        use_fallback=use_fallback,
                    )
                )
            else:
                assert layout_used is None
                transpose_in.append(torch.nn.Identity())

        # script() requires literal addressing and fails to eliminate dead branches
        self.transpose_in = TRANSPOSE_DISPATCHERS[len(transpose_in) - 1](transpose_in)

        if isinstance(e.output, cue.IrrepsAndLayout):
            layout_out = default_layout(layout_out)
            self.transpose_out = cuet.TransposeIrrepsLayout(
                e.output.irreps,
                source=e.output.layout,
                target=layout_out,
                device=device,
                use_fallback=use_fallback,
            )
        else:
            assert layout_out is None
            self.transpose_out = torch.nn.Identity()

        if (
            len(e.ds) > 1
            or any(d.num_operands != e.num_inputs + 1 for d in e.ds)
            or any(
                d.num_operands == 2 for d in e.ds
            )  # special case for Spherical Harmonics ls = [1]
        ):
            if e.num_inputs == 1:
                self.tp = SymmetricTPDispatcher(
                    cuet.SymmetricTensorProduct(
                        e.ds,
                        device=device,
                        math_dtype=math_dtype,
                        use_fallback=use_fallback,
                    )
                )
            elif e.num_inputs == 2:
                self.tp = IWeightedSymmetricTPDispatcher(
                    cuet.IWeightedSymmetricTensorProduct(
                        e.ds,
                        device=device,
                        math_dtype=math_dtype,
                        use_fallback=use_fallback,
                    )
                )
            else:
                raise NotImplementedError("This should not happen")
        else:
            tp = cuet.TensorProduct(
                e.ds[0],
                device=device,
                math_dtype=math_dtype,
                use_fallback=use_fallback,
            )
            self.tp = TPDispatcher(tp, tp.descriptor)

        self.operands_dims = [op.dim for op in e.operands]

    def extra_repr(self) -> str:
        return str(self.etp)

    def forward(
        self,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
        x3: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        If ``indices`` is not None, the first input is indexed by ``indices``.
        """

        if x3 is not None and x2 is not None and x1 is not None:
            inputs = [x0, x1, x2, x3]
        elif x2 is not None and x1 is not None:
            inputs = [x0, x1, x2]
        elif x1 is not None:
            inputs = [x0, x1]
        else:
            inputs = [x0]

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            if len(inputs) != self.etp.num_inputs:
                raise ValueError(
                    f"Expected {self.etp.num_inputs} input tensors, got {len(inputs)}"
                )
            for oid, input in enumerate(inputs):
                torch._assert(
                    input.ndim == 2,
                    f"input {oid} should have ndim=2",
                )
                torch._assert(
                    input.shape[1] == self.operands_dims[oid],
                    f"input {oid} should have shape (batch, {self.operands_dims[oid]}), got {input.shape}",
                )

        # Transpose inputs
        inputs = self.transpose_in(inputs)

        # Compute tensor product
        output = self.tp(inputs, indices)

        # Transpose output
        output = self.transpose_out(output)

        return output
