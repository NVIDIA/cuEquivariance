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

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from cuequivariance_torch.primitives.segmented_polynomial_fused_tp import (
    SegmentedPolynomialFusedTP,
)
from cuequivariance_torch.primitives.segmented_polynomial_indexed_linear import (
    SegmentedPolynomialIndexedLinear,
)
from cuequivariance_torch.primitives.segmented_polynomial_naive import (
    SegmentedPolynomialNaive,
)
from cuequivariance_torch.primitives.segmented_polynomial_uniform_1d import (
    SegmentedPolynomialFromUniform1dJit,
)

import cuequivariance as cue

try:
    import cuequivariance_ops_torch  # noqa: F401

    HAS_CUE_OPS = True
except ImportError:
    HAS_CUE_OPS = False


class SegmentedPolynomial(nn.Module):
    """PyTorch module that computes a segmented polynomial.

    Args:
        polynomial: The segmented polynomial to compute, an instance of
            `cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`.
        method: Specifies the implementation method to use. Options are:

            - ``"naive"``: Uses a naive PyTorch implementation. It always works but is not optimized.
            - ``"uniform_1d"``: Uses a CUDA implementation for polynomials with a single uniform mode.
            - ``"fused_tp"``: Uses a CUDA implementation for polynomials with 3- and 4-operand contractions.
            - ``"indexed_linear"``: Uses a CUDA implementation for linear layers with indexed weights.

        math_dtype: Optional data type for computational operations.
            If specified, internal buffers will be of this dtype,
            and operands will be converted to this type for all computations.

            Values can be specified as a string corresponding to a torch.dtype,
            or as a torch.dtype.
            For some methods, special values can be used:

            - For method ``"naive"``: Any torch.dtype or corresponding string.
            - For method ``"uniform_1d"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"fused_tp"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"indexed_linear"``: this is not supported and will be ignored.

            .. note::
               This will not be affected by changes to the module dtype,
               and not all methods support all dtypes.

            If ``math_dtype`` is not specified:

            - For method ``"naive"``, the dtype of the input tensors will be used.
            - For method ``"uniform_1d"``, the dtype of the input tensors will be used if allowed
              (FP32 or FP64), otherwise float32 will be used.
            - For method ``"fused_tp"``, the default dtype (FP32) will be used.
            - For method ``"indexed_linear"``, the dtype of the input tensors will be used.

        output_dtype_map: Optional list that, for each output buffer, specifies
            the index of the input buffer from which it inherits its data type.
            -1 means the math_dtype is used.
            Default is 0 if there are input tensors, otherwise -1.
        name: Optional name for the operation. Defaults to "segmented_polynomial".

    Examples:
        Basic usage with spherical harmonics:

        >>> import torch
        >>> import cuequivariance as cue
        >>> from cuequivariance_torch import SegmentedPolynomial
        >>>
        >>> # Create spherical harmonics polynomial
        >>> poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2]).polynomial
        >>> sp = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Compute spherical harmonics for unit vector along y-axis
        >>> x = torch.tensor([[0.0, 1.0, 0.0]])
        >>> result = sp([x])
        >>> print(result[0].shape)
        torch.Size([1, 9])

        Example with a linear layer:

        >>> # Create a linear transformation
        >>> input_irreps = cue.Irreps(cue.O3, "5x0e + 3x1o")
        >>> output_irreps = cue.Irreps(cue.O3, "4x0e + 2x1o")
        >>> poly = cue.descriptors.linear(input_irreps, output_irreps).polynomial
        >>>
        >>> # Create the module
        >>> linear = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Create random weights and input
        >>> weights = torch.randn(1, poly.inputs[0].size)
        >>> x = torch.randn(10, poly.inputs[1].size)
        >>>
        >>> # Forward pass
        >>> result = linear([weights, x])
        >>> print(result[0].shape)
        torch.Size([10, 10])

        Example with indexed operations:

        >>> # Create indexed weights for different elements
        >>> weights = torch.randn(3, poly.inputs[0].size)  # 3 different weight sets
        >>> x = torch.randn(5, poly.inputs[1].size)        # 5 input vectors
        >>>
        >>> # Index tensor specifying which weights to use for each input
        >>> weight_indices = torch.tensor([0, 1, 0, 2, 1])  # Use weights 0,1,0,2,1
        >>>
        >>> result = linear([weights, x],
        ...                input_indices={0: weight_indices})
        >>> print(result[0].shape)
        torch.Size([5, 10])
    """

    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        method: str = "",
        math_dtype: str | torch.dtype = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
    ):
        super().__init__()

        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.method = method
        self.repr = polynomial.__repr__()

        if method == "":
            warnings.warn(
                "Hello! It looks like you're using code that was written for an older version of this library.\n"
                "Starting in v0.6.0, the `method` argument is suggested when using `SegmentedPolynomial()`.\n"
                "This change helps ensure you get optimal performance by explicitly choosing the computation method.\n"
                "For the moment, we will default to the 'uniform_1d' method.\n\n"
                "To remove this warning, add a `method` parameter to your function call. Here are the available options:\n"
                "• 'naive' - Works everywhere but not optimized (good for testing)\n"
                "• 'uniform_1d' - Fast CUDA implementation for single uniform mode polynomials\n"
                "• 'fused_tp' - A more general CUDA implementation, supporting many 3 and 4 operands contractions.\n"
                "• 'indexed_linear' - A CUDA implementation for linear layers with indexed weights.\n"
            )
            method = "uniform_1d"

        if not isinstance(polynomial, cue.SegmentedPolynomial):
            raise ValueError(
                f"The polynomial is not a cue.SegmentedPolynomial, but a {type(polynomial)}",
                "Did you forget to call `.polynomial` on the descriptor?",
            )

        if method != "naive" and not HAS_CUE_OPS:
            method = "naive"
            warnings.warn(
                "cuequivariance_ops_torch is not available. Falling back to naive implementation."
            )

        if method == "uniform_1d":
            self.m = SegmentedPolynomialFromUniform1dJit(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        elif method == "naive":
            self.m = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        elif method == "fused_tp":
            self.m = SegmentedPolynomialFusedTP(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
        elif method == "indexed_linear":
            self.m = SegmentedPolynomialIndexedLinear(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        else:
            raise ValueError(f"Invalid method: {method}")

    def __repr__(self):
        return self.repr + f"\n{super().__repr__()}"

    # For torch.jit.trace, we cannot pass explicit optionals,
    # so these must be passed as kwargs then.
    # List[Optional[Tensor]] does not work for similar reasons, hence, Dict
    # is the only option.
    # Also, shapes cannot be passed as integers, so they are passed via a
    # (potentially small-strided) tensor with the right shape.
    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """Compute the segmented polynomial based on the specified descriptor.

        Args:
            inputs: The input tensors. The number of input tensors must match
                the number of input buffers in the descriptor.
                Each input tensor should have a shape of ``(batch, operand_size)`` or
                ``(1, operand_size)`` or ``(index, operand_size)`` in the indexed case.
                Here, ``operand_size`` is the size of each operand as defined in
                the descriptor.
            input_indices: A dictionary that contains an optional indexing tensor
                for each input tensor. The key is the index into the inputs.
                If a key is not present, no indexing takes place.
                The contents of the index tensor must be suitable to index the
                input tensor (i.e., ``0 <= index_tensor[i] < input.shape[0]``).

                .. note::
                   Method ``"indexed_linear"`` requires the indices to be sorted.

            output_shapes: A dictionary specifying the size of the output batch
                dimensions using Tensors. We only read ``shape_tensor.shape[0]``.
                This is mandatory if the output tensor is indexed. Otherwise,
                the default shape is ``(batch, operand_size)``.
            output_indices: A dictionary that contains an optional indexing tensor
                for each output tensor. See ``input_indices`` for details.

        Returns:
            The output tensors resulting from the segmented polynomial.
            Their shapes are specified just like the inputs.
        """

        # General checks
        empty_dict: Dict[int, torch.Tensor] = {}
        if input_indices is None:
            input_indices = dict(empty_dict)
        if output_shapes is None:
            output_shapes = dict(empty_dict)
        if output_indices is None:
            output_indices = dict(empty_dict)

        inputs = list(inputs)
        if not torch.jit.is_scripting():
            if (
                not torch.jit.is_tracing()
                and not torch.compiler.is_compiling()
                and not torch.fx._symbolic_trace.is_fx_tracing()
            ):
                torch._assert(
                    len(inputs) == self.num_inputs,
                    "the number of inputs must match the number of inputs of the polynomial",
                )

                for k, v in input_indices.items():
                    torch._assert(
                        0 <= k < self.num_inputs, "input index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                for k, v in output_indices.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                for k, v in output_shapes.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 2, "output shape must be two-dimensional")

                # If the input is on the CPU and we're using fused_tp, we need to fall back to naive
                if (
                    inputs[0].device == torch.device("cpu")
                    and self.method == "fused_tp"
                ):
                    warnings.warn(
                        "Fused TP is not supported on CPU. Falling back to naive implementation."
                    )
                    return self.fallback(
                        inputs, input_indices, output_shapes, output_indices
                    )

        return self.m(inputs, input_indices, output_shapes, output_indices)
