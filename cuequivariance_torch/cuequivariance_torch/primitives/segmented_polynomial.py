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

import cuequivariance as cue
from cuequivariance_torch.primitives.segmented_polynomial_fused_tp import (
    SegmentedPolynomialFusedTP,
)
from cuequivariance_torch.primitives.segmented_polynomial_naive import (
    SegmentedPolynomialNaive,
)
from cuequivariance_torch.primitives.segmented_polynomial_uniform_1d import (
    SegmentedPolynomialFromUniform1dJit,
)


class SegmentedPolynomial(nn.Module):
    """
    PyTorch module that computes a segmented polynomial.

    Currently, it supports segmented polynomials where all segment sizes are the same,
    and each operand is one or zero dimensional.

    Args:
        polynomial: The segmented polynomial to compute, an instance of
            `cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`.
        method: Specifies the implementation method to use. Options are:
            - "naive": Uses a naive PyTorch implementation. It always works but is not optimized.
            - "uniform_1d": Uses a CUDA implementation for polynomials with a single uniform mode.
        math_dtype: Data type for computational operations, defaulting to float32.
        output_dtype_map: Optional list that, for each output buffer, specifies
            the index of the input buffer from which it inherits its data type.
            -1 means the math_dtype is used.
            Default 0 if there are input tensors, otherwise -1.
        name: Optional name for the operation. Defaults to "segmented_polynomial".
    """

    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        method: str = "",
        math_dtype: torch.dtype = torch.float32,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
    ):
        super().__init__()

        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs

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
            )
            method = "uniform_1d"
        if method == "uniform_1d":
            self.m = SegmentedPolynomialFromUniform1dJit(
                polynomial, math_dtype, output_dtype_map, name
            )
        elif method == "naive":
            self.m = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
        elif method == "fused_tp":
            self.m = SegmentedPolynomialFusedTP(
                polynomial, math_dtype, output_dtype_map, name
            )
        else:
            raise ValueError(f"Invalid method: {method}")

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
        """
        Computes the segmented polynomial based on the specified descriptor.

        Args:
            inputs: The input tensors. The number of input tensors must match
                the number of input buffers in the descriptor.
                Each input tensor should have a shape of (batch, operand_size) or
                (1, operand_size) or (index, operand_size) in the indexed case.
                Here, `operand_size` is the size of each operand as defined in
                the descriptor.
            input_indices: A dictionary that contains an optional indexing tensor
                for each input tensor. The key is the index into the inputs.
                If a key is not present, no indexing takes place.
                The contents of the index tensor must be suitable to index the
                input tensor (i.e. 0 <= index_tensor[i] < input.shape[0].
            output_shapes: A dictionary specifying the size of the output batch
                dimensions using Tensors. We only read shape_tensor.shape[0].
                This is mandatory if the output tensor is indexed. Otherwise,
                the default shape is (batch, operand_size).
            output_indices: A dictionary that contains an optional indexing tensor
                for each output tensor. See input_indices for details.

        Returns:
            List[torch.Tensor]:
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
        torch._assert(
            len(inputs) == self.num_inputs,
            "the number of inputs must match the number of inputs of the polynomial",
        )

        for k, v in input_indices.items():
            torch._assert(0 <= k < self.num_inputs, "input index must be in range")
            torch._assert(v.ndim == 1, "input index must be one-dimensional")
            torch._assert(
                v.dtype in [torch.int32, torch.int64], "input index must be integral"
            )
        for k, v in output_indices.items():
            torch._assert(0 <= k < self.num_outputs, "output index must be in range")
            torch._assert(v.ndim == 1, "input index must be one-dimensional")
            torch._assert(
                v.dtype in [torch.int32, torch.int64], "input index must be integral"
            )
        for k, v in output_shapes.items():
            torch._assert(0 <= k < self.num_outputs, "output index must be in range")
            torch._assert(v.ndim == 2, "output shape must be two-dimensional")

        return self.m(inputs, input_indices, output_shapes, output_indices)
