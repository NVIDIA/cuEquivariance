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

from itertools import accumulate
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import cuequivariance as cue

try:
    from cuequivariance_ops_torch.tensor_product_uniform_1d_jit import (
        BATCH_DIM_AUTO,
        BATCH_DIM_BATCHED,
        BATCH_DIM_INDEXED,
        BATCH_DIM_SHARED,
    )

    try:
        # keep us an option to be independent of the torch.library machinery
        from cuequivariance_ops_torch.tensor_product_uniform_1d_jit import (
            tensor_product_uniform_1d_jit,
        )
    except Exception:

        def tensor_product_uniform_1d_jit(
            name: str,
            math_dtype: torch.dtype,
            operand_extent: int,
            num_inputs: int,
            num_outputs: int,
            num_index: int,
            buffer_dim: List[int],
            buffer_num_segments: List[int],
            batch_dim: List[int],
            index_buffer: List[int],
            dtypes: List[int],
            num_operations: int,
            num_operands: List[int],
            operations: List[int],
            num_paths: List[int],
            path_indices_start: List[int],
            path_coefficients_start: List[int],
            path_indices: List[int],
            path_coefficients: List[float],
            batch_size: int,
            tensors: List[torch.Tensor],
        ) -> List[torch.Tensor]:
            return torch.ops.cuequivariance.tensor_product_uniform_1d_jit(
                name,
                math_dtype,
                operand_extent,
                num_inputs,
                num_outputs,
                num_index,
                buffer_dim,
                buffer_num_segments,
                batch_dim,
                index_buffer,
                dtypes,
                num_operations,
                num_operands,
                operations,
                num_paths,
                path_indices_start,
                path_coefficients_start,
                path_indices,
                path_coefficients,
                batch_size,
                tensors,
            )
except ImportError:
    tensor_product_uniform_1d_jit = None


class SegmentedPolynomialFromUniform1dJit(nn.Module):
    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        math_dtype: Optional[str | torch.dtype] = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
    ):
        super().__init__()

        if tensor_product_uniform_1d_jit is None:
            raise ImportError(
                "Failed to construct SegmentedPolynomialFromUniform1dJit: "
                "the 'cuequivariance_ops_torch.tensor_product_uniform_1d_jit' extension "
                "is not available. Please install 'cuequivariance_ops_torch' "
                "for method 'uniform_1d'."
            )

        if not torch.jit.is_scripting():
            try:
                polynomial = polynomial.flatten_coefficient_modes()
            except ValueError as e:
                raise ValueError(
                    "Failed to prepare polynomial for method 'uniform_1d': this method "
                    "does not support coefficient modes and automatic flattening with "
                    "'flatten_coefficient_modes()' failed.\n"
                    f"Original error:\n{e}\n"
                    f"Problematic polynomial:\n{polynomial}"
                ) from e
        else:
            polynomial = polynomial.flatten_coefficient_modes()

        polynomial = polynomial.squeeze_modes()

        operand_extent = None
        for o in polynomial.operands:
            torch._assert(
                o.ndim in [0, 1],
                "For method 'uniform_1d', only 0D (scalar) or 1D operands are supported.",
            )
            torch._assert(
                all(len(s) == o.ndim for s in o.segments),
                "For method 'uniform_1d', all segments of an operand must have the same "
                "number of dimensions as the operand.",
            )
            torch._assert(
                o.all_same_segment_shape(),
                "For method 'uniform_1d', all segments of a given operand must have the "
                "same shape. If segment extents differ but share a non-trivial greatest "
                "common divisor, you may preprocess the polynomial with 'split_mode()' "
                "on the corresponding mode to factor out this divisor and obtain "
                "uniform segment extents. Using 'split_mode()' in this way increases "
                "the number of segments and tensor-product paths, which can degrade "
                "performance. In particular, choosing a very small common divisor "
                "relative to the largest extent (for example, extents [64, 128, 2] "
                "with gcd=2) creates many short segments, and segment extents smaller "
                "than 32 or not divisible by 32 are generally inefficient on the GPU.",
            )
            if o.ndim == 1 and len(o.segments) > 0:
                if operand_extent is None:
                    (operand_extent,) = o.segment_shape
                else:
                    torch._assert(
                        (operand_extent,) == o.segment_shape,
                        "For method 'uniform_1d', all 1D operands must share the same "
                        "segment extent (a single uniform mode across operands).",
                    )
        if operand_extent is None:
            operand_extent = 1

        for o, stp in polynomial.operations:
            torch._assert(
                stp.num_operands == len(o.buffers),
                "In method 'uniform_1d', the number of STP operands must match the "
                "number of buffers referenced by each operation.",
            )
            torch._assert(
                stp.coefficient_subscripts == "",
                "In method 'uniform_1d', coefficients must be scalar "
                "(coefficient_subscripts must be empty).",
            )

        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.name = name
        if isinstance(math_dtype, str):
            try:
                # Accept string names like "float32" or "float64"
                math_dtype = getattr(torch, math_dtype)
            except AttributeError:
                raise ValueError(
                    f"Invalid math_dtype '{math_dtype}' for method 'uniform_1d'. "
                    "Expected 'float32', 'float64', or None."
                )
        if math_dtype not in [None, torch.float32, torch.float64]:
            raise ValueError(
                "For method 'uniform_1d', math_dtype must be 'float32', "
                "'float64', or None; got "
                f"{math_dtype}"
            )
        self.math_dtype = math_dtype
        self.operand_extent = operand_extent
        self.buffer_dim = [o.ndim for o in polynomial.operands]
        torch._assert(
            all(buffer_dim in [0, 1] for buffer_dim in self.buffer_dim),
            "For method 'uniform_1d', buffer dimensions must be 0 or 1.",
        )
        self.buffer_num_segments = [len(o.segments) for o in polynomial.operands]
        default_dtype_map = [
            0 if polynomial.num_inputs >= 1 else -1
        ] * polynomial.num_outputs
        self.dtypes = list(range(self.num_inputs)) + (
            default_dtype_map if output_dtype_map is None else output_dtype_map
        )
        self.num_operations = len(polynomial.operations)
        self.num_operands = [len(o.buffers) for o, stp in polynomial.operations]
        self.operations = [b for o, stp in polynomial.operations for b in o.buffers]
        self.num_paths = [stp.num_paths for o, stp in polynomial.operations]
        self.path_indices_start = [0] + list(
            accumulate(
                [stp.num_paths * stp.num_operands for o, stp in polynomial.operations]
            )
        )[:-1]
        self.path_coefficients_start = [0] + list(
            accumulate([stp.num_paths for o, stp in polynomial.operations])
        )[:-1]
        self.path_indices = [
            i for o, stp in polynomial.operations for p in stp.paths for i in p.indices
        ]
        self.path_coefficients = [
            float(p.coefficients) for o, stp in polynomial.operations for p in stp.paths
        ]

        self.BATCH_DIM_AUTO = BATCH_DIM_AUTO
        self.BATCH_DIM_SHARED = BATCH_DIM_SHARED
        self.BATCH_DIM_BATCHED = BATCH_DIM_BATCHED
        self.BATCH_DIM_INDEXED = BATCH_DIM_INDEXED

    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Dict[int, torch.Tensor],
        output_shapes: Dict[int, torch.Tensor],
        output_indices: Dict[int, torch.Tensor],
    ):
        num_index = 0
        batch_dim = [self.BATCH_DIM_AUTO] * (self.num_inputs + self.num_outputs)
        index_buffer = [-1] * (self.num_inputs + self.num_outputs)
        tensors = list(inputs)

        if self.math_dtype is None:
            if inputs[0].dtype in [torch.float32, torch.float64]:
                math_dtype = inputs[0].dtype
            else:
                math_dtype = torch.float32
        else:
            math_dtype = self.math_dtype

        for idx_pos, idx_tensor in input_indices.items():
            batch_dim[idx_pos] = self.BATCH_DIM_INDEXED
            tensors.append(idx_tensor)
            index_buffer[idx_pos] = num_index
            num_index += 1
            index_buffer.append(inputs[idx_pos].shape[0])

        for idx_pos, idx_tensor in output_indices.items():
            batch_dim[idx_pos + self.num_inputs] = self.BATCH_DIM_INDEXED
            tensors.append(idx_tensor)
            index_buffer[idx_pos + self.num_inputs] = num_index
            num_index += 1
            torch._assert(
                idx_pos in output_shapes,
                "output shapes must be provided for output indices",
            )
            index_buffer.append(output_shapes[idx_pos].size(0))

        batch_size = self.BATCH_DIM_AUTO
        for idx_pos, idx_shape in output_shapes.items():
            if batch_dim[idx_pos + self.num_inputs] == self.BATCH_DIM_AUTO:
                if idx_shape.size(0) == 1:
                    batch_dim[idx_pos + self.num_inputs] = self.BATCH_DIM_SHARED
                else:
                    torch._assert(
                        batch_size == self.BATCH_DIM_AUTO
                        or batch_size == idx_shape.size(0),
                        "batch size must be auto or the output shape",
                    )
                    batch_dim[idx_pos + self.num_inputs] = self.BATCH_DIM_BATCHED
                    batch_size = idx_shape.size(0)

        return tensor_product_uniform_1d_jit(
            self.name,
            math_dtype,
            self.operand_extent,
            self.num_inputs,
            self.num_outputs,
            num_index,
            self.buffer_dim,
            self.buffer_num_segments,
            batch_dim,
            index_buffer,
            self.dtypes,
            self.num_operations,
            self.num_operands,
            self.operations,
            self.num_paths,
            self.path_indices_start,
            self.path_coefficients_start,
            self.path_indices,
            self.path_coefficients,
            batch_size,
            tensors,
        )
