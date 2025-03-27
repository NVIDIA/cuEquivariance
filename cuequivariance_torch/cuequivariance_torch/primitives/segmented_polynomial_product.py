import torch
import torch.nn as nn
from typing import List, Optional
from itertools import accumulate

try:
    # keep us an option to be independent of the torch.library machinery
    from cuequivariance_ops_torch.tensor_product_uniform_1d_jit import (
        tensor_product_uniform_1d_jit,
        BATCH_DIM_SHARED,
        BATCH_DIM_BATCHED,
        BATCH_DIM_INDEXED,
        BATCH_DIM_AUTO,
    )
except Exception:
    from cuequivariance_ops_torch.tensor_product_uniform_1d_jit import (
        BATCH_DIM_SHARED,
        BATCH_DIM_BATCHED,
        BATCH_DIM_INDEXED,
        BATCH_DIM_AUTO,
    )

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
        return torch.ops.cuequivariance_ops.tensor_product_uniform_1d_jit(
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


class SegmentedPolynomialProductJit(nn.Module):
    def __init__(
        self,
        polynomial,
        math_dtype: torch.dtype = torch.float32,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
    ):
        super().__init__()

        operand_extent = None
        for o in polynomial.operands:
            torch._assert(
                o.ndim in [0, 1], "only 0 or 1 dimensional operands are supported"
            )
            torch._assert(
                all(len(s) == o.ndim for s in o.segments),
                "all segments must have the same number of dimensions as the operand",
            )
            if o.ndim == 1 and len(o.segments) > 0:
                if operand_extent is None:
                    operand_extent = o.segments[0][0]
                else:
                    torch._assert(
                        operand_extent == o.segments[0][0],
                        "all operands must have the same extent",
                    )
                torch._assert(
                    all(operand_extent == s[0] for s in o.segments),
                    "the extent of the operand must all be indentical",
                )
        if operand_extent is None:
            operand_extent = 1

        for o, stp in polynomial.operations:
            torch._assert(
                stp.num_operands == len(o.buffers),
                "the number of operands must match the number of buffers",
            )
            torch._assert(
                stp.coefficient_subscripts == "", "the coefficients must be scalar"
            )

        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.name = name
        self.math_dtype = math_dtype
        self.operand_extent = operand_extent
        self.buffer_dim = [o.ndim for o in polynomial.operands]
        torch._assert(
            all(buffer_dim in [0, 1] for buffer_dim in self.buffer_dim),
            "buffer dimensions must be 0 or 1",
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
        self.operations = [
            b for o, stp in polynomial.operations for b in o.buffers
        ]
        self.num_paths = [stp.num_paths for o, stp in polynomial.operations]
        self.path_indices_start = [0] + list(
            accumulate(
                [
                    stp.num_paths * stp.num_operands
                    for o, stp in polynomial.operations
                ]
            )
        )[:-1]
        self.path_coefficients_start = [0] + list(
            accumulate([stp.num_paths for o, stp in polynomial.operations])
        )[:-1]
        self.path_indices = [
            i
            for o, stp in polynomial.operations
            for p in stp.paths
            for i in p.indices
        ]
        self.path_coefficients = [
            float(p.coefficients)
            for o, stp in polynomial.operations
            for p in stp.paths
        ]

        self.BATCH_DIM_AUTO = BATCH_DIM_AUTO
        self.BATCH_DIM_SHARED = BATCH_DIM_SHARED
        self.BATCH_DIM_BATCHED = BATCH_DIM_BATCHED
        self.BATCH_DIM_INDEXED = BATCH_DIM_INDEXED

    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[List[Optional[torch.Tensor]]] = None,
        output_shapes: Optional[List[Optional[int]]] = None,
        output_indices: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        torch._assert(
            len(inputs) == self.num_inputs,
            "the number of inputs must match the number of inputs of the polynomial",
        )
        torch._assert(
            len(input_indices) <= self.num_inputs,
            "the number of input indices must be less than or equal to the number of inputs of the polynomial",
        )
        torch._assert(
            len(output_indices) <= self.num_outputs,
            "the number of output indices must be less than or equal to the number of outputs of the polynomial",
        )
        torch._assert(
            len(output_shapes) <= self.num_outputs,
            "the number of output shapes must be less than or equal to the number of outputs of the polynomial",
        )
        if input_indices is None:
            input_indices = []
        if output_shapes is None:
            output_shapes = []
        if output_indices is None:
            output_indices = []

        num_index = 0
        batch_dim = [self.BATCH_DIM_AUTO] * (self.num_inputs + self.num_outputs)
        index_buffer = [-1] * (self.num_inputs + self.num_outputs)
        tensors = list(inputs)

        for idx_pos, idx_tensor in enumerate(input_indices):
            if idx_tensor is not None and idx_tensor.numel() > 0:
                batch_dim[idx_pos] = self.BATCH_DIM_INDEXED
                tensors.append(idx_tensor)
                index_buffer[idx_pos] = num_index
                num_index += 1
                index_buffer.append(inputs[idx_pos].shape[0])

        for idx_pos, idx_tensor in enumerate(output_indices):
            if idx_tensor is not None and idx_tensor.numel() > 0:
                batch_dim[idx_pos + self.num_inputs] = (
                    self.BATCH_DIM_INDEXED
                )
                tensors.append(idx_tensor)
                index_buffer[idx_pos + self.num_inputs] = num_index
                num_index += 1
                output_shape = output_shapes[idx_pos]
                torch._assert(output_shape is not None, "output shapes must be provided for output indices")
                if output_shape is not None:
                    index_buffer.append(output_shape)

        batch_size = self.BATCH_DIM_AUTO
        for idx_pos, idx_shape in enumerate(output_shapes):
            if batch_dim[idx_pos + self.num_inputs] == self.BATCH_DIM_AUTO:
                if idx_shape is not None:
                    if idx_shape == 1:
                        batch_dim[idx_pos + self.num_inputs] = (
                            self.BATCH_DIM_SHARED
                        )
                    else:
                        torch._assert(batch_size == self.BATCH_DIM_AUTO or batch_size == idx_shape, "batch size must be auto or the output shape")
                        batch_dim[idx_pos + self.num_inputs] = (
                            self.BATCH_DIM_BATCHED
                        )
                        batch_size = idx_shape

        return tensor_product_uniform_1d_jit(
            self.name,
            self.math_dtype,
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


class SegmentedPolynomialProduct(nn.Module):
    def __init__(
        self,
        polynomial,
        math_dtype: torch.dtype = torch.float32,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
    ):
        super().__init__()
        # try:
        self.m = SegmentedPolynomialProductJit(
            polynomial, math_dtype, output_dtype_map, name
        )
        # except Exception:
        #     self.m = SegmentedPolynomialTensorProductFallback(
        #         polynomial, math_dtype, name
        #     )

    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[List[Optional[torch.Tensor]]] = None,
        output_shapes: Optional[List[Optional[int]]] = None,
        output_indices: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        return self.m(inputs, input_indices, output_shapes, output_indices)
