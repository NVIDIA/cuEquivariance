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
except ImportError:
    tensor_product_uniform_1d_jit = None


class SegmentedPolynomialFromUniform1dJit(nn.Module):
    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
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
            torch._assert(
                o.all_same_segment_shape(), "all segments must have the same shape"
            )
            if o.ndim == 1 and len(o.segments) > 0:
                if operand_extent is None:
                    (operand_extent,) = o.segment_shape
                else:
                    torch._assert(
                        (operand_extent,) == o.segment_shape,
                        "all operands must have the same extent",
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
        empty_dict: Dict[int, torch.Tensor] = {}
        if input_indices is None:
            input_indices = dict(empty_dict)
        if output_shapes is None:
            output_shapes = dict(empty_dict)
        if output_indices is None:
            output_indices = dict(empty_dict)

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

        num_index = 0
        batch_dim = [self.BATCH_DIM_AUTO] * (self.num_inputs + self.num_outputs)
        index_buffer = [-1] * (self.num_inputs + self.num_outputs)
        tensors = list(inputs)

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


class SegmentedPolynomial(nn.Module):
    """
    PyTorch module that computes a segmented polynomial.

    Currently, it supports segmented polynomials where all segment sizes are the same,
    and each operand is one or zero dimensional.

    Args:
        polynomial: The segmented polynomial to compute, an instance of
            `cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`.
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
        math_dtype: torch.dtype = torch.float32,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
    ):
        super().__init__()
        self.m = SegmentedPolynomialFromUniform1dJit(
            polynomial, math_dtype, output_dtype_map, name
        )

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
        return self.m(inputs, input_indices, output_shapes, output_indices)
