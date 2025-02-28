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
from __future__ import annotations

import copy
import dataclasses
import itertools
from typing import Callable, Sequence

import numpy as np

import cuequivariance as cue
from cuequivariance.operation import IVARS, OVARS


@dataclasses.dataclass(init=False, frozen=True)
class SegmentedPolynomial:
    """A polynomial representation using segmented tensor products.

    This class represents a polynomial using a collection of segmented tensor products, where each product
    is associated with an operation that specifies how inputs are combined. The polynomial maps a set of
    input tensors to output tensors through these tensor products.

    Args:
        num_inputs (int): Number of input tensors.
        num_outputs (int): Number of output tensors.
        tensor_products (list of tuple of Operation and SegmentedTensorProduct): List of operation and tensor product pairs
            that define the polynomial transformation.

    Example:
        >>> # Create a polynomial with 2 inputs and 1 output
        >>> poly = SegmentedPolynomial(2, 1, [(op1, stp1), (op2, stp2)])
        >>> outputs = poly(input1, input2)  # Evaluate polynomial on inputs (numpy reference implementation)
    """

    num_inputs: int
    num_outputs: int
    tensor_products: list[tuple[cue.Operation, cue.SegmentedTensorProduct]]

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        tensor_products: Sequence[tuple[cue.Operation, cue.SegmentedTensorProduct]],
    ):
        object.__setattr__(self, "num_inputs", num_inputs)
        object.__setattr__(self, "num_outputs", num_outputs)
        object.__setattr__(self, "tensor_products", sorted(tensor_products))

    @classmethod
    def eval_last_operand(cls, stp: cue.SegmentedTensorProduct):
        return cls(
            stp.num_operands - 1,
            1,
            [(cue.Operation(tuple(range(stp.num_operands))), stp)],
        )

    def __hash__(self) -> int:
        return hash((self.num_inputs, self.num_outputs, tuple(self.tensor_products)))

    def __eq__(self, value) -> bool:
        assert isinstance(value, SegmentedPolynomial)
        return (
            self.num_inputs == value.num_inputs
            and self.num_outputs == value.num_outputs
            and self.tensor_products == value.tensor_products
        )

    def __lt__(self, value) -> bool:
        assert isinstance(value, SegmentedPolynomial)
        return (
            self.num_inputs,
            self.num_outputs,
            self.tensor_products,
        ) < (
            value.num_inputs,
            value.num_outputs,
            value.tensor_products,
        )

    def __mul__(self, factor: float) -> SegmentedPolynomial:
        return SegmentedPolynomial(
            self.num_inputs,
            self.num_outputs,
            [(ope, factor * stp) for ope, stp in self.tensor_products],
        )

    def __rmul__(self, factor: float) -> SegmentedPolynomial:
        return self.__mul__(factor)

    def __repr__(self):
        return self.to_string()

    def to_string(self, buffer_names: list[str] | None = None) -> str:
        buffer_txts = (
            IVARS[: self.num_inputs]
            + OVARS[self.num_inputs : self.num_inputs + self.num_outputs]
        )
        if buffer_names is not None:
            buffer_txts = [
                f"{symbol}={name}" for symbol, name in zip(buffer_txts, buffer_names)
            ]

        header = (
            " ".join(buffer_txts[: self.num_inputs])
            + " -> "
            + " ".join(
                buffer_txts[self.num_inputs : self.num_inputs + self.num_outputs]
            )
        )
        lines = [
            "│  " + ope.to_string(self.num_inputs) for ope, _ in self.tensor_products
        ]
        if len(lines) > 0:
            lines[-1] = "╰─" + lines[-1][2:]
        n = max(len(line) for line in lines)

        lines = [
            line + " " + "─" * (n - len(line)) + "─ " + str(stp)
            for line, (_, stp) in zip(lines, self.tensor_products)
        ]

        modes = sorted(
            {mode for _, stp in self.tensor_products for mode in stp.subscripts.modes()}
        )
        if len(modes) > 1:
            modes = []
        for a in ["sizes=", "num_segments=", "num_paths="] + [f"{m}=" for m in modes]:
            if not all(line.count(a) == 1 for line in lines):
                continue

            splits = [line.split(a) for line in lines]
            n = max(len(before) for before, _ in splits)
            lines = [
                before + " " * (n - len(before)) + a + after for before, after in splits
            ]

        lines = ["╭ " + header] + lines

        return "\n".join(lines)

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        inferred_shape = np.broadcast_shapes(*[x.shape[:-1] for x in inputs])
        inferred_dtype = np.result_type(*[x.dtype for x in inputs])
        outputs = [
            np.zeros(inferred_shape + (size,), dtype=inferred_dtype)
            for size in self.output_sizes
        ]
        for ope, stp in self.tensor_products:
            oid, bid = ope.output_operand_buffer(self.num_inputs)
            outputs[bid - self.num_inputs] += (
                cue.segmented_tensor_product.compute_last_operand(
                    stp.move_operand_last(oid),
                    *[inputs[bid] for bid in ope.input_buffers(self.num_inputs)],
                    dtype=inferred_dtype,
                )
            )
        return outputs

    @property
    def num_operands(self) -> int:
        """Number of operands in the polynomial."""
        return self.num_inputs + self.num_outputs

    @property
    def buffer_sizes(self) -> list[int | None]:
        """Sizes of the buffers in the polynomial."""
        sizes = [None] * (self.num_inputs + self.num_outputs)
        for ope, stp in self.tensor_products:
            for buffer, operand in zip(ope.buffers, stp.operands):
                if sizes[buffer] is None:
                    sizes[buffer] = operand.size
                if sizes[buffer] != operand.size:
                    raise ValueError(
                        f"Buffer {buffer} has inconsistent sizes: {sizes[buffer]} vs {operand.size}"
                    )
        return sizes

    @property
    def input_sizes(self) -> list[int | None]:
        """Sizes of the input buffers in the polynomial."""
        return self.buffer_sizes[: self.num_inputs]

    @property
    def output_sizes(self) -> list[int | None]:
        """Sizes of the output buffers in the polynomial."""
        return self.buffer_sizes[self.num_inputs :]

    def map_tensor_products(
        self,
        f: Callable[
            [cue.Operation, cue.SegmentedTensorProduct],
            tuple[cue.Operation, cue.SegmentedTensorProduct] | None,
        ],
    ) -> SegmentedPolynomial:
        new_tensor_products = [f(ope, stp) for ope, stp in self.tensor_products]
        new_tensor_products = [
            ope_stp for ope_stp in new_tensor_products if ope_stp is not None
        ]
        return SegmentedPolynomial(
            self.num_inputs, self.num_outputs, new_tensor_products
        )

    def fuse_stps(self) -> SegmentedPolynomial:
        """Fuse segmented tensor products with identical operations and operands."""
        groups = itertools.groupby(
            self.tensor_products,
            key=lambda x: (x[0], x[1].operands, x[1].coefficient_subscripts),
        )
        new_tensor_products = [
            (
                ope,
                cue.SegmentedTensorProduct(
                    operands=operands,
                    coefficient_subscripts=coefficient_subscripts,
                    paths=[path for _, stp in elements for path in stp.paths],
                ).consolidate_paths(),
            )
            for (ope, operands, coefficient_subscripts), elements in groups
        ]
        return SegmentedPolynomial(
            self.num_inputs, self.num_outputs, new_tensor_products
        )

    def consolidate(self) -> SegmentedPolynomial:
        """Consolidate the segmented tensor products."""

        def f(ope: cue.Operation, stp: cue.SegmentedTensorProduct):
            stp = (
                stp.consolidate_modes()
                .squeeze_modes()
                .remove_empty_segments()
                .consolidate_paths()
                .sort_paths()
            )
            if stp.num_paths == 0:
                return None
            return ope, stp

        return self.fuse_stps().map_tensor_products(f)

    def used_buffers(self) -> list[int]:
        """Buffers used in the polynomial. (List of integers)"""
        return sorted(
            set(
                itertools.chain.from_iterable(
                    ope.buffers for ope, _ in self.tensor_products
                )
            )
        )

    def buffer_used(self) -> list[bool]:
        """Buffers used in the polynomial. (List of boolean values)"""
        return [
            any(buffer in ope.buffers for ope, _ in self.tensor_products)
            for buffer in range(self.num_inputs + self.num_outputs)
        ]

    def remove_unused_buffers(self) -> SegmentedPolynomial:
        """Remove unused buffers from the polynomial."""
        used = self.buffer_used()
        new_index = []
        i = 0
        for u in used:
            if u:
                new_index.append(i)
                i += 1
            else:
                new_index.append(None)

        return SegmentedPolynomial(
            sum(used[: self.num_inputs]),
            sum(used[self.num_inputs :]),
            [
                (cue.Operation([new_index[buffer] for buffer in ope.buffers]), stp)
                for ope, stp in self.tensor_products
            ],
        )

    @classmethod
    def stack(
        cls, polys: list[SegmentedPolynomial], stacked: list[bool]
    ) -> SegmentedPolynomial:
        """Stack segmented polynomials together."""
        assert len(polys) > 0
        num_inputs = polys[0].num_inputs
        num_outputs = polys[0].num_outputs
        assert all(pol.num_inputs == num_inputs for pol in polys)
        assert all(pol.num_outputs == num_outputs for pol in polys)
        assert len(stacked) == num_inputs + num_outputs

        tensor_products: list[tuple[cue.Operation, cue.SegmentedTensorProduct]] = []
        for index, pol in enumerate(polys):
            for ope, stp in pol.tensor_products:
                stp = copy.deepcopy(stp)
                for oid, buffer in enumerate(ope.buffers):
                    if stacked[buffer]:
                        for p in reversed(polys[:index]):
                            stp.insert_segments(oid, 0, p.buffer_segments(buffer))
                        for p in polys[index + 1 :]:
                            stp.insert_segments(oid, -1, p.buffer_segments(buffer))
                tensor_products.append((ope, stp))
        return cls(num_inputs, num_outputs, tensor_products)

    def squeeze_modes(self) -> SegmentedPolynomial:
        """Squeeze the modes of the segmented tensor products."""
        return SegmentedPolynomial(
            self.num_inputs,
            self.num_outputs,
            [(ope, stp.squeeze_modes()) for ope, stp in self.tensor_products],
        )

    def flatten_coefficient_modes(self) -> SegmentedPolynomial:
        """Flatten the coefficient modes of the segmented tensor products."""
        return SegmentedPolynomial(
            self.num_inputs,
            self.num_outputs,
            [
                (ope, stp.flatten_coefficient_modes())
                for ope, stp in self.tensor_products
            ],
        )

    def jvp(self, has_tangent: list[bool]) -> SegmentedPolynomial:
        """Compute the Jacobian-vector product of the polynomial."""
        assert len(has_tangent) == self.num_inputs

        new_tps = []
        for ope, stp in self.tensor_products:
            jvps = ope.jvp(has_tangent)
            permutations: list[tuple[int, ...]] = stp.symmetries()
            for multiplicator, ope in cue.Operation.group_by_operational_symmetries(
                permutations, jvps
            ):
                new_tps.append((ope, multiplicator * stp))
        return SegmentedPolynomial(
            self.num_inputs + sum(has_tangent), self.num_outputs, new_tps
        )

    def transpose(
        self,
        is_undefined_primal: list[bool],
        has_cotangent: list[bool],
    ) -> SegmentedPolynomial:
        """Transpose the polynomial."""
        assert len(is_undefined_primal) == self.num_inputs
        assert len(has_cotangent) == self.num_outputs

        new_tps = []
        for ope, stp in self.tensor_products:
            ope = ope.transpose(is_undefined_primal, has_cotangent)
            if ope is not None:
                new_tps.append((ope, stp))
        return SegmentedPolynomial(
            sum(map(lambda u: not u, is_undefined_primal)) + sum(has_cotangent),
            sum(is_undefined_primal),
            new_tps,
        )

    def backward(
        self, requires_gradient: list[bool], has_cotangent: list[bool]
    ) -> SegmentedPolynomial:
        """Compute the backward pass of the polynomial."""
        return self.jvp(requires_gradient).transpose(
            is_undefined_primal=[False] * self.num_inputs
            + [True] * sum(requires_gradient),
            has_cotangent=has_cotangent,
        )

    def flops(self, batch_size: int = 1) -> int:
        """Compute the number of floating point operations in the polynomial."""
        n = 0
        for ope, stp in self.tensor_products:
            oid, _ = ope.output_operand_buffer(self.num_inputs)
            n += stp.flop_cost(oid)
        return batch_size * n

    def memory(self, batch_sizes: list[int]) -> int:
        """Compute the memory usage of the polynomial."""
        assert len(batch_sizes) == self.num_operands
        return sum(Z * size for Z, size in zip(batch_sizes, self.buffer_sizes))

    def buffer_segments(self, buffer: int) -> list[tuple[int, ...]]:
        segments = None
        for ope, stp in self.tensor_products:
            if buffer in ope.buffers:
                ope = stp.operands[ope.buffers.index(buffer)]
                if segments is None:
                    segments = ope.segments
                elif segments != ope.segments:
                    raise ValueError(
                        f"Buffer {buffer} has inconsistent segments: {segments} vs {ope.segments}"
                    )
        if segments is None:
            raise ValueError(f"Buffer {buffer} is not used")
        return segments

    def sort_indices_for_identical_operands(self) -> SegmentedPolynomial:
        """Sort the indices of the segmented tensor products for identical operands."""

        def optimize_paths(ope: cue.Operation, stp: cue.SegmentedTensorProduct):
            for set_of_operands in ope.operands_with_identical_buffers():
                stp = stp.sort_indices_for_identical_operands(set_of_operands)
            stp = stp.sort_paths()
            return ope, stp

        return self.map_tensor_products(optimize_paths)
