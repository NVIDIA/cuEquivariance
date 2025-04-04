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
from typing import Any, Callable, Sequence

import numpy as np

import cuequivariance as cue
from cuequivariance.segmented_polynomials.operation import IVARS, OVARS

from .dimensions_dict import format_dimensions_dict


@dataclasses.dataclass(init=False, frozen=True)
class SegmentedPolynomial:
    """A polynomial representation using segmented tensor products.

    This class represents a polynomial using a collection of segmented tensor products, where each product
    is associated with an operation that specifies how inputs are combined. The polynomial maps a set of
    input tensors to output tensors through these tensor products.

    Args:
        inputs (tuple of SegmentedOperand): Input operands.
        outputs (tuple of SegmentedOperand): Output operands.
        operations (list of tuple of Operation and SegmentedTensorProduct): List of operation and tensor product pairs
            that define the polynomial transformation.
    """

    inputs: tuple[cue.SegmentedOperand, ...]
    outputs: tuple[cue.SegmentedOperand, ...]
    operations: tuple[tuple[cue.Operation, cue.SegmentedTensorProduct], ...]

    def __init__(
        self,
        inputs: Sequence[cue.SegmentedOperand],
        outputs: Sequence[cue.SegmentedOperand],
        operations: Sequence[
            tuple[cue.Operation | Sequence[int], cue.SegmentedTensorProduct]
        ],
    ):
        inputs = tuple(inputs)
        outputs = tuple(outputs)
        operands = inputs + outputs

        tmp = []
        for opt, stp in operations:
            opt = cue.Operation(opt)
            assert isinstance(opt, cue.Operation)
            assert isinstance(stp, cue.SegmentedTensorProduct)
            assert len(opt.buffers) == stp.num_operands
            for buffer_id, operand in zip(opt.buffers, stp.operands):
                assert operand == operands[buffer_id]

            out_oid, bid = opt.output_operand_buffer(len(inputs))
            tmp.append(
                (bid, opt.move_operand_last(out_oid), stp.move_operand_last(out_oid))
            )
        tmp = sorted(tmp)
        operations = [(opt, stp) for _, opt, stp in tmp]

        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "operations", tuple(operations))

    @classmethod
    def eval_last_operand(cls, stp: cue.SegmentedTensorProduct):
        """Create a polynomial that evaluates the last operand of a segmented tensor product.

        Args:
            stp (:class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>`): The tensor product to evaluate.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: A polynomial evaluating the last operand.
        """
        return cls(
            stp.operands[:-1],
            (stp.operands[-1],),
            ((cue.Operation(tuple(range(stp.num_operands))), stp),),
        )

    @classmethod
    def from_default_buffers(
        cls,
        inputs: Sequence[cue.SegmentedOperand | None],
        outputs: Sequence[cue.SegmentedOperand | None],
        operations: Sequence[
            tuple[cue.Operation | Sequence[int], cue.SegmentedTensorProduct]
        ],
    ):
        """Create a polynomial from default buffer assignments.

        Args:
            inputs (list of :class:`cue.SegmentedOperand <cuequivariance.SegmentedOperand>` | None): Default input operands, replaced by the STP operands if needed.
            outputs (list of :class:`cue.SegmentedOperand <cuequivariance.SegmentedOperand>` | None): Default output operands, replaced by the STP operands if needed.
            operations (list of tuple of :class:`cue.Operation <cuequivariance.Operation>` | list of int and :class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>`): Operations and tensor products.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: The constructed polynomial.
        """
        buffers = list(inputs) + list(outputs)
        for ope, stp in operations:
            ope = cue.Operation(ope)
            assert isinstance(stp, cue.SegmentedTensorProduct)
            assert len(ope.buffers) == stp.num_operands
            for buffer_id, operand in zip(ope.buffers, stp.operands):
                buffers[buffer_id] = operand

        return cls(buffers[: len(inputs)], buffers[len(inputs) :], operations)

    def __hash__(self) -> int:
        """Compute hash value for the polynomial.

        Returns:
            int: Hash value based on inputs, outputs and operations.
        """
        return hash((self.inputs, self.outputs, self.operations))

    def __eq__(self, value) -> bool:
        """Check equality with another polynomial.

        Args:
            value (:class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`): Polynomial to compare with.

        Returns:
            bool: True if polynomials are equal.
        """
        assert isinstance(value, SegmentedPolynomial)
        return (
            self.inputs == value.inputs
            and self.outputs == value.outputs
            and self.operations == value.operations
        )

    def __lt__(self, value) -> bool:
        """Compare polynomials for ordering.

        Args:
            value (:class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`): Polynomial to compare with.

        Returns:
            bool: True if this polynomial is less than the other.
        """
        assert isinstance(value, SegmentedPolynomial)
        return (
            self.inputs,
            self.outputs,
            self.operations,
        ) < (
            value.inputs,
            value.outputs,
            value.operations,
        )

    def __mul__(self, factor: float) -> SegmentedPolynomial:
        """Multiply polynomial by a scalar.

        Args:
            factor (float): Scalar factor.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Scaled polynomial.
        """
        return SegmentedPolynomial(
            self.inputs,
            self.outputs,
            tuple((ope, factor * stp) for ope, stp in self.operations),
        )

    def __rmul__(self, factor: float) -> SegmentedPolynomial:
        """Right multiply polynomial by a scalar.

        Args:
            factor (float): Scalar factor.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Scaled polynomial.
        """
        return self.__mul__(factor)

    def __repr__(self):
        def sfmt(shape: tuple[int, ...]) -> str:
            return "(" + ",".join(str(d) for d in shape) + ")"

        buffer_names = []
        for ope in self.operands:
            if ope.all_same_segment_shape():
                buffer_names.append(
                    f"[{ope.size}:{ope.num_segments}⨯{sfmt(ope.segment_shape)}]"
                )
            else:
                txts = []
                n = 20
                for s in ope.segments:
                    txts.append(sfmt(s))
                    if len("+".join(txts)) > n:
                        txts.pop()
                        break
                if len(txts) < len(ope.segments):
                    txts.append("...")
                buffer_names.append(f"[{ope.size}:{'+'.join(txts)}]")
        return self.to_string(buffer_names)

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
            + " ".join(buffer_txts[self.num_inputs :])
        )

        def f(ope: cue.Operation, stp: cue.SegmentedTensorProduct) -> str:
            items = [
                f"{buffer}[{ss}]"
                for buffer, ss in zip(
                    ope.to_letters(self.num_inputs), stp.subscripts.operands
                )
            ]
            out = items[-1]
            items = [f"[{stp.coefficient_subscripts}]"] + items[:-1]
            return "·".join(items) + "➜" + out

        lines = ["│  " + f(ope, stp) for ope, stp in self.operations]
        if len(lines) > 0:
            lines[-1] = "╰─" + lines[-1][2:]

        n = max(len(line) for line in lines)
        lines = [
            line
            + " "
            + "─" * (n - len(line))
            + "─ "
            + f"num_paths={stp.num_paths} {format_dimensions_dict(stp.get_dimensions_dict())}"
            for line, (_, stp) in zip(lines, self.operations)
        ]
        lines = ["╭ " + header] + lines

        lines = [line.rstrip() for line in lines]
        return "\n".join(lines)

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Evaluate the polynomial on the given inputs.

        Args:
            *inputs (np.ndarray): Input arrays to evaluate the polynomial on.

        Returns:
            list[np.ndarray]: List of output arrays.
        """
        inferred_shape = np.broadcast_shapes(*[x.shape[:-1] for x in inputs])
        inferred_dtype = np.result_type(*[x.dtype for x in inputs])
        outputs = [
            np.zeros(inferred_shape + (ope.size,), dtype=inferred_dtype)
            for ope in self.outputs
        ]
        for ope, stp in self.operations:
            oid, bid = ope.output_operand_buffer(self.num_inputs)
            outputs[bid - self.num_inputs] += (
                cue.segmented_polynomials.compute_last_operand(
                    stp.move_operand_last(oid),
                    *[inputs[bid] for bid in ope.input_buffers(self.num_inputs)],
                    dtype=inferred_dtype,
                )
            )
        return outputs

    @property
    def operands(self) -> tuple[cue.SegmentedOperand, ...]:
        """Get all operands (inputs and outputs) of the polynomial.

        Returns:
            tuple[:class:`cue.SegmentedOperand <cuequivariance.SegmentedOperand>`, ...]: Tuple of all operands.
        """
        return self.inputs + self.outputs

    @property
    def num_inputs(self) -> int:
        """Get the number of input operands.

        Returns:
            int: Number of input operands.
        """
        return len(self.inputs)

    @property
    def num_outputs(self) -> int:
        """Get the number of output operands.

        Returns:
            int: Number of output operands.
        """
        return len(self.outputs)

    @property
    def num_operands(self) -> int:
        """Get the total number of operands (inputs + outputs).

        Returns:
            int: Total number of operands.
        """
        return self.num_inputs + self.num_outputs

    def apply_fn(
        self,
        f: Callable[
            [cue.Operation, cue.SegmentedTensorProduct],
            tuple[cue.Operation, cue.SegmentedTensorProduct] | None,
        ],
    ) -> SegmentedPolynomial:
        """Apply a function to each tensor product in the polynomial.

        Args:
            f (Callable): Function to apply to each operation and tensor product pair.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: New polynomial with transformed tensor products.
        """
        new_tensor_products = [f(ope, stp) for ope, stp in self.operations]
        new_tensor_products = tuple(
            ope_stp for ope_stp in new_tensor_products if ope_stp is not None
        )
        return SegmentedPolynomial.from_default_buffers(
            self.inputs, self.outputs, new_tensor_products
        )

    def fuse_stps(self) -> SegmentedPolynomial:
        """Fuse segmented tensor products with identical operations and operands.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial with fused tensor products.
        """
        poly = self.apply_fn(lambda ope, stp: (ope, stp.canonicalize_subscripts()))

        groups = itertools.groupby(
            poly.operations,
            key=lambda x: (
                x[0],
                x[1].operands_and_subscripts,
                x[1].coefficient_subscripts,
            ),
        )
        new_tensor_products = tuple(
            (
                ope,
                cue.SegmentedTensorProduct(
                    operands_and_subscripts=operands_and_subscripts,
                    coefficient_subscripts=coefficient_subscripts,
                    paths=[path for _, stp in elements for path in stp.paths],
                ).consolidate_paths(),
            )
            for (
                ope,
                operands_and_subscripts,
                coefficient_subscripts,
            ), elements in groups
        )
        return SegmentedPolynomial(self.inputs, self.outputs, new_tensor_products)

    def consolidate(self) -> SegmentedPolynomial:
        """Consolidate the segmented tensor products by removing empty segments and squeezing modes.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Consolidated polynomial.
        """

        def f(ope: cue.Operation, stp: cue.SegmentedTensorProduct):
            stp = (
                stp.consolidate_modes()
                .squeeze_modes()
                .remove_empty_segments()
                .consolidate_paths()
            )
            if stp.num_paths == 0:
                return None
            return ope, stp

        return self.fuse_stps().apply_fn(f)

    def used_inputs(self) -> list[bool]:
        """Get list of boolean values indicating which inputs are used in the polynomial.

        Returns:
            list[bool]: List where True indicates the input is used.
        """
        return [
            any(buffer in ope.buffers for ope, _ in self.operations)
            for buffer in range(self.num_inputs)
        ]

    def used_outputs(self) -> list[bool]:
        """Get list of boolean values indicating which outputs are used in the polynomial.

        Returns:
            list[bool]: List where True indicates the output is used.
        """
        return [
            any(buffer in ope.buffers for ope, _ in self.operations)
            for buffer in range(self.num_inputs, self.num_inputs + self.num_outputs)
        ]

    def used_buffers(self) -> list[bool]:
        """Get list of boolean values indicating which buffers are used in the polynomial.

        Returns:
            list[bool]: List where True indicates the buffer is used.
        """
        return self.used_inputs() + self.used_outputs()

    def filter_keep_operands(self, keep: list[bool]) -> SegmentedPolynomial:
        """Select which buffers to keep in the polynomial.

        Use this method when you want to compute only a subset of the polynomial outputs
        and have control over which inputs to keep. For keeping all inputs (even if
        not used), use filter_keep_outputs. For automatically removing unused operands,
        use filter_drop_unsued_operands.

        Args:
            keep (list[bool]): List indicating which buffers to keep.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial with selected buffers.
        """
        assert len(keep) == self.num_operands

        # Create a mapping from old buffer indices to new buffer indices
        new_index = []
        i = 0
        for u in keep:
            if u:
                new_index.append(i)
                i += 1
            else:
                new_index.append(None)

        # Filter tensor products that write to buffers we want to keep
        # and remap the buffer indices
        new_tensor_products = []
        for ope, stp in self.operations:
            # Check if the operation writes to a buffer we want to keep
            bid = ope.output_buffer(self.num_inputs)
            if keep[bid]:
                # Check if all input buffers needed by this operation are kept
                if not all(keep[buffer] for buffer in ope.buffers):
                    raise ValueError(
                        f"Operation {ope} writes to buffer {bid} which is kept, but requires input buffers that are being dropped"
                    )

                # Remap buffer indices
                new_ope = cue.Operation([new_index[buffer] for buffer in ope.buffers])
                new_tensor_products.append((new_ope, stp))

        return SegmentedPolynomial(
            [x for x, k in zip(self.inputs, keep[: self.num_inputs]) if k],
            [x for x, k in zip(self.outputs, keep[self.num_inputs :]) if k],
            new_tensor_products,
        )

    def filter_keep_outputs(self, keep: list[bool]) -> SegmentedPolynomial:
        """Select which outputs to keep in the polynomial.

        Args:
            keep (list[bool]): List indicating which outputs to keep.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial with selected outputs.
        """
        assert len(keep) == self.num_outputs
        return self.filter_keep_operands([True] * self.num_inputs + keep)

    def filter_drop_unsued_operands(self) -> SegmentedPolynomial:
        """Remove all unused buffers from the polynomial.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial with unused buffers removed.
        """
        return self.filter_keep_operands(self.used_buffers())

    def compute_only(self, keep: list[bool]) -> SegmentedPolynomial:
        """Create a polynomial that only computes selected outputs.

        Args:
            keep (list[bool]): List indicating which outputs to compute.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial computing only selected outputs.
        """
        assert len(keep) == self.num_outputs
        return SegmentedPolynomial(
            self.inputs,
            self.outputs,  # on purpose, we keep all outputs
            [
                (ope, stp)
                for ope, stp in self.operations
                if keep[ope.output_buffer(self.num_inputs) - self.num_inputs]
            ],
        )

    @classmethod
    def stack_tensor_products(
        cls,
        inputs: Sequence[cue.SegmentedOperand | None],
        outputs: Sequence[cue.SegmentedOperand | None],
        operations: Sequence[
            tuple[cue.Operation | Sequence[int], cue.SegmentedTensorProduct]
        ],
    ) -> SegmentedPolynomial:
        """Stack segmented tensor products together.

        Args:
            inputs (list of :class:`cue.SegmentedOperand <cuequivariance.SegmentedOperand>` | None): Input operands.
            outputs (list of :class:`cue.SegmentedOperand <cuequivariance.SegmentedOperand>` | None): Output operands.
            operations (list of tuple of :class:`cue.Operation <cuequivariance.Operation>` | list of int and :class:`cue.SegmentedTensorProduct <cuequivariance.SegmentedTensorProduct>`): Operations and tensor products.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: The stacked polynomial.
        """
        inputs, outputs = list(inputs), list(outputs)
        return cls.stack(
            [
                cls.from_default_buffers(inputs, outputs, [(ope, stp)])
                for ope, stp in operations
            ],
            [ope is None for ope in inputs + outputs],
        )

    @classmethod
    def stack(
        cls, polys: list[SegmentedPolynomial], stacked: list[bool]
    ) -> SegmentedPolynomial:
        """Stack segmented polynomials together.

        This method combines multiple polynomials by stacking their operands where indicated
        by the stacked parameter. Non-stacked operands must be identical across all polynomials.

        Args:
            polys (list[:class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`]): List of polynomials to stack.
            stacked (list[bool]): List indicating which buffers should be stacked.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`:
                The stacked polynomial.

        Example:
            >>> p1 = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2]).polynomial
            >>> p2 = cue.descriptors.spherical_harmonics(cue.SO3(1), [2, 3]).polynomial
            >>> cue.SegmentedPolynomial.stack([p1, p2], [False, True])
            ╭ a=[3:3⨯()] -> B=[20:20⨯()]
            │  []·a[]➜B[] ───────── num_paths=3
            │  []·a[]·a[]➜B[] ───── num_paths=22
            ╰─ []·a[]·a[]·a[]➜B[] ─ num_paths=41

            Note how the STPs of degree 2 has been automatically fused into a single STP.
        """
        assert len(polys) > 0
        num_inputs = polys[0].num_inputs
        num_outputs = polys[0].num_outputs
        assert all(pol.num_inputs == num_inputs for pol in polys)
        assert all(pol.num_outputs == num_outputs for pol in polys)
        assert len(stacked) == num_inputs + num_outputs

        operands = []
        for bid in range(num_inputs + num_outputs):
            if stacked[bid]:
                operands.append(
                    cue.SegmentedOperand.stack(
                        [
                            pol.operands[bid]
                            for pol in polys
                            if pol.operands[bid]
                            is not None  # special case for stack_tensor_products
                        ]
                    )
                )
            else:
                ope = polys[0].operands[bid]
                assert all(pol.operands[bid] == ope for pol in polys)
                operands.append(ope)

        tensor_products: list[tuple[cue.Operation, cue.SegmentedTensorProduct]] = []
        for index, pol in enumerate(polys):
            for ope, stp in pol.operations:
                stp = copy.deepcopy(stp)
                for oid, buffer in enumerate(ope.buffers):
                    if stacked[buffer]:
                        for p in reversed(polys[:index]):
                            stp.insert_segments(oid, 0, p.operands[buffer].segments)
                        for p in polys[index + 1 :]:
                            stp.insert_segments(oid, -1, p.operands[buffer].segments)
                tensor_products.append((ope, stp))

        return cls(
            operands[:num_inputs], operands[num_inputs:], tensor_products
        ).consolidate()

    @classmethod
    def concatenate(
        cls,
        inputs: Sequence[cue.SegmentedOperand],
        outputs: Sequence[cue.SegmentedOperand],
        polys: list[tuple[SegmentedPolynomial, Sequence[int | None]]],
    ) -> SegmentedPolynomial:
        """Concatenate segmented polynomials.

        Args:
            inputs (list of :class:`cue.SegmentedOperand <cuequivariance.SegmentedOperand>`): Input operands for the concatenated polynomial.
            outputs (list of :class:`cue.SegmentedOperand <cuequivariance.SegmentedOperand>`): Output operands for the concatenated polynomial.
            polys (list of tuple of :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` and list of int | None): List of tuples containing
                (polynomial, buffer_mapping), where buffer_mapping[i] is the buffer index in the polynomial
                that corresponds to the i-th buffer in the concatenated polynomial. If buffer_mapping[i] is None,
                the i-th buffer in the concatenated polynomial is not used in the polynomial.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`:
                The concatenated polynomial.

        Example:
            >>> p1 = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2]).polynomial
            >>> p2 = cue.descriptors.spherical_harmonics(cue.SO3(1), [2, 3]).polynomial
            >>> [vec, sh1] = p1.operands
            >>> [_, sh2] = p2.operands
            >>> cue.SegmentedPolynomial.concatenate(
            ...     [vec],
            ...     [sh1, sh2],
            ...     [(p1, [0, 1, None]), (p2, [0, None, 1])],
            ... )
            ╭ a=[3:3⨯()] -> B=[8:8⨯()] C=[12:12⨯()]
            │  []·a[]➜B[] ───────── num_paths=3
            │  []·a[]·a[]➜B[] ───── num_paths=11
            │  []·a[]·a[]➜C[] ───── num_paths=11
            ╰─ []·a[]·a[]·a[]➜C[] ─ num_paths=41
        """
        return cls(
            inputs,
            outputs,
            [
                ([mp.index(bid) for bid in ope.buffers], stp)
                for pol, mp in polys
                for ope, stp in pol.operations
            ],
        )

    def flatten_modes(self, modes: list[str]) -> SegmentedPolynomial:
        """Flatten specified modes in the polynomial.

        Args:
            modes (list[str]): List of mode names to flatten.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial with flattened modes.
        """
        return SegmentedPolynomial.from_default_buffers(
            self.inputs,
            self.outputs,
            [(ope, stp.flatten_modes(modes)) for ope, stp in self.operations],
        )

    def all_same_segment_shape(self) -> bool:
        """Check if all operands have the same segment shape.

        Returns:
            bool: True if all operands have the same segment shape.
        """
        return all(ope.all_same_segment_shape() for ope in self.operands)

    def canonicalize_subscripts(self) -> SegmentedPolynomial:
        """Canonicalize the subscripts of the segmented tensor products.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial with canonicalized subscripts.
        """
        return SegmentedPolynomial.from_default_buffers(
            self.inputs,
            self.outputs,
            [(ope, stp.canonicalize_subscripts()) for ope, stp in self.operations],
        )

    def squeeze_modes(self, modes: str | None = None) -> SegmentedPolynomial:
        """Squeeze specified modes in the polynomial.

        Args:
            modes (str | None, optional): Modes to squeeze. If None, squeezes all modes.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial with squeezed modes.
        """
        return SegmentedPolynomial.from_default_buffers(
            self.inputs,
            self.outputs,
            [(ope, stp.squeeze_modes(modes)) for ope, stp in self.operations],
        )

    def flatten_coefficient_modes(self) -> SegmentedPolynomial:
        """Flatten the coefficient modes of the segmented tensor products.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`: Polynomial with flattened coefficient modes.
        """
        return SegmentedPolynomial.from_default_buffers(
            self.inputs,
            self.outputs,
            [(ope, stp.flatten_coefficient_modes()) for ope, stp in self.operations],
        )

    def jvp(
        self, has_tangent: list[bool]
    ) -> tuple[
        SegmentedPolynomial,
        Callable[[tuple[list[Any], list[Any]]], tuple[list[Any], list[Any]]],
    ]:
        """Compute the Jacobian-vector product of the polynomial.

        Args:
            has_tangent (list[bool]): List indicating which inputs have tangents.

        Returns:
            tuple[:class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`, Callable]:
                The JVP polynomial and a mapping function for inputs/outputs.
        """
        assert len(has_tangent) == self.num_inputs

        # Symmetrizing the polynomial helps identify simplifications by group_by_operational_symmetries
        sym_poly = self.symmetrize_for_identical_operands()

        new_operations = []
        for ope, stp in sym_poly.operations:
            jvps = ope.jvp(has_tangent)
            permutations: list[tuple[int, ...]] = stp.symmetries()
            for multiplicator, ope in cue.Operation.group_by_operational_symmetries(
                permutations, jvps
            ):
                new_operations.append((ope, multiplicator * stp))

        def mapping(x: tuple[list[Any], list[Any]]) -> tuple[list[Any], list[Any]]:
            inputs, outputs = x
            inputs, outputs = list(inputs), list(outputs)
            assert len(inputs) == self.num_inputs
            assert len(outputs) == self.num_outputs

            new_inputs = inputs + [x for has, x in zip(has_tangent, inputs) if has]
            new_outputs = outputs

            return new_inputs, new_outputs

        jvp_poly = SegmentedPolynomial(
            *mapping((self.inputs, self.outputs)), new_operations
        )
        return jvp_poly, mapping

    def transpose(
        self,
        is_undefined_primal: list[bool],
        has_cotangent: list[bool],
    ) -> tuple[
        SegmentedPolynomial,
        Callable[[tuple[list[Any], list[Any]]], tuple[list[Any], list[Any]]],
    ]:
        """Transpose the polynomial for reverse-mode automatic differentiation.

        Args:
            is_undefined_primal (list[bool]): List indicating which inputs have undefined primals.
            has_cotangent (list[bool]): List indicating which outputs have cotangents.

        Returns:
            tuple[:class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`, Callable]:
                The transposed polynomial and a mapping function for inputs/outputs.
        """
        assert len(is_undefined_primal) == self.num_inputs
        assert len(has_cotangent) == self.num_outputs

        new_operations = []
        for ope, stp in self.operations:
            ope = ope.transpose(is_undefined_primal, has_cotangent)
            if ope is not None:
                new_operations.append((ope, stp))

        def mapping(x: tuple[list[Any], list[Any]]) -> tuple[list[Any], list[Any]]:
            inputs, outputs = x
            inputs, outputs = list(inputs), list(outputs)
            assert len(inputs) == self.num_inputs
            assert len(outputs) == self.num_outputs

            new_inputs = [
                x for undef, x in zip(is_undefined_primal, inputs) if not undef
            ] + [x for has, x in zip(has_cotangent, outputs) if has]
            new_outputs = [x for undef, x in zip(is_undefined_primal, inputs) if undef]

            return new_inputs, new_outputs

        tr_poly = SegmentedPolynomial(
            *mapping((self.inputs, self.outputs)), new_operations
        )
        return tr_poly, mapping

    def backward(
        self, requires_gradient: list[bool], has_cotangent: list[bool]
    ) -> tuple[
        SegmentedPolynomial,
        Callable[[tuple[list[Any], list[Any]]], tuple[list[Any], list[Any]]],
    ]:
        """Compute the backward pass of the polynomial for gradient computation.

        Args:
            requires_gradient (list[bool]): List indicating which inputs require gradients.
            has_cotangent (list[bool]): List indicating which outputs have cotangents.

        Returns:
            tuple[:class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`, Callable]:
                The backward polynomial and a mapping function for inputs/outputs.
        """
        p, map1 = self.jvp(requires_gradient)
        p, map2 = p.transpose(
            [False] * self.num_inputs + [True] * sum(requires_gradient),
            has_cotangent,
        )

        def mapping(x: tuple[list[Any], list[Any]]) -> tuple[list[Any], list[Any]]:
            return map2(map1(x))

        return p, mapping

    def flop(self, batch_size: int = 1) -> int:
        """Compute the number of floating point operations in the polynomial.

        Args:
            batch_size (int, optional): Batch size for computation. Defaults to 1.

        Returns:
            int: Number of floating point operations.
        """
        n = 0
        for ope, stp in self.operations:
            oid, _ = ope.output_operand_buffer(self.num_inputs)
            n += stp.flop(oid)
        return batch_size * n

    def memory(self, batch_sizes: list[int]) -> int:
        """Compute the memory usage of the polynomial.

        Args:
            batch_sizes (list[int]): List of batch sizes for each operand. Each operand
                can have its own batch size, allowing for different batch dimensions
                per tensor.

        Returns:
            int: Memory usage in number of elements.
        """
        assert len(batch_sizes) == self.num_operands
        return sum(Z * ope.size for Z, ope in zip(batch_sizes, self.operands))

    def symmetrize_for_identical_operands(self) -> SegmentedPolynomial:
        """Symmetrize the paths of the segmented tensor products for identical operands.

        This operation increases the number of paths in the segmented tensor products.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`:
                Polynomial with symmetrized paths.
        """
        symmetrized_tensor_products = []
        for ope, stp in self.operations:
            for set_of_operands in ope.operands_with_identical_buffers():
                stp = stp.symmetrize_operands(set_of_operands)
            stp = stp.sort_paths()
            symmetrized_tensor_products.append((ope, stp))

        return SegmentedPolynomial(
            self.inputs, self.outputs, symmetrized_tensor_products
        )

    def unsymmetrize_for_identical_operands(self) -> SegmentedPolynomial:
        """Unsymmetrize the paths of the segmented tensor products for identical operands.

        This operation decreases the number of paths in the segmented tensor products.

        Returns:
            :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`:
                Polynomial with unsymmetrized paths.
        """

        def optimize_paths(ope: cue.Operation, stp: cue.SegmentedTensorProduct):
            for set_of_operands in ope.operands_with_identical_buffers():
                stp = stp.sort_indices_for_identical_operands(set_of_operands)
            stp = stp.sort_paths()
            return ope, stp

        return self.apply_fn(optimize_paths)
