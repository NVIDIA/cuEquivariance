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

import dataclasses
from typing import Any, Callable

import numpy as np

import cuequivariance as cue
from cuequivariance.segmented_polynomials.segmented_tensor_product import (
    _canonicalize_index,
)


@dataclasses.dataclass(init=False, frozen=True)
class EquivariantPolynomial:
    """A polynomial representation with equivariance constraints.

    This class extends :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` by incorporating information about the group representations
    of each input and output tensor. It ensures that operations performed by the polynomial respect
    the equivariance constraints defined by these representations, making it suitable for building
    equivariant neural networks.

    Args:
        inputs (tuple of :class:`cue.Rep <cuequivariance.Rep>`): Group representations for input tensors.
        outputs (tuple of :class:`cue.Rep <cuequivariance.Rep>`): Group representations for output tensors.
        polynomial (:class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`): The underlying polynomial transformation.
    """

    inputs: tuple[cue.Rep, ...]
    outputs: tuple[cue.Rep, ...]
    polynomial: cue.SegmentedPolynomial

    # ------------------------------------------------------------------------
    # Core Structure and Initialization
    # ------------------------------------------------------------------------

    def __init__(
        self,
        inputs: list[cue.Rep],
        outputs: list[cue.Rep],
        polynomial: cue.SegmentedPolynomial,
    ):
        assert isinstance(polynomial, cue.SegmentedPolynomial)
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "outputs", tuple(outputs))
        object.__setattr__(self, "polynomial", polynomial)
        if len(self.inputs) != self.polynomial.num_inputs:
            raise ValueError(
                f"Number of inputs {len(self.inputs)} must equal the number of inputs"
                f" in the polynomial {self.polynomial.num_inputs}"
            )
        if len(self.outputs) != self.polynomial.num_outputs:
            raise ValueError(
                f"Number of outputs {len(self.outputs)} must equal the number of outputs"
                f" in the polynomial {self.polynomial.num_outputs}"
            )
        for rep, ope in zip(self.operands, self.polynomial.operands):
            assert ope.size == rep.dim, (
                f"{ope} incompatible with {rep}. {ope.size=} != {rep.dim=}"
            )

    @property
    def operands(self) -> tuple[cue.Rep, ...]:
        """Get all operands (inputs and outputs) of the polynomial.

        Returns:
            tuple of :class:`cue.Rep <cuequivariance.Rep>`: Tuple of all operands.
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

    # ------------------------------------------------------------------------
    # Class Construction Methods
    # ------------------------------------------------------------------------

    # Method eval_last_operand is present in SegmentedPolynomial but not in EquivariantPolynomial

    @classmethod
    def stack(
        cls, polys: list[EquivariantPolynomial], stacked: list[bool]
    ) -> EquivariantPolynomial:
        """Stack equivariant polynomials together.

        This method combines multiple polynomials by stacking their operands where indicated
        by the stacked parameter. Non-stacked operands must be identical across all polynomials.

        Args:
            polys (list of :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`): List of polynomials to stack.
            stacked (list of bool): List indicating which operands should be stacked.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: The stacked polynomial.

        Example:
            >>> p1 = cue.descriptors.spherical_harmonics(cue.SO3(1), [1, 2])
            >>> p2 = cue.descriptors.spherical_harmonics(cue.SO3(1), [2, 3])
            >>> cue.EquivariantPolynomial.stack([p1, p2], [False, True])
            ╭ a=1 -> B=1+2+2+3
            │  []·a[]➜B[] ───────── num_paths=3
            │  []·a[]·a[]➜B[] ───── num_paths=22
            ╰─ []·a[]·a[]·a[]➜B[] ─ num_paths=41

            Note how the STPs of degree 2 has been automatically fused into a single STP.
        """
        assert len(polys) > 0
        num_operands = polys[0].num_operands

        assert all(pol.num_operands == num_operands for pol in polys)
        assert len(stacked) == num_operands

        operands = []
        for oid in range(num_operands):
            if stacked[oid]:
                for pol in polys:
                    if not isinstance(pol.operands[oid], cue.IrrepsAndLayout):
                        raise ValueError(
                            f"Cannot stack operand {oid} of type {type(pol.operands[oid])}"
                        )
                operands.append(cue.concatenate([pol.operands[oid] for pol in polys]))
            else:
                ope = polys[0].operands[oid]
                for pol in polys:
                    if pol.operands[oid] != ope:
                        raise ValueError(
                            f"Operand {oid} must be the same for all polynomials."
                            f" Found {ope} and {pol.operands[oid]}"
                        )
                operands.append(ope)

        p = cue.SegmentedPolynomial.stack([pol.polynomial for pol in polys], stacked)
        return cls(operands[: p.num_inputs], operands[p.num_inputs :], p)

    # Method stack_tensor_products is present in SegmentedPolynomial but not in EquivariantPolynomial

    # Method concatenate is present in SegmentedPolynomial but not in EquivariantPolynomial

    # ------------------------------------------------------------------------
    # Standard Python Methods
    # ------------------------------------------------------------------------

    def __repr__(self):
        return self.polynomial.to_string([f"{rep}" for rep in self.operands])

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Evaluate the polynomial on the given inputs.

        Args:
            *inputs (np.ndarray): Input arrays to evaluate the polynomial on.

        Returns:
            list of np.ndarray: List of output arrays.

        Note:
            This is a reference implementation using numpy and may not be optimized for performance.
        """
        return self.polynomial(*inputs)

    def __hash__(self) -> int:
        return hash((self.inputs, self.outputs, self.polynomial))

    def __eq__(self, value) -> bool:
        assert isinstance(value, EquivariantPolynomial)
        return (
            self.inputs == value.inputs
            and self.outputs == value.outputs
            and self.polynomial == value.polynomial
        )

    def __lt__(self, value) -> bool:
        assert isinstance(value, EquivariantPolynomial)
        return (
            self.inputs,
            self.outputs,
            self.polynomial,
        ) < (
            value.inputs,
            value.outputs,
            value.polynomial,
        )

    def __mul__(self, factor: float) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            self.inputs,
            self.outputs,
            self.polynomial * factor,
        )

    def __rmul__(self, factor: float) -> EquivariantPolynomial:
        return self.__mul__(factor)

    # ------------------------------------------------------------------------
    # Analysis Methods
    # ------------------------------------------------------------------------

    def all_same_segment_shape(self) -> bool:
        """Check if all operands have the same segment shape.

        Returns:
            bool: True if all operands have the same segment shape.
        """
        return self.polynomial.all_same_segment_shape()

    def used_inputs(self) -> list[bool]:
        """Get list of boolean values indicating which inputs are used in the polynomial.

        Returns:
            list of bool: List where True indicates the input is used.
        """
        return self.polynomial.used_inputs()

    def used_outputs(self) -> list[bool]:
        """Get list of boolean values indicating which outputs are used in the polynomial.

        Returns:
            list of bool: List where True indicates the output is used.
        """
        return self.polynomial.used_outputs()

    def used_operands(self) -> list[bool]:
        """Get list of boolean values indicating which operands are used in the polynomial.

        Returns:
            list of bool: List where True indicates the operand is used.
        """
        return self.polynomial.used_operands()

    def flop(self, batch_size: int = 1) -> int:
        """Compute the number of floating point operations in the polynomial.

        Args:
            batch_size (int, optional): Batch size for computation. Defaults to 1.

        Returns:
            int: Number of floating point operations.
        """
        return self.polynomial.flop(batch_size)

    def memory(self, batch_sizes: list[int]) -> int:
        """Compute the memory usage of the polynomial.

        Args:
            batch_sizes (list of int): List of batch sizes for each operand. Each operand
                can have its own batch size, allowing for different batch dimensions
                per tensor.

        Returns:
            int: Memory usage in number of elements.
        """
        assert len(batch_sizes) == len(self.operands)
        return sum(Z * rep.dim for Z, rep in zip(batch_sizes, self.operands))

    # ------------------------------------------------------------------------
    # Transformation Methods
    # ------------------------------------------------------------------------

    def apply_fn(
        self,
        f: Callable[
            [cue.Operation, cue.SegmentedTensorProduct],
            tuple[cue.Operation, cue.SegmentedTensorProduct] | None,
        ],
    ) -> EquivariantPolynomial:
        """Apply a function to each tensor product in the polynomial.

        Args:
            f (Callable): Function to apply to each operation and tensor product pair.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: New polynomial with transformed tensor products.
        """
        new_polynomial = self.polynomial.apply_fn(f)
        return EquivariantPolynomial(self.inputs, self.outputs, new_polynomial)

    def fuse_stps(self) -> EquivariantPolynomial:
        """Fuse segmented tensor products with identical operations and operands.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with fused tensor products.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.fuse_stps()
        )

    def consolidate(self) -> EquivariantPolynomial:
        """Consolidate the segmented tensor products by removing empty segments and squeezing modes.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Consolidated polynomial.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.consolidate()
        )

    def flatten_modes(self, modes: list[str]) -> EquivariantPolynomial:
        """Flatten specified modes in the polynomial.

        Args:
            modes (list of str): List of mode names to flatten.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with flattened modes.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.flatten_modes(modes)
        )

    def canonicalize_subscripts(self) -> EquivariantPolynomial:
        """Canonicalize the subscripts of the segmented tensor products.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with canonicalized subscripts.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.canonicalize_subscripts()
        )

    def squeeze_modes(self, modes: str | None = None) -> EquivariantPolynomial:
        """Squeeze specified modes in the polynomial.

        Args:
            modes (str | None, optional): Modes to squeeze. If None, squeezes all modes.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with squeezed modes.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.squeeze_modes(modes)
        )

    def split_mode(self, mode: str, size: int) -> EquivariantPolynomial:
        """Split specified mode in the polynomial.

        Args:
            mode (str): Mode to split.
            size (int): Size to split the mode into.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with split mode.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.split_mode(mode, size)
        )

    def flatten_coefficient_modes(self) -> EquivariantPolynomial:
        """Flatten the coefficient modes of the segmented tensor products.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with flattened coefficient modes.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.flatten_coefficient_modes()
        )

    def symmetrize_for_identical_operands(self) -> EquivariantPolynomial:
        """Symmetrize the paths of the segmented tensor products for identical operands.

        This operation increases the number of paths in the segmented tensor products.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with symmetrized paths.
        """
        return EquivariantPolynomial(
            self.inputs,
            self.outputs,
            self.polynomial.symmetrize_for_identical_operands(),
        )

    def unsymmetrize_for_identical_operands(self) -> EquivariantPolynomial:
        """Unsymmetrize the paths of the segmented tensor products for identical operands.

        This operation decreases the number of paths in the segmented tensor products.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with unsymmetrized paths.
        """
        return EquivariantPolynomial(
            self.inputs,
            self.outputs,
            self.polynomial.unsymmetrize_for_identical_operands(),
        )

    def split_operand_by_irrep(self, operand_id: int) -> EquivariantPolynomial:
        """Split an irreps operand into separate operands for each (mul, ir) pair.

        Args:
            operand_id (int): Index of the operand to split.

        Returns:
            EquivariantPolynomial: New polynomial with the specified operand split by irreps.

        Raises:
            AssertionError: If the operand is not an IrrepsAndLayout instance.

        Example:
            >>> e = cue.descriptors.channelwise_tensor_product(
            ...     cue.Irreps(cue.SO3, "64x0 + 32x1"), cue.Irreps(cue.SO3, "0 + 1"), simplify_irreps3=True
            ... )
            >>> e.split_operand_by_irrep(-1)
            ╭ a=256x0 b=64x0+32x1 c=0+1 -> D=96x0 E=128x1 F=32x2
            │  []·a[u]·b[u]·c[]➜D[u] ─ num_paths=4 u={32, 64}
            │  []·a[u]·b[u]·c[]➜E[u] ─ num_paths=12 u={32, 64}
            ╰─ []·a[u]·b[u]·c[]➜F[u] ─ num_paths=11 u={32, 64}
        """
        operand_id = _canonicalize_index("operand_id", operand_id, self.num_operands)
        operand = self.operands[operand_id]
        assert isinstance(operand, cue.IrrepsAndLayout)

        # Create polynomial slices for each irrep
        offsets = [0]
        i = 0
        for mul, ir in operand.irreps:
            i += mul * ir.dim
            offsets.append(i)

        poly = self.polynomial.split_operand_by_size(operand_id, offsets)

        # # Calculate new operand counts and create mapping function
        num_new = len(offsets) - 1
        new_num_inputs = self.num_inputs + (
            num_new - 1 if operand_id < self.num_inputs else 0
        )

        # Create final operands structure
        splits = tuple(
            cue.IrrepsAndLayout(operand.irreps[i : i + 1], operand.layout)
            for i in range(len(operand.irreps))
        )
        new_operands = (
            self.operands[:operand_id] + splits + self.operands[operand_id + 1 :]
        )

        return EquivariantPolynomial(
            new_operands[:new_num_inputs], new_operands[new_num_inputs:], poly
        )

    # ------------------------------------------------------------------------
    # Filtering Methods
    # ------------------------------------------------------------------------

    def filter_keep_operands(self, keep: list[bool]) -> EquivariantPolynomial:
        """Select which operands to keep in the polynomial.

        Use this method when you want to compute only a subset of the polynomial outputs
        and have control over which inputs to keep. For keeping all inputs (even if
        not used), use filter_keep_outputs. For automatically removing unused operands,
        use filter_drop_unsued_operands.

        Args:
            keep (list of bool): List indicating which operands to keep.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with selected operands.
        """
        assert len(keep) == self.num_operands

        filtered_polynomial = self.polynomial.filter_keep_operands(keep)

        filtered_inputs = [
            rep for rep, k in zip(self.inputs, keep[: self.num_inputs]) if k
        ]
        filtered_outputs = [
            rep for rep, k in zip(self.outputs, keep[self.num_inputs :]) if k
        ]

        return EquivariantPolynomial(
            filtered_inputs, filtered_outputs, filtered_polynomial
        )

    def filter_keep_outputs(self, keep: list[bool]) -> EquivariantPolynomial:
        """Select which outputs to keep in the polynomial.

        Args:
            keep (list of bool): List indicating which outputs to keep.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with selected outputs.
        """
        assert len(keep) == self.num_outputs
        return self.filter_keep_operands([True] * self.num_inputs + keep)

    def filter_drop_unsued_operands(self) -> EquivariantPolynomial:
        """Remove all unused operands from the polynomial.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial with unused operands removed.
        """
        used = self.used_operands()

        filtered_inputs = [
            rep
            for rep, used_flag in zip(self.inputs, used[: self.num_inputs])
            if used_flag
        ]
        filtered_outputs = [
            rep
            for rep, used_flag in zip(self.outputs, used[self.num_inputs :])
            if used_flag
        ]

        filtered_polynomial = self.polynomial.filter_drop_unsued_operands()

        return EquivariantPolynomial(
            filtered_inputs, filtered_outputs, filtered_polynomial
        )

    def compute_only(self, keep: list[bool]) -> EquivariantPolynomial:
        """Create a polynomial that only computes selected outputs.

        The new polynomial will keep the same operands as the original one,
        but will only compute the selected outputs.

        Args:
            keep (list of bool): List indicating which outputs to compute.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: Polynomial computing only selected outputs.
        """
        assert len(keep) == self.num_outputs
        filtered_polynomial = self.polynomial.compute_only(keep)
        return EquivariantPolynomial(self.inputs, self.outputs, filtered_polynomial)

    # ------------------------------------------------------------------------
    # Automatic Differentiation Methods
    # ------------------------------------------------------------------------

    def jvp(
        self, has_tangent: list[bool]
    ) -> tuple[
        EquivariantPolynomial,
        Callable[[tuple[list[Any], list[Any]]], tuple[list[Any], list[Any]]],
    ]:
        """Compute the Jacobian-vector product of the polynomial.

        Args:
            has_tangent (list of bool): List indicating which inputs have tangents.

        Returns:
            tuple of :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>` and Callable:
                The JVP polynomial and a mapping function for inputs/outputs.
        """
        p, m = self.polynomial.jvp(has_tangent)
        return EquivariantPolynomial(*m((self.inputs, self.outputs)), p), m

    def transpose(
        self,
        is_undefined_primal: list[bool],
        has_cotangent: list[bool],
    ) -> tuple[
        EquivariantPolynomial,
        Callable[[tuple[list[Any], list[Any]]], tuple[list[Any], list[Any]]],
    ]:
        """Transpose the polynomial for reverse-mode automatic differentiation.

        Args:
            is_undefined_primal (list of bool): List indicating which inputs have undefined primals.
            has_cotangent (list of bool): List indicating which outputs have cotangents.

        Returns:
            tuple of :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>` and Callable:
                The transposed polynomial and a mapping function for inputs/outputs.
        """
        p, m = self.polynomial.transpose(is_undefined_primal, has_cotangent)
        return EquivariantPolynomial(*m((self.inputs, self.outputs)), p), m

    def backward(
        self, requires_gradient: list[bool], has_cotangent: list[bool]
    ) -> tuple[
        EquivariantPolynomial,
        Callable[[tuple[list[Any], list[Any]]], tuple[list[Any], list[Any]]],
    ]:
        """Compute the backward pass of the polynomial for gradient computation.

        Args:
            requires_gradient (list of bool): List indicating which inputs require gradients.
            has_cotangent (list of bool): List indicating which outputs have cotangents.

        Returns:
            tuple of :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>` and Callable:
                The backward polynomial and a mapping function for inputs/outputs.
        """
        p, m = self.polynomial.backward(requires_gradient, has_cotangent)
        return EquivariantPolynomial(*m((self.inputs, self.outputs)), p), m
