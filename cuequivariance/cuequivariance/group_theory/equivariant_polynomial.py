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


@dataclasses.dataclass(init=False, frozen=True)
class EquivariantPolynomial:
    """A polynomial representation with equivariance constraints.

    This class extends :class:`cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>` by incorporating information about the group representations
    of each input and output tensor. It ensures that operations performed by the polynomial respect
    the equivariance constraints defined by these representations, making it suitable for building
    equivariant neural networks.

    Args:
        inputs (list of :class:`cue.Rep <cuequivariance.Rep>`): Group representations for all operands (inputs and outputs).
        outputs (list of :class:`cue.Rep <cuequivariance.Rep>`): Group representations for all operands (inputs and outputs).
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
        """The group representations of the operands (inputs + outputs)."""
        return self.inputs + self.outputs

    @property
    def num_operands(self) -> int:
        """The total number of operands (inputs and outputs) in the polynomial."""
        return len(self.operands)

    @property
    def num_inputs(self) -> int:
        """The number of input tensors expected by the polynomial."""
        return self.polynomial.num_inputs

    @property
    def num_outputs(self) -> int:
        """The number of output tensors produced by the polynomial."""
        return self.polynomial.num_outputs

    # ------------------------------------------------------------------------
    # Class Construction Methods
    # ------------------------------------------------------------------------

    # Method eval_last_operand is present in SegmentedPolynomial but not in EquivariantPolynomial

    @classmethod
    def stack(
        cls, polys: list[EquivariantPolynomial], stacked: list[bool]
    ) -> EquivariantPolynomial:
        """Stack multiple equivariant polynomials together.

        This method combines multiple polynomials by stacking their operands according to the
        stacked parameter. Operands with the same index that are not stacked must be identical
        across all polynomials.

        Args:
            polys (list of :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`): List of polynomials to stack.
            stacked (list of bool): Boolean flags indicating which operands should be stacked.

        Returns:
            :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: A new polynomial combining the stacked polynomials.

        Raises:
            ValueError: If operands that are not stacked differ across polynomials.
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
            *inputs (numpy.ndarray): Input tensors to evaluate the polynomial on.

        Returns:
            list of numpy.ndarray: Output tensors resulting from the polynomial evaluation.
        """
        return self.polynomial(*inputs)

    def __hash__(self) -> int:
        return hash((self.operands, self.polynomial))

    def __eq__(self, value) -> bool:
        assert isinstance(value, EquivariantPolynomial)
        return self.operands == value.operands and self.polynomial == value.polynomial

    def __lt__(self, value) -> bool:
        assert isinstance(value, EquivariantPolynomial)
        return (
            self.num_inputs,
            self.num_outputs,
            self.operands,
            self.polynomial,
        ) < (
            value.num_inputs,
            value.num_outputs,
            value.operands,
            value.polynomial,
        )

    def __mul__(self, factor: float) -> EquivariantPolynomial:
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial * factor
        )

    def __rmul__(self, factor: float) -> EquivariantPolynomial:
        return self.__mul__(factor)

    # ------------------------------------------------------------------------
    # Analysis Methods
    # ------------------------------------------------------------------------

    def all_same_segment_shape(self) -> bool:
        """Whether all the segments have the same shape."""
        return self.polynomial.all_same_segment_shape()

    # Method used_inputs is present in SegmentedPolynomial but not in EquivariantPolynomial

    # Method used_outputs is present in SegmentedPolynomial but not in EquivariantPolynomial

    # Method used_buffers is present in SegmentedPolynomial but not in EquivariantPolynomial

    def flop(self, batch_size: int = 1) -> int:
        """Compute the number of floating point operations in the polynomial.

        Args:
            batch_size (int, optional): The batch size for the computation. Defaults to 1.

        Returns:
            int: The estimated number of floating-point operations.
        """
        return self.polynomial.flop(batch_size)

    def memory(self, batch_sizes: list[int]) -> int:
        """Compute the memory usage of the polynomial.

        Args:
            batch_sizes (list of int): The batch sizes for each operand.

        Returns:
            int: The estimated memory usage in number of scalar elements.
        """
        assert len(batch_sizes) == len(self.operands)
        return sum(Z * rep.dim for Z, rep in zip(batch_sizes, self.operands))

    # ------------------------------------------------------------------------
    # Transformation Methods
    # ------------------------------------------------------------------------

    # Method apply_fn is present in SegmentedPolynomial but not in EquivariantPolynomial

    def fuse_stps(self) -> EquivariantPolynomial:
        """Fuse segmented tensor products with identical operations and operands.

        Returns:
            EquivariantPolynomial: A new polynomial with fused tensor products.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.fuse_stps()
        )

    def consolidate(self) -> EquivariantPolynomial:
        """Consolidate the segmented tensor products.

        Returns:
            EquivariantPolynomial: A new polynomial with consolidated tensor products.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.consolidate()
        )

    def flatten_modes(self, modes: list[str]) -> EquivariantPolynomial:
        """Flatten the specified modes of the segmented tensor products."""
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.flatten_modes(modes)
        )

    def canonicalize_subscripts(self) -> EquivariantPolynomial:
        """Canonicalize the subscripts of the segmented tensor products."""
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.canonicalize_subscripts()
        )

    def squeeze_modes(self, modes: str | None = None) -> EquivariantPolynomial:
        """Squeeze the modes of the segmented tensor products.

        Returns:
            EquivariantPolynomial: A new polynomial with squeezed modes.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.squeeze_modes(modes)
        )

    def flatten_coefficient_modes(self) -> EquivariantPolynomial:
        """Flatten the coefficient modes of the segmented tensor products.

        Returns:
            EquivariantPolynomial: A new polynomial with flattened coefficient modes.
        """
        return EquivariantPolynomial(
            self.inputs, self.outputs, self.polynomial.flatten_coefficient_modes()
        )

    # Method symmetrize_for_identical_operands is present in SegmentedPolynomial but not in EquivariantPolynomial

    # Method unsymmetrize_for_identical_operands is present in SegmentedPolynomial but not in EquivariantPolynomial

    # ------------------------------------------------------------------------
    # Filtering Methods
    # ------------------------------------------------------------------------

    # Method filter_keep_operands is present in SegmentedPolynomial but not in EquivariantPolynomial

    # Method filter_keep_outputs is present in SegmentedPolynomial but not in EquivariantPolynomial

    # Method filter_drop_unsued_operands is present in SegmentedPolynomial but not in EquivariantPolynomial

    # Method compute_only is present in SegmentedPolynomial but not in EquivariantPolynomial

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

        This method creates a new polynomial that, when evaluated, computes the Jacobian-vector
        product of the original polynomial. This is used for forward-mode automatic differentiation.

        Args:
            has_tangent (list of bool): Boolean flags indicating which inputs have tangent vectors.

        Returns:
            tuple: A tuple containing:
                - :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: A new polynomial representing the JVP operation.
                - callable: A function that maps input/output representations to JVP input/output representations.
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
        """Transpose the polynomial operation.

        This method creates a new polynomial that represents the transpose of the original operation.
        The transpose is essential for reverse-mode automatic differentiation.

        Args:
            is_undefined_primal (list of bool): Boolean flags indicating which inputs are undefined primals.
            has_cotangent (list of bool): Boolean flags indicating which outputs have cotangents.

        Returns:
            tuple: A tuple containing:
                - :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: A new polynomial representing the transposed operation.
                - callable: A function that maps input/output representations to transposed input/output representations.

        Raises:
            ValueError: If the polynomial is non-linear and cannot be transposed.
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

        This method combines the JVP and transpose operations to create a new polynomial that,
        when evaluated, computes gradients of outputs with respect to inputs. This is the
        core operation in reverse-mode automatic differentiation.

        Args:
            requires_gradient (list of bool): Boolean flags indicating which inputs require gradients.
            has_cotangent (list of bool): Boolean flags indicating which outputs have cotangents.

        Returns:
            tuple: A tuple containing:
                - :class:`cue.EquivariantPolynomial <cuequivariance.EquivariantPolynomial>`: A new polynomial for gradient computation.
                - callable: A function that maps input/output representations to backward input/output representations.
        """
        p, m = self.polynomial.backward(requires_gradient, has_cotangent)
        return EquivariantPolynomial(*m((self.inputs, self.outputs)), p), m
