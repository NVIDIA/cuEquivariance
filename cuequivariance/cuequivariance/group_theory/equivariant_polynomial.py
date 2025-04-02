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

import numpy as np

import cuequivariance as cue


@dataclasses.dataclass(init=False, frozen=True)
class EquivariantPolynomial:
    """A polynomial representation with equivariance constraints.

    This class extends SegmentedPolynomial by incorporating information about the group representations
    of each input and output tensor. It ensures that operations performed by the polynomial respect
    the equivariance constraints defined by these representations, making it suitable for building
    equivariant neural networks.

    Args:
        operands (list[cue.Rep]): Group representations for all operands (inputs and outputs).
        polynomial (cue.SegmentedPolynomial): The underlying polynomial transformation.
    """

    operands: tuple[cue.Rep, ...]
    polynomial: cue.SegmentedPolynomial

    def __init__(self, operands: list[cue.Rep], polynomial: cue.SegmentedPolynomial):
        assert isinstance(polynomial, cue.SegmentedPolynomial)
        object.__setattr__(self, "operands", tuple(operands))
        object.__setattr__(self, "polynomial", polynomial)
        if len(self.operands) != self.polynomial.num_operands:
            raise ValueError(
                f"Number of operands {len(self.operands)} must equal the number of inputs"
                f" {self.polynomial.num_inputs} plus the number of outputs {self.polynomial.num_outputs}"
            )
        for rep, ope in zip(self.operands, self.polynomial.operands):
            assert ope.size == rep.dim, (
                f"{ope} incompatible with {rep}. {ope.size=} != {rep.dim=}"
            )

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
        return EquivariantPolynomial(self.operands, self.polynomial * factor)

    def __rmul__(self, factor: float) -> EquivariantPolynomial:
        return self.__mul__(factor)

    def __repr__(self):
        return self.polynomial.to_string([f"{rep}" for rep in self.operands])

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Evaluate the polynomial on the given inputs.

        Args:
            *inputs (np.ndarray): Input tensors to evaluate the polynomial on.

        Returns:
            list[np.ndarray]: Output tensors resulting from the polynomial evaluation.
        """
        return self.polynomial(*inputs)

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

    @property
    def inputs(self) -> tuple[cue.Rep, ...]:
        """The group representations of the input tensors."""
        return self.operands[: self.num_inputs]

    @property
    def outputs(self) -> tuple[cue.Rep, ...]:
        """The group representations of the output tensors."""
        return self.operands[self.num_inputs :]

    def fuse_stps(self) -> EquivariantPolynomial:
        """Fuse segmented tensor products with identical operations and operands.

        Returns:
            EquivariantPolynomial: A new polynomial with fused tensor products.
        """
        return EquivariantPolynomial(self.operands, self.polynomial.fuse_stps())

    def consolidate(self) -> EquivariantPolynomial:
        """Consolidate the segmented tensor products.

        Returns:
            EquivariantPolynomial: A new polynomial with consolidated tensor products.
        """
        return EquivariantPolynomial(self.operands, self.polynomial.consolidate())

    @classmethod
    def stack(
        cls, polys: list[EquivariantPolynomial], stacked: list[bool]
    ) -> EquivariantPolynomial:
        """Stack multiple equivariant polynomials together.

        This method combines multiple polynomials by stacking their operands according to the
        stacked parameter. Operands with the same index that are not stacked must be identical
        across all polynomials.

        Args:
            polys (list[EquivariantPolynomial]): List of polynomials to stack.
            stacked (list[bool]): Boolean flags indicating which operands should be stacked.

        Returns:
            EquivariantPolynomial: A new polynomial combining the stacked polynomials.

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

        return cls(
            operands,
            cue.SegmentedPolynomial.stack([pol.polynomial for pol in polys], stacked),
        )

    def squeeze_modes(self) -> EquivariantPolynomial:
        """Squeeze the modes of the segmented tensor products.

        Returns:
            EquivariantPolynomial: A new polynomial with squeezed modes.
        """
        return EquivariantPolynomial(
            self.operands,
            self.polynomial.squeeze_modes(),
        )

    def flatten_coefficient_modes(self) -> EquivariantPolynomial:
        """Flatten the coefficient modes of the segmented tensor products.

        Returns:
            EquivariantPolynomial: A new polynomial with flattened coefficient modes.
        """
        return EquivariantPolynomial(
            self.operands,
            self.polynomial.flatten_coefficient_modes(),
        )

    def jvp(self, has_tangent: list[bool]) -> EquivariantPolynomial:
        """Compute the Jacobian-vector product of the polynomial.

        This method creates a new polynomial that, when evaluated, computes the Jacobian-vector
        product of the original polynomial. This is used for forward-mode automatic differentiation.

        Args:
            has_tangent (list[bool]): Boolean flags indicating which inputs have tangent vectors.

        Returns:
            EquivariantPolynomial: A new polynomial representing the JVP operation.
        """
        return EquivariantPolynomial(
            list(self.inputs)
            + [x for has, x in zip(has_tangent, self.inputs) if has]
            + list(self.outputs),
            self.polynomial.jvp(has_tangent),
        )

    def transpose(
        self,
        is_undefined_primal: list[bool],
        has_cotangent: list[bool],
    ) -> EquivariantPolynomial:
        """Transpose the polynomial operation.

        This method creates a new polynomial that represents the transpose of the original operation.
        The transpose is essential for reverse-mode automatic differentiation.

        Args:
            is_undefined_primal (list[bool]): Boolean flags indicating which inputs are undefined primals.
            has_cotangent (list[bool]): Boolean flags indicating which outputs have cotangents.

        Returns:
            EquivariantPolynomial: A new polynomial representing the transposed operation.

        Raises:
            ValueError: If the polynomial is non-linear and cannot be transposed.
        """
        return EquivariantPolynomial(
            # defined inputs
            [
                x
                for is_undefined, x in zip(is_undefined_primal, self.inputs)
                if not is_undefined
            ]
            # cotangent outputs
            + [x for has, x in zip(has_cotangent, self.outputs) if has]
            # undefined inputs
            + [
                x
                for is_undefined, x in zip(is_undefined_primal, self.inputs)
                if is_undefined
            ],
            self.polynomial.transpose(is_undefined_primal, has_cotangent),
        )

    def backward(
        self, requires_gradient: list[bool], has_cotangent: list[bool]
    ) -> EquivariantPolynomial:
        """Compute the backward pass of the polynomial for gradient computation.

        This method combines the JVP and transpose operations to create a new polynomial that,
        when evaluated, computes gradients of outputs with respect to inputs. This is the
        core operation in reverse-mode automatic differentiation.

        Args:
            requires_gradient (list[bool]): Boolean flags indicating which inputs require gradients.
            has_cotangent (list[bool]): Boolean flags indicating which outputs have cotangents.

        Returns:
            EquivariantPolynomial: A new polynomial for gradient computation.
        """
        return EquivariantPolynomial(
            list(self.inputs)
            + [x for has, x in zip(has_cotangent, self.outputs) if has]
            + [x for req, x in zip(requires_gradient, self.inputs) if req],
            self.polynomial.backward(requires_gradient, has_cotangent),
        )

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
            batch_sizes (list[int]): The batch sizes for each operand.

        Returns:
            int: The estimated memory usage in number of scalar elements.
        """
        assert len(batch_sizes) == len(self.operands)
        return sum(Z * rep.dim for Z, rep in zip(batch_sizes, self.operands))
