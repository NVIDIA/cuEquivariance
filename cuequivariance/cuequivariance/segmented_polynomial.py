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
from typing import Callable, Sequence

import cuequivariance as cue
from cuequivariance.operation import IVARS, OVARS


@dataclasses.dataclass(init=False, frozen=True)
class SegmentedPolynomial:
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

    def __hash__(self) -> int:
        return hash((self.num_inputs, self.num_outputs, tuple(self.tensor_products)))

    def __eq__(self, value):
        assert isinstance(value, SegmentedPolynomial)
        return (
            self.num_inputs == value.num_inputs
            and self.num_outputs == value.num_outputs
            and self.tensor_products == value.tensor_products
        )

    def __lt__(self, value):
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
        header = (
            " ".join(IVARS[: self.num_inputs])
            + " -> "
            + " ".join(OVARS[self.num_inputs : self.num_inputs + self.num_outputs])
            + " "
        )
        ope_txts = [
            "  " + ope.to_string(self.num_inputs) for ope, _ in self.tensor_products
        ]
        n = max(len(ope_txt) for ope_txt in ope_txts)
        n = max(len(header), n)

        text = header + "═" * (n - len(header)) + "═╗"

        for ope_txt, (_, stp) in zip(ope_txts, self.tensor_products):
            text += "\n" + ope_txt + " " * (n - len(ope_txt)) + " ║ " + str(stp)
        return text

    @property
    def buffer_sizes(self) -> list[int | None]:
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
        return self.buffer_sizes[: self.num_inputs]

    @property
    def output_sizes(self) -> list[int | None]:
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

    def jvp(self, has_tangent: list[bool]) -> SegmentedPolynomial:
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
