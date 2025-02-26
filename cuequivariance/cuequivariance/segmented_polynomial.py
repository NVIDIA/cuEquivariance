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
from typing import Sequence

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

    def __repr__(self):
        text = ""
        text += " ".join(IVARS[self.num_inputs])
        text += " -> "
        text += " ".join(OVARS[self.num_inputs : self.num_inputs + self.num_outputs])
        tab = "\n  "
        for ope, stp in self.tensor_products:
            text += tab + f"{stp}"
            text += tab + ope.to_string(self.num_inputs)
        return text
