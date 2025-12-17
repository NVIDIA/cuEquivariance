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
from __future__ import annotations

import dataclasses
import itertools
from collections import defaultdict

IVARS = "abcdefghijklmnopqrstuvwxyz"
OVARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclasses.dataclass(init=False, frozen=True)
class Operation:
    """Descriptor mapping input/output buffers to tensor product operands.

    The buffers are identified by their index (0, 1, 2, ...).
    The order of the buffers corresponds to the order of the operands.

    Example:

        This list of operations would typically be used for the symmetric contraction operation.

        >>> ops = [
        ...     Operation((0, 1, 2)),
        ...     Operation((0, 1, 1, 2)),
        ...     Operation((0, 1, 1, 1, 2)),
        ... ]
        >>> print(Operation.list_to_string(ops, 2, 1))
        (a, b) -> (C)
          a b C
          a b b C
          a b b b C
    """

    buffers: tuple[int, ...]

    def __init__(self, buffers: tuple[int, ...] | Operation):
        if isinstance(buffers, Operation):
            buffers = buffers.buffers
        assert len(buffers) > 0, buffers
        assert all(isinstance(b, int) for b in buffers), buffers
        assert all(i >= 0 for i in buffers), buffers
        object.__setattr__(self, "buffers", tuple(int(b) for b in buffers))

    def __repr__(self):
        return f"Operation({self.buffers})"

    def to_letters(self, num_inputs: int) -> list[str]:
        return [IVARS[b] if b < num_inputs else OVARS[b] for b in self.buffers]

    @staticmethod
    def list_to_string(
        operations: list[Operation], num_inputs: int, num_outputs: int
    ) -> str:
        i = ", ".join(IVARS[:num_inputs])
        o = ", ".join(OVARS[num_inputs : num_inputs + num_outputs])
        s = f"({i}) -> ({o})"
        for op in operations:
            s += "\n  " + " ".join(op.to_letters(num_inputs))
        return s

    def __lt__(self, value):
        assert isinstance(value, Operation)
        return (len(self.buffers), self.buffers) < (len(value.buffers), value.buffers)

    def __hash__(self) -> int:
        return hash(self.buffers)

    def __eq__(self, value):
        assert isinstance(value, Operation)
        return self.buffers == value.buffers

    def permute_operands(self, permutation: tuple[int, ...]) -> Operation:
        return Operation(tuple(self.buffers[p] for p in permutation))

    def move_operand_last(self, operand: int) -> Operation:
        buffers = list(self.buffers)
        b = buffers.pop(operand)
        buffers.append(b)
        return Operation(tuple(buffers))

    def input_operands_buffers(self, num_inputs: int) -> list[tuple[int, int]]:
        return [(op, i) for op, i in enumerate(self.buffers) if i < num_inputs]

    def output_operand_buffer(self, num_inputs: int) -> tuple[int, int]:
        def _raise():
            raise ValueError(
                f"Operation must have exactly one output buffer. {self=} {num_inputs=}"
            )

        result = None
        for op, i in enumerate(self.buffers):
            if i < num_inputs:
                continue
            if result is not None:
                _raise()
            result = (op, i)
        if result is None:
            _raise()

        return result

    def input_buffers(self, num_inputs: int) -> list[int]:
        return [i for i in self.buffers if i < num_inputs]

    def output_buffer(self, num_inputs: int) -> int:
        return self.output_operand_buffer(num_inputs)[1]

    def transpose(
        self,
        is_undefined_primal: list[bool],
        has_cotangent: list[bool],
    ) -> Operation | None:
        """
        Args:
            is_undefined_primal (list[bool]): whether the primal is undefined
            has_cotangent (list[bool]): whether the cotangent is defined

        Returns:
            Operation: the transposed operation, if any
                in the returned operation, the buffers are:
                 - new inputs: defined primals + cotangents (=True)
                 - new outputs: undefined primals
        """
        # number of input buffers in the original operation
        # note that self might not involve all input buffers
        num_inputs = len(is_undefined_primal)

        if not has_cotangent[self.output_buffer(num_inputs) - num_inputs]:
            # The output buffer of self has no cotangent,
            # so there is no contribution to the primal.
            return None

        num_undef_primal = 0
        for i in self.input_buffers(num_inputs):
            if is_undefined_primal[i]:
                num_undef_primal += 1

        if num_undef_primal > 1:
            raise ValueError(
                f"Operation must have at most one undefined primal input. {self=} {is_undefined_primal=}."
                " Otherwise it means that the operation has not been linearized correctly."
            )
        if num_undef_primal == 0:
            # The operation has no undefined primal as input
            return None

        new_num_inputs = 0
        new_num_outputs = 0
        primals_to_new_input: list[int | None] = []
        primals_to_new_output: list[int | None] = []

        for undef in is_undefined_primal:
            if undef:
                primals_to_new_input.append(None)
                primals_to_new_output.append(new_num_outputs)
                new_num_outputs += 1
            else:
                primals_to_new_input.append(new_num_inputs)
                primals_to_new_output.append(None)
                new_num_inputs += 1

        outputs_to_new_input: list[int | None] = []
        for has in has_cotangent:
            if has:
                outputs_to_new_input.append(new_num_inputs)
                new_num_inputs += 1
            else:
                outputs_to_new_input.append(None)

        # output buffers are identified by their index being >= new_num_inputs
        primals_to_new_output = [
            None if i is None else new_num_inputs + i for i in primals_to_new_output
        ]

        new_buffers = []
        for i in self.buffers:
            if i < num_inputs:
                if is_undefined_primal[i]:
                    new_buffers.append(primals_to_new_output[i])
                else:
                    new_buffers.append(primals_to_new_input[i])
            else:
                new_buffers.append(outputs_to_new_input[i - num_inputs])

        return Operation(tuple(new_buffers))

    def jvp(self, has_tangent: list[bool]) -> list[Operation]:
        """
        Args:
            has_tangent (list[bool]): whether the input has a tangent

        Returns:
            list[Operation]: the JVPs of the operation
                in the returned operations, the buffers are:
                 - new inputs: original inputs + tangents (=True)
                 - new outputs: original outputs
        """
        # number of input buffers in the original operation
        # note that self might not involve all input buffers
        num_inputs = len(has_tangent)

        new_num_inputs = num_inputs
        mapping: list[int | None] = []
        for has in has_tangent:
            if has:
                mapping.append(new_num_inputs)
                new_num_inputs += 1
            else:
                mapping.append(None)

        jvps = []
        for op, i in self.input_operands_buffers(num_inputs):
            if has_tangent[i]:
                new_buffers = list(self.buffers)
                new_buffers[op] = mapping[i]
                op, i = self.output_operand_buffer(num_inputs)
                new_buffers[op] += new_num_inputs - num_inputs
                jvps.append(Operation(new_buffers))

        return jvps

    def operands_with_identical_buffers(self) -> frozenset[frozenset[int]]:
        """
        Groups of operands sharing the same buffer.
        """
        bid_to_oid = defaultdict(list)
        for oid, b in enumerate(self.buffers):
            bid_to_oid[b].append(oid)
        return frozenset(map(frozenset, bid_to_oid.values()))

    @staticmethod
    def group_by_idential_buffers(
        operations: list[Operation],
    ) -> list[tuple[frozenset[frozenset[int]], list[Operation]]]:
        """
        Args:
            operations (list[Operation]): the operations to group
            num_inputs (int): the number of input buffers

        Returns:
            list of tuples: Each tuple contains:
                - frozenset of frozensets of operands bound to identical buffers
                - list of operations
        """
        return [
            (p, list(group))
            for p, group in itertools.groupby(
                operations, key=Operation.operands_with_identical_buffers
            )
        ]

    @staticmethod
    def group_by_operational_symmetries(
        symmetries: list[tuple[int, ...]],
        operations: list[Operation],
    ) -> list[tuple[int, Operation]]:
        """
        Args:
            symmetries (list[tuple[int, ...]]): the permutation group
            operations (list[Operation]): the operations to group

        Returns:
            list of tuples: Each tuple contains:
                - multiplicity (int)
                - a representative operation
        """

        def partition(operation: Operation) -> tuple[int, ...]:
            return frozenset(operation.permute_operands(perm) for perm in symmetries)

        groups = []
        for _, group in itertools.groupby(operations, key=partition):
            group = sorted(group)
            groups.append((len(group), group[0]))
        return groups
