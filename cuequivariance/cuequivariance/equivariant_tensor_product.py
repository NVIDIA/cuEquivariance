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
from typing import Optional, Sequence, Union

import cuequivariance as cue
from cuequivariance import segmented_tensor_product as stp


@dataclasses.dataclass(init=True, frozen=True)
class Operand:
    irreps: cue.Irreps
    layout: cue.IrrepsLayout


@dataclasses.dataclass(init=False, frozen=True)
class EquivariantTensorProduct:
    """
    Descriptor of an equivariant tensor product.
    This class is a wrapper around a list of :class:`STP <cuequivariance.SegmentedTensorProduct>`.
    While an STP is a single homogeneous polynomial without specification of the role of each operand,
    an ETP determines the role of each operand (input or output), the representation of each operand (irreps),
    and the layout of each operand (multiplicity first or irreducible representation first).

    Requirements:
        - An ETP must contain at least one :class:`STP <cuequivariance.SegmentedTensorProduct>`.
        - Each STP must have at least one operand (the output).

    Examples:
        +------+--------+--------+--------+--------+---------------------------------------------------+
        |      | Input0 | Input1 | Input2 | Output |                    Comment                        |
        +======+========+========+========+========+===================================================+
        | STP0 |   x    |   x    |   x    |   x    |  common case, the number of operands is the same  |
        +------+--------+--------+--------+--------+---------------------------------------------------+
        | STP1 |   x    |   x    |        |   x    |  some inputs are not used by all STPs             |
        +------+--------+--------+--------+--------+---------------------------------------------------+
        | STP2 |   x    |        |        |   x    |  -- " --                                          |
        +------+--------+--------+--------+--------+---------------------------------------------------+
        | STP3 |        |        |        |   x    |  -- " --                                          |
        +------+--------+--------+--------+--------+---------------------------------------------------+
        | STP4 |   x    |   x    |  x x x |   x    |  the last input is fed multiple times             |
        +------+--------+--------+--------+--------+---------------------------------------------------+

    .. rubric:: Methods
    """

    operands: tuple[Operand, ...]
    ds: list[stp.SegmentedTensorProduct]

    def __init__(
        self,
        d: Union[stp.SegmentedTensorProduct, Sequence[stp.SegmentedTensorProduct]],
        operands: list[Union[cue.Irreps, Operand]],
        layout: Optional[cue.IrrepsLayout] = None,
    ):
        operands = tuple(
            (
                ope
                if isinstance(ope, Operand)
                else Operand(
                    irreps=cue.Irreps(ope), layout=layout or cue.get_layout_scope()
                )
            )
            for ope in operands
        )
        if isinstance(d, stp.SegmentedTensorProduct):
            assert len(operands) == d.num_operands
            for oid in range(d.num_operands):
                assert operands[oid].irreps.dim == d.operands[oid].size
            ds = [d]
        else:
            ds = list(d)

            nin = len(operands) - 1

            for d in ds:
                # all non-repeated input operands are the same
                for i in range(nin - 1):
                    if not (i < d.num_operands - 1):
                        continue

                    if operands[i].irreps.dim != d.operands[i].size:
                        raise ValueError(
                            f"Input {i} size mismatch: {operands[i].irreps} vs {d.operands[i]}"
                        )
                # the repeated input operand is the same
                assert len(operands) >= 2
                for d_ope in d.operands[nin - 1 : -1]:
                    if operands[-2].irreps.dim != d_ope.size:
                        raise ValueError(
                            f"Last input size mismatch: {operands[-2].irreps} vs {d_ope}"
                        )
                if operands[-1].irreps.dim != d.operands[-1].size:
                    raise ValueError(
                        f"Output size mismatch: {operands[-1].irreps} vs {d.operands[-1]}"
                    )

            # all non-repeated inputs have the same operands
            for i in range(nin - 1):
                assert len({d.operands[i] for d in ds if i < d.num_operands - 1}) <= 1
            # all the repeated inputs have the same operands
            if len({o for d in ds for o in d.operands[nin - 1 : -1]}) > 1:
                tmp = {o for d in ds for o in d.operands[nin - 1 : -1]}
                raise ValueError(f"Different operands for the last input, {tmp}")
            # all the output operands are the same
            assert len({d.operands[-1] for d in ds}) == 1

        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "ds", ds)

    def __hash__(self) -> int:
        return hash((self.operands, tuple(self.ds)))

    @property
    def d(self) -> stp.SegmentedTensorProduct:
        assert len(self.ds) == 1
        return self.ds[0]

    @property
    def num_operands(self) -> int:
        return len(self.operands)

    @property
    def num_inputs(self) -> int:
        return self.num_operands - 1

    @property
    def inputs(self) -> tuple[Operand, ...]:
        return self.operands[:-1]

    @property
    def output(self) -> Operand:
        return self.operands[-1]

    def _degrees(self, i: int) -> set[int]:
        assert 0 <= i < self.num_inputs
        deg = set()
        if i == self.num_inputs - 1:
            for d in self.ds:
                deg.add(max(0, d.num_operands - self.num_inputs))
        else:
            for d in self.ds:
                deg.add(1 if i < d.num_operands - 1 else 0)
        return deg

    def __repr__(self) -> str:
        irs = [str(ope.irreps) for ope in self.operands]
        inputs = irs[:-1]
        output = irs[-1]

        for i in range(self.num_inputs):
            d = self._degrees(i)
            if d == set():
                inputs[i] = f"({inputs[i]})"
            elif len(d) > 1:
                inputs[i] = f"({inputs[i]})^({min(d)}..{max(d)})"
            elif d != {1}:
                inputs[i] = f"({inputs[i]})^{min(d)}"

        return f"{self.__class__.__name__}({' x '.join(inputs)} -> {output})"

    def permute_operands(
        self, permutation: tuple[int, ...]
    ) -> EquivariantTensorProduct:
        """Permute the operands of the tensor product."""
        assert sorted(permutation) == list(range(self.num_operands))
        assert all(d.num_operands == self.num_operands for d in self.ds)
        return EquivariantTensorProduct(
            [d.permute_operands(permutation) for d in self.ds],
            tuple(self.operands[pid] for pid in permutation),
        )

    def move_operand(self, src: int, dst: int) -> EquivariantTensorProduct:
        """Move an operand to a new position."""
        if src < 0:
            src += self.num_operands
        if dst < 0:
            dst += self.num_operands
        return self.permute_operands(
            tuple(
                {src: dst, dst: src}.get(oid, oid) for oid in range(self.num_operands)
            )
        )

    def move_operand_first(self, src: int) -> EquivariantTensorProduct:
        """Move an operand to the front."""
        return self.move_operand(src, 0)

    def move_operand_last(self, src: int) -> EquivariantTensorProduct:
        """Move an operand to the back."""
        return self.move_operand(src, -1)

    def squeeze_modes(self, modes: Optional[str] = None) -> EquivariantTensorProduct:
        """Squeeze the modes."""
        return EquivariantTensorProduct(
            [d.squeeze_modes(modes) for d in self.ds], self.operands
        )

    def consolidate_paths(self) -> EquivariantTensorProduct:
        """Consolidate the paths."""
        return EquivariantTensorProduct(
            [d.consolidate_paths() for d in self.ds], self.operands
        )

    def canonicalize_subscripts(self) -> EquivariantTensorProduct:
        """Canonicalize the subscripts."""
        return EquivariantTensorProduct(
            [d.canonicalize_subscripts() for d in self.ds], self.operands
        )

    def flatten_modes(
        self, modes: str, *, skip_zeros: bool = True, force: bool = False
    ) -> EquivariantTensorProduct:
        """Flatten modes."""
        return EquivariantTensorProduct(
            [
                d.flatten_modes(modes, skip_zeros=skip_zeros, force=force)
                for d in self.ds
            ],
            self.operands,
        )

    def all_same_segment_shape(self) -> bool:
        """Whether all the segments have the same shape."""
        return all(d.all_same_segment_shape() for d in self.ds)

    def flatten_coefficient_modes(self) -> EquivariantTensorProduct:
        """Flatten the coefficient modes."""
        return EquivariantTensorProduct(
            [d.flatten_coefficient_modes() for d in self.ds], self.operands
        )

    def _map_operands_from_stp_to_etp(self, d: stp.SegmentedTensorProduct) -> list[int]:
        inputs = list(range(self.num_operands - 1))
        output = self.num_operands - 1

        if d.num_operands == self.num_operands:
            return inputs + [output]
        if d.num_operands < self.num_operands:
            return inputs[: d.num_operands - 1] + [output]
        if d.num_operands > self.num_operands:
            return (
                inputs + [inputs[-1]] * (d.num_operands - self.num_operands) + [output]
            )

    def change_layout(
        self, layout: Union[cue.IrrepsLayout, list[cue.IrrepsLayout]]
    ) -> EquivariantTensorProduct:
        if isinstance(layout, Sequence):
            layouts = list(layout)
            assert len(layouts) == self.num_operands
        else:
            layouts = [layout] * self.num_operands
        del layout
        layouts = [cue.IrrepsLayout.as_layout(layout) for layout in layouts]

        def f(d: stp.SegmentedTensorProduct) -> stp.SegmentedTensorProduct:
            ii = self._map_operands_from_stp_to_etp(d)
            assert len(ii) == d.num_operands

            operands = [self.operands[i] for i in ii]
            layouts_ = [layouts[i] for i in ii]

            new_subscripts = []
            for oid, (operand, layout) in enumerate(zip(operands, layouts_)):
                subscripts = d.subscripts.operands[oid]
                if operand.layout == layout:
                    new_subscripts.append(subscripts)
                    continue
                if operand.irreps.layout_insensitive():
                    new_subscripts.append(subscripts)
                    continue
                assert d.operands[oid].num_segments == len(operand.irreps)
                assert len(subscripts) > 0
                if operand.layout == cue.ir_mul:
                    assert [seg[0] for seg in d.operands[oid]] == [
                        ir.dim for _, ir in operand.irreps
                    ]
                    new_subscripts.append(subscripts[1:] + subscripts[:1])
                    continue
                if operand.layout == cue.mul_ir:
                    assert [seg[-1] for seg in d.operands[oid]] == [
                        ir.dim for _, ir in operand.irreps
                    ]
                    new_subscripts.append(subscripts[-1:] + subscripts[:-1])
                    continue
                raise NotImplementedError
            return d.add_or_transpose_modes(
                stp.Subscripts.from_operands(new_subscripts, d.coefficient_subscripts)
            )

        return EquivariantTensorProduct(
            [f(d) for d in self.ds],
            [
                Operand(ope.irreps, layout)
                for ope, layout in zip(self.operands, layouts)
            ],
        )

    def flop_cost(self, batch_size: int) -> int:
        """Compute the number of flops of the tensor product."""
        return sum(d.flop_cost(-1) for d in self.ds) * batch_size

    def memory_cost(
        self, batch_sizes: tuple[int, ...], itemsize: Union[int, tuple[int, ...]]
    ) -> int:
        """Compute the number of memory accesses of the tensor product."""
        assert len(batch_sizes) == self.num_operands
        if isinstance(itemsize, int):
            itemsize = (itemsize,) * self.num_operands
        return sum(
            bs * operand.irreps.dim * iz
            for iz, bs, operand in zip(itemsize, batch_sizes, self.operands)
        )

    def backward(self, input: int) -> tuple[EquivariantTensorProduct, tuple[int, ...]]:
        """
        The backward pass of the equivariant tensor product.

        Args:
            input: The input with respect to which the backward pass is computed.

        Returns:
            A tuple containing the ETP representing the backward pass and the permutation of the operands between the original and the backward ETP.

        Examples:
            >>> e = cue.descriptors.fully_connected_tensor_product(
            ...     cue.Irreps("SO3", "4x0+1x1"),
            ...     cue.Irreps("SO3", "4x0+2x1"),
            ...     cue.Irreps("SO3", "4x0+3x1")
            ... )
            >>> e
            EquivariantTensorProduct(114x0 x 4x0+1 x 4x0+2x1 -> 4x0+3x1)
            >>> e_bwd, i = e.backward(0)
            >>> e_bwd
            EquivariantTensorProduct(4x0+3x1 x 4x0+1 x 4x0+2x1 -> 114x0)
            >>> i
            (3, 1, 2, 0)

            >>> e = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2, 3])
            >>> e
            EquivariantTensorProduct((1)^(0..3) -> 0+1+2+3)
            >>> e_bwd, i = e.backward(0)
            >>> e_bwd
            EquivariantTensorProduct(0+1+2+3 x (1)^(0..2) -> 1)
            >>> i
            (1, 0, 0)
        """
        assert input < self.num_inputs

        remove_input = (input < self.num_inputs - 1) or all(
            d.num_operands <= self.num_operands for d in self.ds
        )

        ds = []
        for d in self.ds:
            d = d.move_operand_first(-1)

            if self.num_operands == d.num_operands:
                ds.append(d.move_operand_last(input + 1))
            elif self.num_operands > d.num_operands:
                if input < d.num_operands - 1:
                    ds.append(d.move_operand_last(input + 1))
                else:
                    continue
            else:
                if input < self.num_inputs - 1:
                    ds.append(d.move_operand_last(input + 1))
                else:
                    ds.append(
                        d.move_operand_last(input + 1)
                        * (d.num_operands - self.num_inputs)
                    )

        ii = list(range(self.num_inputs))
        if remove_input:
            oids = [self.num_operands - 1] + ii[:input] + ii[input + 1 :] + [ii[input]]
        else:
            oids = [self.num_operands - 1] + ii + [ii[input]]

        e = EquivariantTensorProduct(ds, tuple(self.operands[i] for i in oids))
        return e, tuple(oids)

    def stp_operand(self, oid: int) -> Optional[stp.Operand]:
        # output
        if oid == self.num_operands - 1:
            return self.ds[0].operands[-1]
        # input
        for d in self.ds:
            if oid < d.num_operands:
                return d.operands[oid]
        return None

    @classmethod
    def stack(
        cls, es: Sequence[EquivariantTensorProduct], stacked: list[bool]
    ) -> EquivariantTensorProduct:
        """Stack multiple equivariant tensor products."""
        assert len(es) > 0
        num_operands = es[0].num_operands
        layouts = [ope.layout for ope in es[0].operands]
        assert all(e.num_operands == num_operands for e in es)
        assert all(
            e.operands[oid].layout == layouts[oid]
            for e in es
            for oid in range(num_operands)
        )
        assert len(stacked) == num_operands

        new_operands = []
        for oid in range(num_operands):
            if stacked[oid]:
                new_operands.append(
                    Operand(
                        irreps=cue.concatenate([e.operands[oid].irreps for e in es]),
                        layout=layouts[oid],
                    )
                )
            else:
                ope = es[0].operands[oid]
                assert all(e.operands[oid].irreps == ope.irreps for e in es)
                new_operands.append(ope)

        new_ds: dict[int, stp.SegmentedTensorProduct] = {}
        for eid, e in enumerate(es):
            for d in e.ds:
                d = copy.deepcopy(d)
                ii = e._map_operands_from_stp_to_etp(d)
                for oid in range(d.num_operands):
                    if stacked[ii[oid]]:
                        for e_ in reversed(es[:eid]):
                            d.insert_segments(oid, 0, e_.stp_operand(ii[oid]))
                        for e_ in es[eid + 1 :]:
                            d.insert_segments(oid, -1, e_.stp_operand(ii[oid]))

                if d.num_operands not in new_ds:
                    new_ds[d.num_operands] = d
                else:
                    for p in d.paths:
                        new_ds[d.num_operands].add_path(*p.indices, c=p.coefficients)

        return cls(new_ds.values(), new_operands)

    def symmetrize_operands(self) -> EquivariantTensorProduct:
        """Symmetrize the operands of the ETP."""
        new_ds = []
        for d in self.ds:
            new_ds.append(
                d.symmetrize_operands(range(self.num_inputs - 1, d.num_operands - 1))
            )
        return EquivariantTensorProduct(new_ds, self.operands)

    def sort_indices_for_identical_operands(self) -> EquivariantTensorProduct:
        """Sort the indices for identical operands."""
        new_ds = []
        for d in self.ds:
            new_ds.append(
                d.sort_indices_for_identical_operands(
                    range(self.num_inputs - 1, d.num_operands - 1)
                )
            )
        return EquivariantTensorProduct(new_ds, self.operands)
