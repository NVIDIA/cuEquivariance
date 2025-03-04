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

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

import cuequivariance as cue
from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep


# This class is inspired by https://github.com/e3nn/e3nn-jax/blob/245e17eb23deaccad9f2c9cfd40fe40515e3c074/e3nn_jax/_src/irreps_array.py
@dataclass(frozen=True, init=False)
class NumpyIrrepsArray:
    r"""
    Data Array transforming according to the irreducible representations of a Lie group.

    Args:
        irreps (Irreps or str): Irreps of the data array.
        array (np.ndarray): Data array.
        layout (IrrepsLayout, optional): Memory layout of the data array.
    """

    irreps: cue.Irreps
    array: np.ndarray
    layout: cue.IrrepsLayout

    def __init__(
        self,
        irreps: Union[cue.Irreps, str],
        array: np.ndarray,
        layout: Optional[cue.IrrepsLayout] = None,
    ):
        irreps = cue.Irreps(irreps)
        layout = cue.IrrepsLayout.as_layout(layout)

        if not hasattr(array, "shape"):
            raise TypeError(f"Expected array with shape, got {type(array)}")

        if array.shape[-1] != irreps.dim:
            raise ValueError(
                f"Last dimension of array should be {irreps.dim}, got {array.shape[-1]}"
            )

        object.__setattr__(self, "irreps", irreps)
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "layout", layout)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the data array."""
        return self.array.shape

    @property
    def dtype(self):
        """Data type of the data array."""
        return self.array.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions of the data array."""
        return self.array.ndim

    def __len__(self) -> int:
        return len(self.array)

    def __repr__(self) -> str:
        r = str(self.array)
        if "\n" in r:
            return f"{self.irreps}\n{r}"
        return f"{self.irreps} {r}"

    @property
    def segments(self) -> List[np.ndarray]:
        """List of segments of the data array."""
        leading_shape = self.array.shape[:-1]
        return [
            np.reshape(self.array[..., i], leading_shape + self.layout.shape(mul_ir))
            for i, mul_ir in zip(self.irreps.slices(), self.irreps)
        ]

    def __eq__(self: NumpyIrrepsArray, other: NumpyIrrepsArray) -> bool:
        if not isinstance(other, NumpyIrrepsArray):
            return NotImplemented
        return (
            self.irreps == other.irreps
            and np.array_equal(self.array, other.array)
            and self.layout == other.layout
        )

    def __neg__(self: NumpyIrrepsArray) -> NumpyIrrepsArray:
        return NumpyIrrepsArray(self.irreps, -self.array, self.layout)

    def reshape(self, shape: Tuple[int, ...]) -> NumpyIrrepsArray:
        if not (shape[-1] == -1 or shape[-1] == self.irreps.dim):
            raise ValueError(
                f"Last dimension of shape should be -1 or {self.irreps.dim}, got {shape[-1]}"
            )
        return NumpyIrrepsArray(self.irreps, self.array.reshape(shape), self.layout)

    def simplify(self) -> NumpyIrrepsArray:
        if self.layout == cue.mul_ir:
            return NumpyIrrepsArray(self.irreps.simplify(), self.array, self.layout)

        return self.change_layout(cue.mul_ir).simplify().change_layout(self.layout)

    def merge_consecutive(self) -> NumpyIrrepsArray:
        return NumpyIrrepsArray(
            self.irreps.merge_consecutive(), self.array, self.layout
        )

    def sort(self) -> NumpyIrrepsArray:
        s = self.irreps.sort()
        # s.irreps, s.perm, s.inv
        segments = self.segments
        sorted_segments = [segments[i] for i in s.inv]
        return from_segments(s.irreps, sorted_segments, self.layout, self.shape[:-1])

    def regroup(self) -> NumpyIrrepsArray:
        return self.sort().simplify()

    def filter(
        self,
        *,
        keep: Union[str, Sequence[cue.Irrep], Callable[[cue.MulIrrep], bool]] = None,
        drop: Union[str, Sequence[cue.Irrep], Callable[[cue.MulIrrep], bool]] = None,
    ) -> NumpyIrrepsArray:
        if keep is not None:
            if drop is not None:
                raise ValueError("Only one of `keep` or `drop` must be defined.")
            else:
                return self._filter_keep(keep)
        else:
            if drop is not None:
                return self._filter_drop(drop)
            else:
                return self

    def _filter_keep(
        self, keep: Union[str, Sequence[cue.Irrep], Callable[[cue.MulIrrep], bool]]
    ):
        if callable(keep):
            return from_segments(
                self.irreps.filter(keep=keep),
                [
                    segment
                    for segment, mulrep in zip(self.segments, self.irreps)
                    if keep(mulrep)
                ],
                self.layout,
                self.shape[:-1],
            )

        keep = into_list_of_irrep(self.irreps.irrep_class, keep)

        if not all(isinstance(rep, cue.Irrep) for rep in keep):
            raise ValueError(f"Invalid `keep` {keep}, expected a list of Irrep")

        return from_segments(
            self.irreps.filter(keep=keep),
            [
                segment
                for segment, mulrep in zip(self.segments, self.irreps)
                if mulrep.ir in keep
            ],
            self.layout,
            self.shape[:-1],
        )

    def _filter_drop(
        self, drop: Union[str, Sequence[cue.Irrep], Callable[[cue.MulIrrep], bool]]
    ):
        if callable(drop):
            return from_segments(
                self.irreps.filter(drop=drop),
                [
                    segments
                    for segments, mulrep in zip(self.segments, self.irreps)
                    if not drop(mulrep)
                ],
                self.layout,
                self.shape[:-1],
            )

        drop = into_list_of_irrep(self.irreps.irrep_class, drop)

        if not all(isinstance(rep, cue.Irrep) for rep in drop):
            raise ValueError(f"Invalid `drop` {drop}, expected a list of Irrep")

        return from_segments(
            self.irreps.filter(drop=drop),
            [
                segments
                for segments, mulrep in zip(self.segments, self.irreps)
                if mulrep.ir not in drop
            ],
            self.layout,
            self.shape[:-1],
        )

    def change_layout(self, layout: cue.IrrepsLayout) -> NumpyIrrepsArray:
        if layout == self.layout:
            return self
        segments = [np.moveaxis(x, -1, -2) for x in self.segments]
        return from_segments(self.irreps, segments, layout, self.shape[:-1])


def from_segments(
    irreps: cue.Irreps,
    segments: List[np.ndarray],
    layout: cue.IrrepsLayout,
    leading_shape: Tuple[int, ...],
) -> NumpyIrrepsArray:
    if len(segments) != len(irreps):
        raise ValueError(f"Expected {len(irreps)} segments, got {len(segments)}")

    for segment, mul_ir in zip(segments, irreps):
        if segment.shape[:-2] != leading_shape:
            raise ValueError(
                f"Expected shape[:-2] {leading_shape}, got {segment.shape[:-2]}"
            )
        if segment.shape[-2:] != layout.shape(mul_ir):
            raise ValueError(
                f"Expected shape[-2:] {layout.shape(mul_ir)}, got {segment.shape[-2:]}"
            )

    if len(segments) > 0:
        array = np.concatenate(
            [
                np.reshape(segment, segment.shape[:-2] + (mul * ir.dim,))
                for segment, (mul, ir) in zip(segments, irreps)
            ],
            axis=-1,
        )
    else:
        array = np.empty(leading_shape + (0,))
    return NumpyIrrepsArray(irreps, array, layout)


def concatenate(
    arrays: Union[
        list[cue.Irreps],
        list[Union[cue.IrrepsAndLayout]],
        list[NumpyIrrepsArray],
    ],
) -> Union[cue.Irreps, cue.IrrepsAndLayout, NumpyIrrepsArray]:
    if len(arrays) == 0:
        raise ValueError("Expected at least one input")

    if all(isinstance(array, cue.IrrepsAndLayout) for array in arrays):
        assert len({x.layout for x in arrays}) == 1
        return cue.IrrepsAndLayout(
            concatenate([x.irreps for x in arrays]), arrays[0].layout
        )

    if all(isinstance(array, cue.Irreps) for array in arrays):
        return sum(arrays, cue.Irreps(arrays[0].irrep_class, []))

    if not all(isinstance(array, NumpyIrrepsArray) for array in arrays):
        raise TypeError("Expected a list of IrrepsArray")

    if not all(
        array.irreps.irrep_class == arrays[0].irreps.irrep_class for array in arrays
    ):
        raise ValueError("Expected all arrays to have the same irrep class")

    if not all(array.layout == arrays[0].layout for array in arrays):
        raise ValueError("Expected all arrays to have the same layout")

    irreps = cue.Irreps(
        arrays[0].irreps.irrep_class,
        sum([[mul_ir for mul_ir in array.irreps] for array in arrays], []),
    )

    return NumpyIrrepsArray(
        irreps,
        np.concatenate([array.array for array in arrays], axis=-1),
        arrays[0].layout,
    )
