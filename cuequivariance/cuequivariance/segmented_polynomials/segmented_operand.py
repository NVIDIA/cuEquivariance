# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math

from cuequivariance.segmented_polynomials.dimensions_dict import format_dimensions_dict


@dataclasses.dataclass(init=False, frozen=True)
class SegmentedOperand:
    """A segmented operand is a list of segment's shapes."""

    ndim: int
    segments: tuple[tuple[int, ...]]
    _dims: dict[int, set[int]]

    def __init__(
        self,
        segments: list[tuple[int, ...]] | None = None,
        *,
        ndim: int | None = None,
        _dims: dict[int, set[int]] | None = None,
    ):
        if segments is None:
            segments = []
        object.__setattr__(self, "segments", tuple(segments))

        if ndim is None:
            assert len(self.segments) > 0
            ndim = len(self.segments[0])
        object.__setattr__(self, "ndim", ndim)

        if _dims is None:
            _dims = dict()
            for segment in self.segments:
                for i, d in enumerate(segment):
                    _dims.setdefault(i, set()).add(d)
        else:
            _dims = _dims.copy()
        object.__setattr__(self, "_dims", _dims)

    @classmethod
    def empty_segments(cls, num_segments: int) -> SegmentedOperand:
        """Create an operand with ndim=0"""
        return cls(ndim=0, segments=[()] * num_segments, _dims=dict())

    @classmethod
    def stack(cls, operands: list[SegmentedOperand]) -> SegmentedOperand:
        """Stack a list of operands together."""
        assert len(operands) > 0
        ndim = operands[0].ndim
        assert all(ope.ndim == ndim for ope in operands)

        _dims = dict()
        for ope in operands:
            for i, d in ope.get_dimensions_dict().items():
                _dims.setdefault(i, set()).update(d)

        return cls(
            ndim=ndim,
            segments=sum([list(ope.segments) for ope in operands], []),
            _dims=_dims,
        )

    def copy(self) -> SegmentedOperand:
        """Copy the operand."""
        return SegmentedOperand(
            ndim=self.ndim,
            segments=self.segments,
            _dims=self._dims,
        )

    def assert_valid(self):
        """Assert that the operand is valid."""
        for segment in self.segments:
            if len(segment) != self.ndim:
                raise ValueError(
                    f"segment {segment} has {len(segment)} dimensions, expected {self.ndim}."
                )

            if not all(isinstance(dim, int) and dim > 0 for dim in segment):
                raise ValueError(f"segment {segment} is not valid.")

            for i, d in enumerate(segment):
                if d not in self.get_dims(i):
                    raise ValueError(
                        f"dimension {d} not in {i} dimensions {self.get_dims(i)}."
                    )

    def insert_segment(self, index: int, segment: tuple[int, ...]):
        """Insert a segment at a given index."""
        if len(segment) != self.ndim:
            raise ValueError(
                f"segment has {len(segment)} dimensions, expected {self.ndim}."
            )

        if index < 0:
            index = len(self.segments) + index

        if index < 0 or index > len(self.segments):
            raise ValueError(
                f"index {index} is out of bounds for segments {self.segments}."
            )

        segment = tuple(int(d) for d in segment)
        object.__setattr__(
            self,
            "segments",
            self.segments[:index] + (segment,) + self.segments[index:],
        )

        for i, d in enumerate(segment):
            self._dims.setdefault(i, set()).add(d)

    def add_segment(self, segment: tuple[int, ...]) -> int:
        """Add a segment to the operand."""
        self.insert_segment(len(self.segments), segment)
        return len(self.segments) - 1

    def __hash__(self) -> int:
        return hash((self.ndim, self.segments))

    def __eq__(self, other: SegmentedOperand) -> bool:
        assert isinstance(other, SegmentedOperand)
        return self.ndim == other.ndim and self.segments == other.segments

    def __lt__(self, other: SegmentedOperand) -> bool:
        assert isinstance(other, SegmentedOperand)
        return (self.ndim, self.segments) < (other.ndim, other.segments)

    def __repr__(self) -> str:
        dims = format_dimensions_dict(self.get_dimensions_dict())
        return f"Operand(ndim={self.ndim} num_segments={self.num_segments} dims={dims})"

    def __getitem__(self, index: int) -> tuple[int, ...]:
        return self.segments[index]

    def __len__(self) -> int:
        return self.num_segments

    def __iter__(self):
        return iter(self.segments)

    @property
    def num_segments(self) -> int:
        """The number of segments in the operand."""
        return len(self.segments)

    @property
    def size(self) -> int:
        """The total size of the operand."""
        if self.all_same_segment_shape():
            return self.num_segments * self.segment_size

        return sum(math.prod(segment) for segment in self.segments)

    def segment_slices(self) -> list[slice]:
        """Return slice object for each segment."""
        offset = 0
        slices = []
        for segment in self.segments:
            slices.append(slice(offset, offset + math.prod(segment)))
            offset += math.prod(segment)
        return slices

    def get_dimensions_dict(self) -> dict[int, set[int]]:
        """Return a dictionary of dimensions for each channel."""
        return self._dims.copy()

    def get_dims(self, i: int) -> set[int]:
        """Return the dimensions for a given channel."""
        return self._dims.get(i, set()).copy()

    def all_same_segment_shape(self) -> bool:
        """Check if all segments have the same shape. Returns False if there are no segments."""
        return all(len(dd) == 1 for dd in self._dims.values()) and self.num_segments > 0

    @property
    def segment_shape(self) -> tuple[int, ...]:
        """The shape of the segments if they are all the same."""
        if not self.all_same_segment_shape():
            raise ValueError("Segments do not have the same shape.")
        return self.segments[0]

    @property
    def segment_size(self) -> int:
        """The size of the segments if they are all the same."""
        if not self.all_same_segment_shape():
            raise ValueError("Segments do not have the same shape.")
        return math.prod(self.segments[0])

    def __add__(self, other: SegmentedOperand) -> SegmentedOperand:
        if self.ndim != other.ndim:
            raise ValueError("ndim do not match.")
        return SegmentedOperand(
            ndim=self.ndim,
            segments=self.segments + other.segments,
            _dims={i: self.get_dims(i) | other.get_dims(i) for i in range(self.ndim)},
        )

    @property
    def slice_by_segment(self) -> _SegmentSlicer:
        """Return a slicer that allows slicing by segment index."""
        return _SegmentSlicer(self)

    @property
    def slice_by_size(self) -> _SizeSlicer:
        """Return a slicer that allows slicing by flat size/offset."""
        return _SizeSlicer(self)


class _SegmentSlicer:
    """Helper class for slicing by segment index."""

    def __init__(self, operand: SegmentedOperand):
        self.operand = operand

    def __getitem__(self, key: slice) -> SegmentedOperand:
        if not isinstance(key, slice):
            raise TypeError(f"Only slice objects are supported, got {type(key)}")
        return SegmentedOperand(
            ndim=self.operand.ndim, segments=self.operand.segments[key]
        )


class _SizeSlicer:
    """Helper class for slicing by size/offset."""

    def __init__(self, operand: SegmentedOperand):
        self.operand = operand

    def __getitem__(self, key: slice) -> SegmentedOperand:
        if not isinstance(key, slice):
            raise TypeError(f"Only slice objects are supported, got {type(key)}")

        # Handle slice
        start, stop, step = key.indices(self.operand.size)

        if step != 1:
            raise ValueError("Step sizes other than 1 are not supported")

        # Find segments that overlap with [start, stop)
        segments = []
        offset = 0

        for segment in self.operand.segments:
            segment_size = math.prod(segment)
            segment_start = offset
            segment_end = offset + segment_size

            # Check if this segment overlaps with [start, stop)
            if segment_start < stop and segment_end > start:
                segments.append(segment)

            offset += segment_size

            # If we've passed the stop point, we can break
            if offset >= stop:
                break

        return SegmentedOperand(ndim=self.operand.ndim, segments=segments)
