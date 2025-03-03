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
from typing import Optional, Sequence, Union

from cuequivariance import segmented_tensor_product as stp

from .segmented_tensor_product.dimensions_dict import format_dimensions_dict


@dataclasses.dataclass(init=False, frozen=True)
class SegmentedOperand:
    """A tensor product operand. It is a list of segments and subscripts."""

    subscripts: stp.Subscripts
    segments: tuple[tuple[int, ...]]
    _dims: dict[str, set[int]]

    def __init__(
        self,
        *,
        subscripts: stp.Subscripts,
        segments: Optional[list[tuple[int, ...]]] = None,
        _dims: Optional[dict[str, set[int]]] = None,
    ):
        object.__setattr__(self, "subscripts", stp.Subscripts(subscripts))

        if segments is None:
            segments = []
        object.__setattr__(self, "segments", tuple(segments))

        if _dims is None:
            _dims = dict()
            for segment in self.segments:
                for m, d in zip(self.subscripts, segment):
                    _dims.setdefault(m, set()).add(d)
        object.__setattr__(self, "_dims", _dims)

    @classmethod
    def empty_segments(cls, num_segments: int) -> SegmentedOperand:
        """Create an operand with empty subscripts"""
        return cls(subscripts="", segments=[()] * num_segments, _dims=dict())

    def assert_valid(self):
        """Assert that the operand is valid."""
        if not all(m.isalpha() and m.islower() for m in self.subscripts):
            raise ValueError(f"subscripts {self.subscripts} is not valid.")

        for segment in self.segments:
            if not all(isinstance(dim, int) and dim > 0 for dim in segment):
                raise ValueError(f"segment {segment} is not valid.")

            if len(segment) != len(self.subscripts):
                raise ValueError(
                    f"segment {segment} has {len(segment)} dimensions, expected {len(self.subscripts)} for subscripts {self.subscripts}."
                )

            for m, d in zip(self.subscripts, segment):
                if d not in self.get_dims(m):
                    raise ValueError(
                        f"dimension {d} not in {m} dimensions {self.get_dims(m)}."
                    )

    def insert_segment(
        self, index: int, segment: Union[tuple[int, ...], dict[str, int]]
    ):
        """Insert a segment at a given index."""
        if isinstance(segment, dict):
            segment = tuple(segment[m] for m in self.subscripts)

        if len(segment) != len(self.subscripts):
            raise ValueError(
                f"segment has {len(segment)} dimensions, expected {len(self.subscripts)} for subscripts {self.subscripts}."
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

        for m, d in zip(self.subscripts, segment):
            self._dims.setdefault(m, set()).add(d)

    def add_segment(self, segment: Union[tuple[int, ...], dict[str, int]]) -> int:
        """Add a segment to the operand."""
        self.insert_segment(len(self.segments), segment)
        return len(self.segments) - 1

    def __hash__(self) -> int:
        return hash((self.segments, self.subscripts))

    def __eq__(self, other: SegmentedOperand) -> bool:
        assert isinstance(other, SegmentedOperand)
        return self.subscripts == other.subscripts and self.segments == other.segments

    def __lt__(self, other: SegmentedOperand) -> bool:
        assert isinstance(other, SegmentedOperand)
        return (self.subscripts, self.segments) < (other.subscripts, other.segments)

    def __repr__(self) -> str:
        dims = format_dimensions_dict(self.get_dimensions_dict())
        return f"Operand(subscripts={self.subscripts} num_segments={self.num_segments} {dims})"

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

    @property
    def ndim(self) -> int:
        """The number of segment dimensions."""
        return len(self.subscripts)

    def segment_slices(self) -> list[slice]:
        """Return slice object for each segment."""
        offset = 0
        slices = []
        for segment in self.segments:
            slices.append(slice(offset, offset + math.prod(segment)))
            offset += math.prod(segment)
        return slices

    def get_dimensions_dict(self) -> dict[str, set[int]]:
        """Return a dictionary of dimensions for each channel."""
        return self._dims.copy()

    def get_dims(self, m: str) -> set[int]:
        """Return the dimensions for a given channel."""
        return self._dims.get(m, set()).copy()

    def transpose_modes(
        self, subscripts: Union[str, Sequence[str], Sequence[int]]
    ) -> SegmentedOperand:
        """Transpose the channels of the operand."""
        if not isinstance(subscripts, Sequence):
            raise TypeError("channels must be a sequence.")

        if isinstance(subscripts[0], str):
            subscripts = "".join(subscripts)
            subscripts = stp.Subscripts.complete_wildcards(subscripts, self.subscripts)
            subscripts = [self.subscripts.index(m) for m in subscripts]

        subscripts: list[int] = list(subscripts)

        if len(subscripts) != len(self.subscripts):
            raise ValueError(
                f"channels has {len(subscripts)} dimensions, expected {len(self.subscripts)} for subscripts {self.subscripts}."
            )

        segments = [tuple(segment[i] for i in subscripts) for segment in self.segments]
        return SegmentedOperand(
            subscripts="".join(self.subscripts[i] for i in subscripts),
            segments=segments,
            _dims=self._dims,
        )

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
        if self.subscripts != other.subscripts:
            raise ValueError("subscripts do not match.")
        return SegmentedOperand(
            subscripts=self.subscripts,
            segments=self.segments + other.segments,
            _dims={m: self.get_dims(m) | other.get_dims(m) for m in self.subscripts},
        )
