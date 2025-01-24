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

import base64
import collections
import copy
import dataclasses
import functools
import itertools
import json
import logging
import math
import re
import zlib
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import opt_einsum

import cuequivariance as cue  # noqa: F401
from cuequivariance import segmented_tensor_product as stp
from cuequivariance.misc.linalg import round_to_rational, round_to_sqrt_rational

from .dimensions_dict import format_dimensions_dict

logger = logging.getLogger(__name__)


@dataclasses.dataclass(init=False, frozen=False)
class SegmentedTensorProduct:
    """
    Irreps-agnostic and dataflow-agnostic descriptor of a segmented tensor product

    Args:
        operands (list of operands): The operands of the tensor product. To each operand corresponds subscripts and a list of segments.
        paths (list of paths): Each path contains coefficients and a list of indices.
            The indices are the indices of the segments of the operands.
        coefficient_subscripts (str): The subscripts of the coefficients.


    We typically use the :func:`from_subscripts <cuequivariance.SegmentedTensorProduct.from_subscripts>` method to create a descriptor and then add segments and paths one by one.

    .. rubric:: Methods
    """

    operands: list[stp.Operand]
    paths: list[stp.Path]
    coefficient_subscripts: str

    ################################ Initializers ################################

    def __init__(
        self,
        *,
        operands: Optional[list[stp.Operand]] = None,
        paths: Optional[list[stp.Path]] = None,
        coefficient_subscripts: str = "",
    ):
        if operands is None:
            operands = []
        self.operands = operands

        if paths is None:
            paths = []
        self.paths = paths
        self.coefficient_subscripts = coefficient_subscripts

    def assert_valid(self):
        assert stp.Subscripts.is_valid(self.subscripts)

        for m in self.subscripts.modes():
            if self.subscripts.count(m) == 1:
                raise ValueError(
                    f"mode {m} is not contracted in {self.subscripts}. It should appear at least twice."
                )

        for operand in self.operands:
            operand.assert_valid()  # check subscripts and segment lengths

        for path in self.paths:
            path.assert_valid()

            if path.coefficients.ndim != len(self.coefficient_subscripts):
                raise ValueError(  # pragma: no cover
                    f"expected {len(self.coefficient_subscripts)} dimensions for the coefficients, got {path.coefficients.ndim}."
                )

            for oid in range(self.num_operands):
                if path.indices[oid] >= self.operands[oid].num_segments:
                    raise ValueError(  # pragma: no cover
                        f"path has an index {path.indices[oid]} out of bounds for operand {oid}."
                        f" Operand {oid} has only {self.operands[oid].num_segments} segments."
                    )

            dims = self.get_path_dimensions_dict(path, returns_sets=True)
            if any(len(dd) != 1 for dd in dims.values()):
                raise ValueError(  # pragma: no cover
                    f"path has ambiguous dimensions {dims}."
                )

    @classmethod
    def from_subscripts(cls, subscripts: stp.Subscripts) -> SegmentedTensorProduct:
        r"""
        Create a descriptor from a subscripts string.

        Examples:
            >>> d = cue.SegmentedTensorProduct.from_subscripts("uv,ui,vj+ij")
            >>> i0 = d.add_segment(0, (2, 3))
            >>> i1 = d.add_segment(1, (2, 5))
            >>> i2 = d.add_segment(2, (3, 4))
            >>> d.add_path(i0, i1, i2, c=np.ones((5, 4)))
            0
            >>> print(d)
            uv,ui,vj+ij operands=[(2, 3)],[(2, 5)],[(3, 4)] paths=[op0[0]*op1[0]*op2[0]*c c.shape=(5, 4) c.nnz=20]
        """
        subscripts = stp.Subscripts(subscripts)
        operands = [stp.Operand(subscripts=operand) for operand in subscripts.operands]

        return cls(
            operands=operands, paths=[], coefficient_subscripts=subscripts.coefficients
        )

    @classmethod
    def empty_segments(cls, num_segments: list[int]) -> SegmentedTensorProduct:
        r"""
        Create a descriptor with a simple structure.

        Examples:
            >>> cue.SegmentedTensorProduct.empty_segments([2, 3, 4])
            ,, sizes=2,3,4 num_segments=2,3,4 num_paths=0
        """
        return cls(
            operands=[stp.Operand.empty_segments(num) for num in num_segments],
            paths=[],
            coefficient_subscripts="",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentedTensorProduct:
        r"""Create a descriptor from a dictionary."""
        d = SegmentedTensorProduct.from_subscripts(data["subscripts"])
        for oid, operand in enumerate(data["operands"]):
            d.add_segments(oid, operand["segments"])
        for indices, coefficients in zip(
            data["paths"]["indices"], data["paths"]["coefficients"]
        ):
            d.add_path(*indices, c=np.array(coefficients))
        return d

    @classmethod
    def from_json(cls, data: str) -> SegmentedTensorProduct:
        r"""Create a descriptor from a JSON string."""
        return cls.from_dict(json.loads(data))

    @classmethod
    def from_bytes(cls, data: bytes) -> SegmentedTensorProduct:
        r"""Create a descriptor from compressed binary data."""
        return cls.from_json(zlib.decompress(data).decode("ascii"))

    @classmethod
    def from_base64(cls, data: str) -> SegmentedTensorProduct:
        r"""Create a descriptor from a base64 string."""
        return cls.from_bytes(base64.b64decode(data))

    ################################ Properties ################################

    def __hash__(self) -> int:
        return hash(
            (tuple(self.operands), tuple(self.paths), self.coefficient_subscripts)
        )

    def __repr__(self) -> str:
        if max(len(operand) for operand in self.operands) == 1 and len(self.paths) == 1:
            operands = ",".join(
                "[" + ",".join(map(str, operand.segments)) + "]"
                for operand in self.operands
            )
            return f"{self.subscripts} operands={operands} paths={self.paths}"
        sizes = ",".join(f"{operand.size}" for operand in self.operands)
        num_segments = ",".join(f"{len(operand)}" for operand in self.operands)
        output = f"{self.subscripts} sizes={sizes} num_segments={num_segments} num_paths={self.num_paths}"
        dims = format_dimensions_dict(self.get_dimensions_dict())
        if dims:
            output += f" {dims}"
        return output

    @property
    def num_operands(self) -> int:
        """Number of operands."""
        return len(self.operands)

    @property
    def num_paths(self) -> int:
        """Number of paths."""
        return len(self.paths)

    @property
    def subscripts(self) -> stp.Subscripts:
        """Subscripts of the tensor product."""
        return stp.Subscripts.from_operands(
            [operand.subscripts for operand in self.operands],
            self.coefficient_subscripts,
        )

    @property
    def indices(self) -> np.ndarray:
        """Indices of the paths."""
        if len(self.paths) == 0:
            return np.empty((0, self.num_operands), dtype=int)
        return np.array([path.indices for path in self.paths], dtype=int)

    @property
    def coefficients_are_stackable(self) -> bool:
        """Check if the coefficients are stackable."""
        if len({path.coefficients.shape for path in self.paths}) == 1:
            return True
        dims = self.get_dimensions_dict()
        if all(
            len(dd) == 1 for ch, dd in dims.items() if ch in self.coefficient_subscripts
        ):
            return True
        return False

    @property
    def stacked_coefficients(self) -> np.ndarray:
        """Stacked coefficients of the paths in a single array."""
        if not self.coefficients_are_stackable:
            raise ValueError("coefficients are not stackable.")  # pragma: no cover

        cs = [path.coefficients for path in self.paths]
        if len(cs) == 0:
            dims = self.get_dimensions_dict()
            shape = tuple(
                next(iter(dims.get(ch))) for ch in self.coefficient_subscripts
            )
            return np.empty((0,) + shape, dtype=np.float64)
        return np.stack(cs)

    ################################ Getters ################################

    def to_text(self, coefficient_formatter=lambda x: f"{x}") -> str:
        """Human-readable text representation of the descriptor.

        Args:
            coefficient_formatter (callable, optional): A function to format the coefficients.

        Examples:
            >>> d = cue.descriptors.fully_connected_tensor_product(
            ...     cue.Irreps("SO3", "4x0+4x1"),
            ...     cue.Irreps("SO3", "4x0+4x1"),
            ...     cue.Irreps("SO3", "4x0+4x1")
            ... ).d
            >>> d = d.flatten_coefficient_modes()
            >>> print(d.to_text())
            uvw,u,v,w sizes=320,16,16,16 num_segments=5,4,4,4 num_paths=16 u=4 v=4 w=4
            operand #0 subscripts=uvw
              | u: [4] * 5
              | v: [4] * 5
              | w: [4] * 5
            operand #1 subscripts=u
              | u: [4] * 4
            operand #2 subscripts=v
              | v: [4] * 4
            operand #3 subscripts=w
              | w: [4] * 4
            Flop cost: 0->1344 1->2368 2->2368 3->2368
            Memory cost: 368
            Path indices: 0 0 0 0, 1 0 1 1, 1 0 2 2, 1 0 3 3, 2 1 0 1, 2 2 0 2, ...
            Path coefficients: [0.17...]
        """
        out = f"{self}"
        dims = self.get_dimensions_dict()
        for oid, operand in enumerate(self.operands):
            out += f"\noperand #{oid} subscripts={operand.subscripts}"
            for i, ch in enumerate(operand.subscripts):
                if len(dims[ch]) == 1:
                    out += f"\n  | {ch}: [{operand.segments[0][i]}] * {len(operand.segments)}"
                else:
                    out += (
                        f"\n  | {ch}: ["
                        + ", ".join(str(s[i]) for s in operand.segments)
                        + "]"
                    )

        out += f"\nFlop cost: {' '.join(f'{oid}->{self.flop_cost(oid)}' for oid in range(self.num_operands))}"
        out += f"\nMemory cost: {self.memory_cost('global')}"

        if len(self.paths) > 0:
            out += "\nPath indices: " + ", ".join(
                " ".join(str(i) for i in row) for row in self.indices
            )

            if coefficient_formatter is not None:
                formatter = {"float": coefficient_formatter}
                if all(len(dims[ch]) == 1 for ch in self.coefficient_subscripts):
                    out += "\nPath coefficients: " + np.array2string(
                        self.stacked_coefficients,
                        max_line_width=np.inf,
                        formatter=formatter,
                        threshold=np.inf,
                    )
                else:
                    out += "\nPath coefficients:\n" + "\n".join(
                        np.array2string(
                            path.coefficients, formatter=formatter, threshold=np.inf
                        )
                        for path in self.paths
                    )
        else:
            out += "\nNo paths."

        return out

    def to_dict(self, extended: bool = False) -> dict[str, Any]:
        """Dictionary representation of the descriptor."""
        paths = {
            "indices": [path.indices for path in self.paths],
            "coefficients": [path.coefficients.tolist() for path in self.paths],
        }
        if not extended:
            return {
                "subscripts": str(self.subscripts),
                "operands": [{"segments": ope.segments} for ope in self.operands],
                "paths": paths,
            }

        segment_slices = [ope.segment_slices() for ope in self.operands]
        extended_dict = {
            "subscripts": str(self.subscripts),
            "coefficient_subscripts": self.coefficient_subscripts,
            "operands": [
                {
                    "subscripts": ope.subscripts,
                    "segments": ope.segments,
                    "size": ope.size,
                    "segment_offsets": [sl.start for sl in slices],
                    "segment_sizes": [sl.stop - sl.start for sl in slices],
                    "flop_cost": self.flop_cost(oid),
                }
                for oid, ope, slices in zip(
                    range(self.num_operands), self.operands, segment_slices
                )
            ],
            "memory_cost": {
                algorithm: self.memory_cost(algorithm)
                for algorithm in ["sequential", "global"]
            },
            "paths": paths,
        }
        return extended_dict

    def to_json(self, extended: bool = False) -> str:
        """JSON representation of the descriptor."""
        return json.dumps(self.to_dict(extended))

    def to_bytes(self, extended: bool = False) -> bytes:
        """Compressed binary representation of the descriptor."""
        return zlib.compress(self.to_json(extended).encode("ascii"))

    def to_base64(self, extended: bool = False) -> str:
        """
        Base64 representation of the descriptor.

        Examples:
            >>> d = cue.descriptors.fully_connected_tensor_product(
            ...     cue.Irreps("SO3", "4x0+4x1"),
            ...     cue.Irreps("SO3", "4x0+4x1"),
            ...     cue.Irreps("SO3", "4x0+4x1")
            ... ).d
            >>> print(d.to_base64())
            eJytkstuwjAQRX/F8r...lTF2zlX91/fHyvj2Z4=
        """
        return base64.b64encode(self.to_bytes(extended)).decode("ascii")

    def get_dimensions_dict(self) -> dict[str, set[int]]:
        """Get the dimensions of the tensor product."""
        dims: dict[str, set[int]] = {ch: set() for ch in self.subscripts.modes()}
        for operand in self.operands:
            for m, dd in operand.get_dimensions_dict().items():
                dims[m].update(dd)
        # Note: no need to go through the coefficients since must be contracted with the operands
        return dims

    def get_dims(self, m: str) -> set[int]:
        """
        Get the dimensions of a specific mode.

        Examples:
            >>> d = cue.descriptors.fully_connected_tensor_product(
            ...     cue.Irreps("SO3", "4x0+8x1"),
            ...     cue.Irreps("SO3", "3x0+3x1"),
            ...     cue.Irreps("SO3", "5x0+7x1")
            ... ).d
            >>> d.get_dims("u")
            {8, 4}
            >>> d.get_dims("v")
            {3}
            >>> d.get_dims("w")
            {5, 7}
        """
        return self.get_dimensions_dict().get(m, set())

    def get_path_dimensions_dict(
        self, path: Union[int, stp.Path], *, returns_sets: bool = False
    ) -> dict[str, Union[int, set[int]]]:
        """Get the dimensions of a specific path."""
        if isinstance(path, int):
            path = self.paths[path]

        dims = {
            m: {d} for m, d in zip(self.coefficient_subscripts, path.coefficients.shape)
        }
        for oid, sid in enumerate(path.indices):
            for m, d in zip(self.operands[oid].subscripts, self.operands[oid][sid]):
                dims.setdefault(m, set()).add(d)

        if returns_sets:
            return dims
        assert all(len(dd) == 1 for dd in dims.values())
        return {m: next(iter(dd)) for m, dd in dims.items()}

    def get_path_dim(
        self, path: Union[int, stp.Path], m: str, *, returns_set=False
    ) -> Union[int, set[int]]:
        """Get the dimension of a specific mode in a specific path."""
        if isinstance(path, int):
            path = self.paths[path]
        return self.get_path_dimensions_dict(path, returns_sets=returns_set).get(
            m, set() if returns_set else 0
        )

    def segment_slice(self, operand: int, path: Union[int, stp.Path]) -> slice:
        """Get the slice of the segment in the given operand selected by the given path."""
        if isinstance(path, int):
            path = self.paths[path]
        return self.operands[operand].segment_slices()[path.indices[operand]]

    def get_segment_shape(
        self, operand: int, path: Union[int, stp.Path]
    ) -> tuple[int, ...]:
        """Get the shape of the segment in the given operand selected by the given path."""
        if isinstance(path, int):
            path = self.paths[path]
        return self.operands[operand][path.indices[operand]]

    def all_segments_are_used(self) -> bool:
        """Check if all segments are used in the tensor product."""
        for op_idx, operand in enumerate(self.operands):
            for seg_idx in range(operand.num_segments):
                if not any(seg_idx == path.indices[op_idx] for path in self.paths):
                    return False
        return True

    def compressed_path_segment(self, operand: int) -> np.ndarray:
        """
        Starting indices of paths for the segments of the specified operand.

        Note: This method requires that the paths are sorted by the specified operand.

        Args:
            operand (int): The index of the operand for which to find path groups.

        Returns:
            np.ndarray: An array of starting path indices for each segment in the specified operand.

        Examples:

            .. code-block:: python

                indices[:, operand], operands[operand].num_segments -> compressed_path_segment(operand)
                [0, 0, 1, 1, 1], 2 -> [0, 2, 5]
                [0, 0, 1, 1, 1], 3 -> [0, 2, 5, 5]
                [0, 0, 2, 2, 2], 3 -> [0, 2, 2, 5]
                [1, 1], 2 -> [0, 0, 2]
        """
        if not np.all(np.diff(self.indices[:, operand]) >= 0):
            raise ValueError("Paths must be sorted by the specified operand.")

        i = self.indices[:, operand]
        n = self.operands[operand].num_segments

        return np.append(0, np.bincount(i, minlength=n).cumsum())

    @functools.lru_cache(maxsize=None)
    def symmetries(self) -> list[tuple[int, ...]]:
        """List of permutations that leave the tensor product invariant."""

        def functionally_equivalent(
            d1: SegmentedTensorProduct, d2: SegmentedTensorProduct
        ) -> bool:
            d1 = d1.consolidate_paths().sort_paths()
            d2 = d2.consolidate_paths().sort_paths()
            return d1 == d2

        ps = []
        for p in itertools.permutations(range(self.num_operands)):
            if functionally_equivalent(self, self.permute_operands(p)):
                ps.append(p)
        return ps

    def coefficients_equal_one(self) -> bool:
        """Check if all coefficients are equal to one."""
        return np.all(self.stacked_coefficients == 1)

    def flop_cost(self, operand: int, algorithm: str = "optimal") -> int:
        """
        Compute the number of flops needed to compute the specified operand.

        Args:
            operand (int): The operand for which to compute the flop cost.
            algorithm (str, optional): The algorithm to use to compute the cost. Can be 'optimal' or 'naive'.

        Returns:
            int: The number of flops needed to compute the specified operand.
        """
        d = self.move_operand_last(operand)
        subscripts = (
            d.coefficient_subscripts
            + "".join("," + operand.subscripts for operand in d.operands[:-1])
            + "->"
            + d.operands[-1].subscripts
        )

        @functools.lru_cache(maxsize=None)
        def compute_cost(segment_shapes: tuple[tuple[int, ...], ...]) -> int:
            _, info = opt_einsum.contract_path(
                subscripts, *segment_shapes, optimize="optimal", shapes=True
            )
            if algorithm == "naive":
                return int(info.naive_cost)
            elif algorithm == "optimal":
                return int(info.opt_cost)
            else:
                raise ValueError(f"unknown algorithm {algorithm}.")

        cost = 0
        for path in d.paths:
            shapes = tuple(
                d.get_segment_shape(oid, path) for oid in range(d.num_operands - 1)
            )
            dims = d.get_path_dimensions_dict(path)
            coeff_shape = tuple(dims[ch] for ch in d.coefficient_subscripts)
            cost += compute_cost((coeff_shape,) + shapes)
        return cost

    def memory_cost(self, algorithm: str = "sequential") -> int:
        """
        Compute the number of memory accesses needed to compute the specified operand.

        Args:
            operand (int):  The operand for which to compute the memory cost.
            algorithm (str, optional):  The algorithm to use to compute the cost. Can be 'sequential' or 'global'.

        Returns:
            int: The number of memory accesses needed to compute the specified operand.
        """
        if algorithm == "sequential":
            return sum(
                sum(
                    math.prod(self.get_segment_shape(oid, path))
                    for oid in range(self.num_operands)
                )
                for path in self.paths
            )
        elif algorithm == "global":
            return sum(operand.size for operand in self.operands)
        else:
            raise ValueError(f"unknown algorithm {algorithm}.")

    ################################ Modifiers ################################

    def insert_path(
        self,
        path_index,
        *segments: Union[int, tuple[int, ...], dict[str, int], None],
        c: np.ndarray,
        dims: Optional[dict[str, int]] = None,
    ) -> int:
        """Insert a path at a specific index."""
        path_index = _canonicalize_index("path_index", path_index, len(self.paths) + 1)

        if len(segments) != self.num_operands:
            raise ValueError(
                f"expected {self.num_operands} indices, got {len(segments)}."
            )

        coefficients = np.asarray(c)
        del c

        if coefficients.ndim != len(self.coefficient_subscripts):
            raise ValueError(
                f"expected {len(self.coefficient_subscripts)} dimensions for the coefficients, got {coefficients.ndim}."
            )

        if dims is not None:
            dims: dict[str, set[int]] = {
                m: {d} for m, d in dims.items() if m in self.subscripts.modes()
            }
        else:
            dims: dict[str, set[int]] = dict()

        for m, d in zip(self.coefficient_subscripts, coefficients.shape):
            dims.setdefault(m, set()).add(d)

        for oid, s in enumerate(segments):
            subscripts = self.operands[oid].subscripts
            if isinstance(s, int):
                if not (
                    -self.operands[oid].num_segments
                    <= s
                    < self.operands[oid].num_segments
                ):
                    raise ValueError(
                        f"segment index {s} out of bounds for operand {oid}."
                    )
                for m, d in zip(subscripts, self.operands[oid][s]):
                    dims.setdefault(m, set()).add(d)
            elif isinstance(s, tuple):
                for m, d in zip(subscripts, s):
                    dims.setdefault(m, set()).add(d)
            elif isinstance(s, dict):
                for m, d in s.items():
                    dims.setdefault(m, set()).add(d)
            elif s is None:
                pass
            else:
                raise ValueError(
                    f"expected a segment index, a tuple or a dictionary. Got {s}."
                )

        if any(len(dd) != 1 for dd in dims.values()):
            raise ValueError(
                f"Ambiguous dimensions for the path. {dims} try to insert {segments}"
            )
        if len(dims) != len(self.subscripts.modes()):
            raise ValueError(
                f"expected a dimension for each subscripts in the descriptor. Got {dims}."
                f" Missing dimensions for {self.subscripts.modes() - dims.keys()}."
            )

        dims = {m: next(iter(dd)) for m, dd in dims.items()}

        self.paths.insert(
            path_index,
            stp.Path(
                [
                    (
                        (s + self.operands[oid].num_segments)
                        % self.operands[oid].num_segments
                        if isinstance(s, int)
                        else self.add_segment(oid, dims)
                    )
                    for oid, s in enumerate(segments)
                ],
                coefficients,
            ),
        )
        return path_index

    def add_path(
        self,
        *segments: Union[int, tuple[int, ...], dict[str, int], None],
        c: np.ndarray,
        dims: Optional[dict[str, int]] = None,
    ) -> int:
        """
        Add a path to the descriptor.

        Args:
            segments: Specifies the segments of the operands that are contracted in the path.
            c (np.ndarray): The coefficients of the path.
            dims (dict[str, int], optional): The extent of the modes.

        Returns:
            int: The index of the added path.

        Examples:
            >>> d = cue.SegmentedTensorProduct.from_subscripts("uv,ui,vj+ij")
            >>> i1 = d.add_segment(1, (2, 3))
            >>> i2 = d.add_segment(2, (2, 5))

            We can use ``None`` to add a new segment on the fly:
            >>> d.add_path(None, i1, i2, c=np.ones((3, 5)))
            0

            The descriptor has now a new segment ``(2, 2)`` in the first operand:
            >>> d
            uv,ui,vj+ij operands=[(2, 2)],[(2, 3)],[(2, 5)] paths=[op0[0]*op1[0]*op2[0]*c c.shape=(3, 5) c.nnz=15]
            >>> d.add_path(0, None, None, c=np.ones((10, 10)))
            1
            >>> d
            uv,ui,vj+ij sizes=4,26,30 num_segments=1,2,2 num_paths=2 i={3, 10} j={5, 10} u=2 v=2

            When the dimensions of the modes cannot be inferred, we can provide them:
            >>> d.add_path(None, None, None, c=np.ones((2, 2)), dims={"u": 2, "v": 2})
            2
            >>> d
            uv,ui,vj+ij sizes=8,30,34 num_segments=2,3,3 num_paths=3 i={2, 3, 10} j={2, 5, 10} u=2 v=2
        """
        return self.insert_path(len(self.paths), *segments, c=c, dims=dims)

    def insert_segment(
        self,
        operand: int,
        sid: int,
        segment: Union[tuple[int, ...], dict[str, int]],
    ):
        """Insert a segment at a specific index."""
        operand = _canonicalize_index("operand", operand, self.num_operands)
        sid = _canonicalize_index("sid", sid, self.operands[operand].num_segments + 1)

        self.operands[operand].insert_segment(sid, segment)
        self.paths = [
            stp.Path(
                [
                    s if s < sid or oid != operand else s + 1
                    for oid, s in enumerate(path.indices)
                ],
                path.coefficients,
            )
            for path in self.paths
        ]

    def insert_segments(
        self,
        operand: int,
        sid: int,
        segments: Union[list[tuple[int, ...]], stp.Operand],
    ):
        """Insert segments at a specific index."""
        operand = _canonicalize_index("operand", operand, self.num_operands)
        sid = _canonicalize_index("sid", sid, self.operands[operand].num_segments + 1)

        if not isinstance(segments, stp.Operand):
            segments = stp.Operand(
                subscripts=self.operands[operand].subscripts, segments=segments
            )

        o = self.operands[operand]
        if o.subscripts != segments.subscripts:
            raise ValueError(
                f"expected segments subscripts {segments.subscripts} to match operand subscripts {o.subscripts}."
            )

        self.operands[operand] = stp.Operand(
            subscripts=o.subscripts,
            segments=o.segments[:sid] + segments.segments + o.segments[sid:],
            _dims={m: o.get_dims(m) | segments.get_dims(m) for m in o.subscripts},
        )
        self.paths = [
            stp.Path(
                [
                    s if s < sid or oid != operand else s + segments.num_segments
                    for oid, s in enumerate(path.indices)
                ],
                path.coefficients,
            )
            for path in self.paths
        ]

    def add_segment(
        self, operand: int, segment: Union[tuple[int, ...], dict[str, int]]
    ) -> int:
        """Add a segment to the descriptor."""
        return self.operands[operand].add_segment(segment)

    def add_segments(
        self, operand: int, segments: list[Union[tuple[int, ...], dict[str, int]]]
    ):
        """Add segments to the descriptor."""
        for segment in segments:
            self.add_segment(operand, segment)

    def canonicalize_subscripts(self) -> SegmentedTensorProduct:
        """
        Return a new descriptor with a canonical representation of the subscripts.

        Examples:
            >>> d = cue.SegmentedTensorProduct.from_subscripts("ab,ax,by+xy")
            >>> d.canonicalize_subscripts()
            uv,ui,vj+ij sizes=0,0,0 num_segments=0,0,0 num_paths=0 i= j= u= v=

        This is useful to identify equivalent descriptors.
        """
        subscripts = stp.Subscripts.canonicalize(self.subscripts)
        return self.add_or_rename_modes(subscripts)

    def add_or_rename_modes(
        self, subscripts: str, *, mapping: Optional[dict[str, str]] = None
    ) -> SegmentedTensorProduct:
        r"""
        Return a new descriptor with the modes renamed according to the new subscripts.

        Args:
            subscripts (str): The new subscripts that contains the new names of the modes.
                The new subscripts can also be a superset of the old subscripts.
            mapping (dict of str to str, optional): The mapping between the old and new modes.

        Returns:
            SegmentedTensorProduct: The new descriptor with the renamed modes.
        """
        subscripts = stp.Subscripts(subscripts)

        if subscripts.is_equivalent(self.subscripts):
            d = SegmentedTensorProduct.from_subscripts(subscripts)
            for oid, operand in enumerate(self.operands):
                d.add_segments(oid, operand.segments)
            for path in self.paths:
                d.paths.append(
                    stp.Path(indices=path.indices, coefficients=path.coefficients)
                )
            return d

        # Non trivial setting: the new subscripts might be a superset of the old subscripts
        # In this case, we need to properly map de mode dimensions and put 1s where needed
        # But we need to be careful, because the new subscripts might rename the modes differently
        if mapping is None:
            mappings = self.subscripts.is_subset_of(subscripts)
            if len(mappings) == 0:
                raise ValueError(
                    f"expected new subscripts {subscripts} to contain all the modes of old subscripts {self.subscripts}."
                )
            if len(mappings) > 1:
                raise ValueError(
                    f"unable to determine non-ambiguous mapping between old subscripts {self.subscripts} and new subscripts {subscripts}."
                )
            mapping = mappings[0]

        if not all(ch in mapping for ope in self.operands for ch in ope.subscripts):
            raise ValueError(
                f"expected all segment modes to be in the mapping {mapping}."
            )
        if not all(ch in subscripts for ch in mapping.values()):
            raise ValueError(
                f"expected all mapped modes to be in the new subscripts {subscripts}."
            )

        D = SegmentedTensorProduct.from_subscripts(subscripts)
        for oid in range(self.num_operands):
            for sid in range(self.operands[oid].num_segments):
                dims = collections.defaultdict(lambda: 1)
                for m, d in zip(self.operands[oid].subscripts, self.operands[oid][sid]):
                    dims[mapping[m]] = d
                D.add_segment(oid, dims)

        for path in self.paths:
            dims: dict[str, int] = dict()
            for oid, sid in enumerate(path.indices):
                for m, d in zip(D.operands[oid].subscripts, D.operands[oid][sid]):
                    dims[m] = d

            D.paths.append(
                stp.Path(
                    indices=path.indices,
                    coefficients=np.reshape(
                        path.coefficients,
                        tuple(dims[m] for m in D.coefficient_subscripts),
                    ),
                )
            )

        return D

    def add_or_transpose_modes(
        self, subscripts: str, dims: Optional[dict[str, int]] = None
    ) -> SegmentedTensorProduct:
        r"""
        Return a new descriptor with the modes transposed according to the new subscripts.

        Args:
            subscripts (str): A new subscripts that contains a permutation of the modes of the current subscripts.
            dims (dict of str to int, optional): The dimensions of the new modes.

        Returns:
            SegmentedTensorProduct: The new descriptor with the transposed modes.
        """
        subscripts = stp.Subscripts.complete_wildcards(subscripts, self.subscripts)

        if dims is None:
            dims = dict()

        for old, new in zip(
            self.subscripts.operands_and_coefficients,
            subscripts.operands_and_coefficients,
        ):
            if not set(old).issubset(set(new)):
                raise ValueError(f"expected {old} to be a subset of {new}.")

            for ch in new:
                if ch not in old and ch not in dims:
                    raise ValueError(f"expected dimension for mode {ch}.")

        if len(self.subscripts.coefficients) != len(subscripts.coefficients):
            raise ValueError(
                "impossible to introduce new modes in the coefficients subscripts."
            )

        d = SegmentedTensorProduct.from_subscripts(subscripts)
        for oid in range(self.num_operands):
            old, new = (
                self.operands[oid].subscripts,
                d.operands[oid].subscripts,
            )
            perm = [old.index(ch) if ch in old else ch for ch in new]
            for sid in range(self.operands[oid].num_segments):
                d.add_segment(
                    oid,
                    tuple(
                        self.operands[oid][sid][i] if isinstance(i, int) else dims[i]
                        for i in perm
                    ),
                )
        old, new = self.coefficient_subscripts, d.coefficient_subscripts
        perm = [old.index(ch) for ch in new]
        for path in self.paths:
            d.paths.append(
                stp.Path(
                    indices=path.indices,
                    coefficients=np.transpose(path.coefficients, perm),
                )
            )
        return d

    def append_modes_to_all_operands(
        self, modes: str, dims: dict[str, int]
    ) -> SegmentedTensorProduct:
        r"""
        Return a new descriptor with the modes appended (to the right) to all operands.

        Args:
            modes (str): The new segment modes to append to all operands.
            dims (dict of str to int): The dimensions of the new modes.

        Returns:
            SegmentedTensorProduct: The new descriptor with the appended modes.
        """
        if not all(ch not in self.subscripts for ch in modes):
            raise ValueError(
                f"expected new modes {modes} to be disjoint from the current subscripts {self.subscripts}."
            )
        if not all(ch in dims for ch in modes):
            raise ValueError(f"expected dimensions for all new modes {modes}.")
        subscripts = stp.Subscripts.from_operands(
            [ope + modes for ope in self.subscripts.operands],
            coefficients=self.subscripts.coefficients,
        )
        return self.add_or_transpose_modes(subscripts, dims)

    def permute_operands(self, perm: tuple[int, ...]) -> SegmentedTensorProduct:
        """Permute the operands of the descriptor."""
        assert set(perm) == set(range(self.num_operands))
        return dataclasses.replace(
            self,
            operands=[self.operands[i] for i in perm],
            paths=[path.permute_operands(perm) for path in self.paths],
        )

    def move_operand(self, operand: int, new_index: int) -> SegmentedTensorProduct:
        """Move an operand to a new index."""
        operand = _canonicalize_index("operand", operand, self.num_operands)
        if new_index < 0:
            new_index += self.num_operands
        perm = list(range(self.num_operands))
        del perm[operand]
        perm.insert(new_index, operand)
        return self.permute_operands(perm)

    def move_operand_first(self, operand: int) -> SegmentedTensorProduct:
        """Move an operand to the first position."""
        return self.move_operand(operand, 0)

    def move_operand_last(self, operand: int) -> SegmentedTensorProduct:
        """Move an operand to the last position."""
        return self.move_operand(operand, -1)

    def permute_segments(
        self, operand: int, perm: tuple[int, ...]
    ) -> SegmentedTensorProduct:
        """Permute the segments of an operand."""
        operand = _canonicalize_index("operand", operand, self.num_operands)
        new_operands = self.operands.copy()
        new_operands[operand] = stp.Operand(
            segments=[self.operands[operand][i] for i in perm],
            subscripts=self.operands[operand].subscripts,
        )
        new_paths = [
            stp.Path(
                indices=tuple(
                    sid if oid != operand else perm.index(sid)
                    for oid, sid in enumerate(path.indices)
                ),
                coefficients=path.coefficients,
            )
            for path in self.paths
        ]
        return dataclasses.replace(self, operands=new_operands, paths=new_paths)

    def sort_paths(
        self, operands_ordering: Optional[Union[int, Sequence[int]]] = None
    ) -> SegmentedTensorProduct:
        """
        Sort the paths by their indices in lexicographical order.

        Args:
            operands_ordering (int or sequence of int, optional): The order in which to sort the paths.
                If an integer, sort by that operand.
                If a sequence, sort by the first operand in the sequence and then by the
                second operand if equal, etc.

        Returns:
            SegmentedTensorProduct: The sorted descriptor.
        """
        if operands_ordering is None:
            operands_ordering = range(self.num_operands)
        if isinstance(operands_ordering, int):
            operands_ordering = [operands_ordering]
        return dataclasses.replace(
            self,
            paths=sorted(
                self.paths,
                key=lambda path: tuple(path.indices[i] for i in operands_ordering),
            ),
        )

    def squeeze_modes(self, modes: Optional[str] = None) -> SegmentedTensorProduct:
        """
        Squeeze the descriptor by removing dimensions that are always 1.

        Args:
            modes (str, optional): The modes to squeeze. If None, squeeze all modes that are always 1.

        Returns:
            SegmentedTensorProduct: The squeezed descriptor.
        """
        to_remove = {m for m, dd in self.get_dimensions_dict().items() if dd == {1}}

        if modes is not None:
            modes: set[str] = set(modes)
            if not modes.issubset(to_remove):
                raise ValueError(
                    f"modes {modes} are not present in the dimensions that can be squeezed."
                )
            to_remove = to_remove & modes

        d = SegmentedTensorProduct.from_subscripts(
            "".join(ch for ch in self.subscripts if ch not in to_remove)
        )

        def filter_shape(shape: tuple[int, ...], subscripts: str) -> tuple[int, ...]:
            return tuple(
                dim for dim, ch in zip(shape, subscripts) if ch not in to_remove
            )

        for oid, operand in enumerate(self.operands):
            d.add_segments(
                oid,
                [
                    filter_shape(segment, operand.subscripts)
                    for segment in operand.segments
                ],
            )
        for path in self.paths:
            d.paths.append(
                stp.Path(
                    indices=path.indices,
                    coefficients=np.reshape(
                        path.coefficients,
                        filter_shape(
                            path.coefficients.shape, self.coefficient_subscripts
                        ),
                    ),
                )
            )
        logger.debug(f"Squeezed {self} to {d}")
        return d

    def split_mode(self, mode: str, size: int) -> SegmentedTensorProduct:
        """
        Split a mode into multiple modes of a given size.

        Args:
            mode (str): The mode to split. All its dimensions must be divisible by the size.
            size (int): The size of the new modes.

        Returns:
            SegmentedTensorProduct: The new descriptor.
        """
        if re.match(r"^[a-z]$", mode) is None:
            raise ValueError("expected a single lowercase letter.")

        if mode not in self.subscripts:
            return self

        for oid, operand in enumerate(self.operands):
            if mode in operand.subscripts and not operand.subscripts.startswith(mode):
                raise ValueError(
                    f"mode {mode} is not the first mode in operand {oid} ({operand.subscripts})."
                )

        if not all(dim % size == 0 for dim in self.get_dims(mode)):
            raise ValueError(f"mode {mode} is not divisible by {size}.")

        if (
            mode in self.coefficient_subscripts
            and not self.coefficient_subscripts.startswith(mode)
        ):
            return (
                self.add_or_transpose_modes(
                    stp.Subscripts.from_operands(
                        self.subscripts.operands,
                        mode + self.coefficient_subscripts.replace(mode, ""),
                    )
                )
                .split_mode(mode, size)
                .add_or_transpose_modes(self.subscripts)
            )

        d = SegmentedTensorProduct.from_subscripts(self.subscripts)

        offsets_per_operand = []
        for oid, operand in enumerate(self.operands):
            if mode not in operand.subscripts:
                for segment in operand:
                    d.add_segment(oid, segment)
                offsets_per_operand.append(None)
                continue

            assert operand.subscripts.startswith(mode)

            offsets = []
            for segment in operand:
                offsets.append(d.operands[oid].num_segments)
                for _ in range(segment[0] // size):
                    d.add_segment(oid, (size,) + segment[1:])
            offsets_per_operand.append(offsets)

        for path in self.paths:
            num_subdivisions = self.get_path_dim(path, mode) // size

            for i in range(num_subdivisions):
                indices = []
                for oid in range(self.num_operands):
                    if offsets_per_operand[oid] is None:
                        indices.append(path.indices[oid])
                    else:
                        sid = path.indices[oid]
                        offset = offsets_per_operand[oid][sid]
                        indices.append(offset + i)
                coefficients = path.coefficients
                if self.coefficient_subscripts.startswith(mode):
                    coefficients = np.split(coefficients, num_subdivisions, axis=0)[i]
                d.paths.append(stp.Path(indices=indices, coefficients=coefficients))

        logger.debug(f"Split {mode} in {self}: got {d}")
        return d

    def all_same_segment_shape(self) -> bool:
        """Check if all segments have the same shape."""
        for operand in self.operands:
            if not operand.all_same_segment_shape():
                return False
        return True

    def normalize_paths_for_operand(self, operand: int) -> SegmentedTensorProduct:
        """
        Normalize the paths for an operand.

        Args:
            operand (int): The index of the operand to normalize.

        Assuming that the input operand have unit variance, this method computes the
        variance of each segment in the selected operand and uniformly normalize the
        coefficients of the paths by the square root of the variance of the segment.
        This is useful to ensure that the output has unit variance.
        """
        operand = _canonicalize_index("operand", operand, self.num_operands)
        cum_variance = [0.0] * self.operands[operand].num_segments

        contracted_modes = {
            u
            for i, subscripts in enumerate(self.subscripts.operands_and_coefficients)
            #   ^^^^^^^-- (0, "uvw") (1, "iu") (2, "jv") (3, "kw") (4, "ijk")
            if i != operand  # e.g. discard (3, "kw")
            for u in subscripts  # u v w i u j v i j k
            if u not in self.operands[operand].subscripts  # e.g. discard k and w
        }  # e.g. {u, v, i, j}

        for path in self.paths:
            dims = self.get_path_dimensions_dict(path)
            num = np.prod([dim for ch, dim in dims.items() if ch in contracted_modes])
            cum_variance[path.indices[operand]] += np.mean(path.coefficients**2) * num

        new_paths = []

        for path in self.paths:
            total_variance = cum_variance[path.indices[operand]]
            if total_variance > 0:
                c = path.coefficients / np.sqrt(total_variance)
            else:
                c = 0.0 * path.coefficients
            new_paths.append(stp.Path(indices=path.indices, coefficients=c))

        d = dataclasses.replace(self, paths=new_paths)
        logger.debug(f"Normalized paths for operand {operand} in {self}: got {d}")
        return d

    def remove_zero_paths(self) -> SegmentedTensorProduct:
        """Remove paths with zero coefficients."""
        return dataclasses.replace(
            self,
            paths=[path for path in self.paths if not np.all(path.coefficients == 0)],
        )

    def fuse_paths_with_same_indices(self) -> SegmentedTensorProduct:
        """Fuse paths with the same indices."""
        paths = dict()
        for path in self.paths:
            indices = tuple(path.indices)
            if indices in paths:
                paths[indices] += path.coefficients
            else:
                paths[indices] = path.coefficients

        return dataclasses.replace(
            self,
            paths=[
                stp.Path(indices=indices, coefficients=coefficients)
                for indices, coefficients in paths.items()
            ],
        )

    def consolidate_paths(self) -> SegmentedTensorProduct:
        """Consolidate the paths by merging duplicates and removing zeros."""
        return self.fuse_paths_with_same_indices().remove_zero_paths()

    def sort_indices_for_identical_operands(
        self, operands: Sequence[int]
    ) -> SegmentedTensorProduct:
        """Reduce the number of paths by sorting the indices for identical operands."""
        operands = sorted(set(operands))
        if len(operands) < 2:
            return self

        assert len({self.operands[oid].num_segments for oid in operands}) == 1

        def f(ii: tuple[int, ...]) -> tuple[int, ...]:
            aa = sorted([ii[oid] for oid in operands])
            return tuple(
                [
                    aa[operands.index(oid)] if oid in operands else ii[oid]
                    for oid in range(self.num_operands)
                ]
            )

        return dataclasses.replace(
            self,
            paths=[
                stp.Path(
                    indices=f(path.indices),
                    coefficients=path.coefficients,
                )
                for path in self.paths
            ],
        ).consolidate_paths()

    def symmetrize_operands(self, operands: Sequence[int]) -> SegmentedTensorProduct:
        """Symmetrize the specified operands permuting the indices."""
        operands = sorted(set(operands))
        if len(operands) < 2:
            return self

        permutations = list(itertools.permutations(range(len(operands))))
        d = self.sort_indices_for_identical_operands(operands)

        paths = []
        for path in d.paths:
            indices = path.indices

            for perm in permutations:
                new_indices = list(indices)
                for i, oid in enumerate(operands):
                    new_indices[oid] = indices[operands[perm[i]]]
                paths.append(
                    stp.Path(
                        indices=new_indices,
                        coefficients=path.coefficients / len(permutations),
                    )
                )

        d = dataclasses.replace(d, paths=paths)
        return d.consolidate_paths()

    def remove_empty_segments(self) -> SegmentedTensorProduct:
        """Remove empty segments."""

        def empty(D, oid, sid):
            return any(dim == 0 for dim in D.operands[oid][sid])

        D = self
        for oid in range(D.num_operands):
            perm = list(range(D.operands[oid].num_segments))
            for sid in range(D.operands[oid].num_segments):
                if empty(D, oid, sid):
                    perm.remove(sid)
                    perm.append(sid)
            D = D.permute_segments(oid, perm)

        D.paths = [
            path
            for path in D.paths
            if not any(empty(D, oid, sid) for oid, sid in enumerate(path.indices))
        ]

        for oid in range(D.num_operands):
            D.operands[oid] = stp.Operand(
                subscripts=D.operands[oid].subscripts,
                segments=[
                    segment
                    for sid, segment in enumerate(D.operands[oid].segments)
                    if not empty(D, oid, sid)
                ],
                _dims={
                    m: {d for d in dd if d > 0}
                    for m, dd in D.operands[oid]._dims.items()
                },
            )

        return D

    def flatten_modes(
        self, modes: Sequence[str], *, skip_zeros=True, force=False
    ) -> SegmentedTensorProduct:
        """
        Remove the specified modes by subdividing segments and paths.

        Args:
            modes (Sequence of str): The modes to remove, they must precede the modes to keep in each operand.
            skip_zeros (bool, optional): Whether to skip paths with zero coefficients. Default is True.
            force (bool, optional): Whether to force the flattening by flattening extra necessary modes. Default is False.
        """
        if not all(len(ch) == 1 and ch.islower() for ch in modes):
            raise ValueError("expected lowercase single letter modes.")

        modes = {ch for ch in modes if ch in self.subscripts}
        modes = "".join(sorted(modes))

        if force:
            extra = ""
            for m in modes:
                for ope in self.operands:
                    sm = ope.subscripts
                    if m in sm:
                        extra += sm[: sm.index(m)]
            modes += extra

        modes = "".join(sorted(set(modes)))
        if len(modes) == 0:
            return self

        pattern = re.compile(rf"^([{modes}]*)([^{modes}]*)$")
        new_operands = []
        offsets_per_operand = []
        rm_shape_per_operand = []
        rm_modes_per_operand = []
        for operand in self.operands:
            ma = pattern.match(operand.subscripts)
            if ma is None:
                raise ValueError(
                    f"expected modes {modes} to be at the beginning of the segment subscripts."
                    f" Got {operand.subscripts}."
                )

            rm_modes, new_subscripts = ma.groups()
            rm_modes_per_operand.append(rm_modes)

            n = len(rm_modes)  # number of axes to split

            new_segments = []

            off = 0
            offsets = []
            rm_shapes = []
            for segment in operand:
                offsets.append(off)
                num = math.prod(segment[:n])
                rm_shapes.append(segment[:n])
                new_segments += [segment[n:]] * num
                off += num
            offsets_per_operand.append(offsets)
            rm_shape_per_operand.append(rm_shapes)

            new_operands.append(
                stp.Operand(segments=new_segments, subscripts=new_subscripts)
            )

        def ravel_multi_index(indices: tuple[int, ...], shape: tuple[int, ...]) -> int:
            if len(indices) == 0:
                # case not supported by some older numpy versions
                return 0
            return np.ravel_multi_index(indices, shape)

        def make_new_path(
            old_indices: tuple[int, ...],  # old segment indices (one per operand)
            sub_indices: dict[str, int],
            coefficients: np.ndarray,
        ) -> stp.Path:
            return stp.Path(
                [
                    offsets[sid]
                    + ravel_multi_index(
                        tuple(sub_indices[m] for m in rm_modes),
                        rm_shapes[sid],
                    )
                    for sid, offsets, rm_modes, rm_shapes in zip(
                        old_indices,
                        offsets_per_operand,
                        rm_modes_per_operand,
                        rm_shape_per_operand,
                    )
                ],
                coefficients,
            )

        new_paths = []
        if (
            skip_zeros
            and len(self.coefficient_subscripts) > 0
            and set(self.coefficient_subscripts).issubset(modes)
        ):
            non_coeff_modes = "".join(
                m for m in modes if m not in self.coefficient_subscripts
            )
            for path in self.paths:
                dims = self.get_path_dimensions_dict(path)
                idx = np.nonzero(path.coefficients)
                vals = path.coefficients[idx]

                for non_coeff_indices in np.ndindex(
                    tuple(dims[m] for m in non_coeff_modes)
                ):
                    d2 = dict(zip(non_coeff_modes, non_coeff_indices))
                    for coeff_indices, c in zip(zip(*idx), vals):
                        d1 = dict(zip(self.coefficient_subscripts, coeff_indices))

                        new_paths.append(
                            make_new_path(
                                path.indices,
                                d1 | d2,
                                c,
                            )
                        )
        else:
            for path in self.paths:
                dims = self.get_path_dimensions_dict(path)
                for index in np.ndindex(tuple(dims[m] for m in modes)):
                    c = path.coefficients[
                        tuple(
                            (index[modes.index(ch)] if ch in modes else slice(None))
                            for ch in self.coefficient_subscripts
                        )
                    ]
                    if skip_zeros and np.all(c == 0):
                        continue

                    new_paths.append(
                        make_new_path(path.indices, dict(zip(modes, index)), c)
                    )
        d = dataclasses.replace(
            self,
            operands=new_operands,
            paths=new_paths,
            coefficient_subscripts="".join(
                ch for ch in self.coefficient_subscripts if ch not in modes
            ),
        )
        logger.debug(f"Flattened {modes} in {self}: got {d}")
        return d

    def flatten_coefficient_modes(
        self, *, skip_zeros=True, force=False
    ) -> SegmentedTensorProduct:
        """
        Flatten the coefficients of the descriptor. Create new segments and paths to flatten the coefficients.

        Args:
            skip_zeros (bool, optional): Whether to skip paths with zero coefficients. Default is True.
            force (bool, optional): Whether to force the flattening by flattening extra necessary modes. Default is False.
        """
        return self.flatten_modes(
            self.coefficient_subscripts, skip_zeros=skip_zeros, force=force
        )

    def consolidate_modes(self, modes: Optional[str] = None) -> SegmentedTensorProduct:
        """
        Consolidate the descriptor by merging modes together.

        Args:
            modes (str, optional): The modes to consolidate. If None, consolidate all modes that are repeated.
        """
        if modes is None:
            # look for opportunities to consolidate
            for m in self.subscripts.modes():
                neighbors: set[str] = set()
                for operand in self.operands:
                    if m in operand.subscripts:
                        i = operand.subscripts.index(m)
                        if i < len(operand.subscripts) - 1:
                            neighbors.add(operand.subscripts[i + 1])
                        else:
                            neighbors.add(".")
                if len(neighbors) != 1:
                    continue
                n = neighbors.pop()
                if not n.isalpha():
                    continue
                # Zuvw_Ziu_Zjv_Zkw+ijk

                ok = True
                for operand in self.operands:
                    if m in operand.subscripts:
                        if n not in operand.subscripts:
                            ok = False
                            break
                    else:
                        if n in operand.subscripts:
                            ok = False
                            break

                if not ok:
                    continue

                if m in self.subscripts.coefficients:
                    if n not in self.subscripts.coefficients:
                        continue
                else:
                    if n in self.subscripts.coefficients:
                        continue

                return self._consolidate_pair_of_modes(m, n).consolidate_modes()

            return self

        for operand in self.operands:
            if modes not in operand.subscripts:
                if any(ch in operand.subscripts for ch in modes):
                    raise ValueError(
                        f"expected {modes} to be contiguous in the subscripts {operand.subscripts}."
                    )

        d = self
        for n in modes[1:]:
            d = d._consolidate_pair_of_modes(modes[0], n)

        logger.debug(f"Consolidated {modes} in {self}: got {d}")
        return d

    def _consolidate_pair_of_modes(self, m: str, n: str) -> SegmentedTensorProduct:
        """keep m, absorb n in m"""
        d0 = self
        if m in d0.coefficient_subscripts:
            if n not in d0.coefficient_subscripts:
                raise ValueError(
                    f"expected both {m} and {n} in the coefficients subscripts. {d0.subscripts}"
                )
            tmp = d0.coefficient_subscripts.replace(m, "").replace(n, "")
            d0 = d0.add_or_transpose_modes(
                ",".join(d0.subscripts.operands) + f"+{m}{n}{tmp}"
            )

        d1 = SegmentedTensorProduct.from_subscripts(d0.subscripts.replace(m + n, m))

        for oid, operand in enumerate(d0.operands):
            for segment in operand:
                if m in operand.subscripts:
                    i = operand.subscripts.index(m)
                    segment = list(segment)
                    segment[i] *= segment[i + 1]
                    segment.pop(i + 1)
                    segment = tuple(segment)
                d1.add_segment(oid, segment)

        if m in d0.coefficient_subscripts:
            i = d0.coefficient_subscripts.index(m)
            for path in d0.paths:
                c = path.coefficients
                c = np.reshape(
                    c, c.shape[:i] + (c.shape[i] * c.shape[i + 1],) + c.shape[i + 2 :]
                )
                d1.paths.append(stp.Path(indices=path.indices, coefficients=c))
        else:
            d1.paths = copy.deepcopy(d0.paths)

        return d1

    def round_coefficients_to_rational(
        self, max_denominator: int
    ) -> SegmentedTensorProduct:
        """
        Round the coefficients to the nearest ``p / q`` number with a given maximum denominator.

        Args:
            max_denominator (int): The maximum denominator, ``q < max_denominator``.
        """
        d = copy.deepcopy(self)
        d.paths = [
            stp.Path(
                indices=path.indices,
                coefficients=round_to_rational(path.coefficients, max_denominator),
            )
            for path in d.paths
        ]
        return d

    def round_coefficients_to_sqrt_rational(
        self, max_denominator: int
    ) -> SegmentedTensorProduct:
        """
        Round the coefficients to the nearest ``sqrt(p / q)`` number with a given maximum denominator.

        Args:
            max_denominator (int): The maximum denominator, ``q < max_denominator``.
        """
        d = copy.deepcopy(self)
        d.paths = [
            stp.Path(
                indices=path.indices,
                coefficients=round_to_sqrt_rational(path.coefficients, max_denominator),
            )
            for path in d.paths
        ]
        return d

    def modify_coefficients(
        self, f: Callable[[np.ndarray], np.ndarray]
    ) -> SegmentedTensorProduct:
        """
        Modify the coefficients of the descriptor.

        Args:
            f (callable): The function to apply to the coefficients.
        """
        d = copy.deepcopy(self)
        d.paths = [
            stp.Path(indices=path.indices, coefficients=f(path.coefficients))
            for path in d.paths
        ]
        return d

    def __mul__(self, factor: float) -> SegmentedTensorProduct:
        """Amplify the path coefficients by a factor."""
        if factor == 1.0:
            return self
        return self.modify_coefficients(lambda c: factor * c)

    def __rmul__(self, factor: float) -> SegmentedTensorProduct:
        """Amplify the path coefficients by a factor."""
        return self * factor


def _canonicalize_index(name: str, index: int, size: int) -> int:
    if not (-size <= index < size):
        raise ValueError(
            f"expected {name} to be between -{size} and {size - 1}, got {index}."
        )
    if index < 0:
        index += size
    return index
