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
import itertools
from collections import defaultdict
from typing import *

import numpy as np


class Buffer(int):
    pass


class InBuffer(Buffer):
    pass


class OutBuffer(Buffer):
    pass


T = TypeVar("T")


class Computation(tuple):
    def __new__(cls, elements):
        elements = list(elements)
        assert all(isinstance(b, Buffer) for b in elements), elements
        assert sum(isinstance(b, OutBuffer) for b in elements) == 1, elements
        return super().__new__(cls, elements)

    @property
    def num_operands(self) -> int:
        return len(self)

    @property
    def in_buffers(self) -> tuple[InBuffer, ...]:
        return tuple(b for b in self if isinstance(b, InBuffer))

    @property
    def out_buffer(self) -> OutBuffer:
        return next(b for b in self if isinstance(b, OutBuffer))

    @property
    def in_operands(self) -> tuple[int, ...]:
        return tuple(oid for oid, b in enumerate(self) if isinstance(b, InBuffer))

    @property
    def out_operand(self) -> int:
        return next(oid for oid, b in enumerate(self) if isinstance(b, OutBuffer))

    def map_operands(
        self,
        in_buffers: Sequence[T],
        out_buffers: Sequence[T] | None = None,
    ) -> list[T | None]:
        in_buffers = list(in_buffers)
        if out_buffers is None:
            return [in_buffers[b] if isinstance(b, InBuffer) else None for b in self]
        else:
            out_buffers = list(out_buffers)
            return [
                in_buffers[b] if isinstance(b, InBuffer) else out_buffers[b]
                for b in self
            ]

    def map_inputs(
        self,
        in_buffers: Sequence[T],
    ) -> list[T]:
        in_buffers = list(in_buffers)
        return [in_buffers[b] for b in self.in_buffers]


class TensorProductExecution:
    computations: tuple[Computation, ...]
    # (num_computations, num_operands)  # which in/out buffer to use for each computation

    def __init__(self, computations: tuple[Computation, ...]):
        self.computations = tuple(Computation(c) for c in computations)

    def __hash__(self) -> int:
        return hash(self.computations)

    def __repr__(self):
        IVARS = "abcdefghijklmnopqrstuvwxyz"
        OVARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        text = [
            "({inputs}) -> ({outputs})".format(
                inputs=", ".join([IVARS[b] for b in self.in_buffers]),
                outputs=", ".join([OVARS[b] for b in self.out_buffers]),
            )
        ]
        for comp in self.computations:
            text += [
                "  "
                + " ".join(
                    IVARS[b] if isinstance(b, InBuffer) else OVARS[b] for b in comp
                )
            ]
        return "\n".join(text)

    @property
    def is_trivial(self) -> bool:
        return len(self.computations) == 0

    @property
    def num_operands(self) -> int:
        assert not self.is_trivial
        for c in self.computations:
            return len(c)

    @property
    def in_buffers(self) -> tuple[int, ...]:
        return tuple(sorted({b for c in self.computations for b in c.in_buffers}))

    @property
    def out_buffers(self) -> tuple[int, ...]:
        return tuple(sorted({c.out_buffer for c in self.computations}))

    @property
    def max_in_buffer(self) -> int:
        if len(self.in_buffers) == 0:
            return -1
        return max(self.in_buffers)

    @property
    def max_out_buffer(self) -> int:
        assert not self.is_trivial
        return max(self.out_buffers)

    @property
    def in_buffers_per_operand(self) -> tuple[tuple[int, ...], ...]:
        x = [set() for _ in range(self.num_operands)]
        for c in self.computations:
            for i, b in zip(c.in_operands, c.in_buffers):
                x[i].add(b)
        return tuple(tuple(sorted(s)) for s in x)

    @property
    def out_buffers_per_operand(self) -> tuple[tuple[int, ...], ...]:
        x = [set() for _ in range(self.num_operands)]
        for c in self.computations:
            x[c.out_operand].add(c.out_buffer)
        return tuple(tuple(sorted(s)) for s in x)

    @property
    def num_inputs_per_operand(self) -> tuple[int, ...]:
        return tuple(len(s) for s in self.in_buffers_per_operand)

    @property
    def num_outputs_per_operand(self) -> tuple[int, ...]:
        return tuple(len(s) for s in self.out_buffers_per_operand)

    def map_buffers(
        self,
        f_in: Callable[[int], int] | None,
        f_out: Callable[[int], int] | None = None,
    ) -> "TensorProductExecution":
        if f_in is None:
            f_in = lambda b: b
        if f_out is None:
            f_out = lambda b: b
        return TensorProductExecution(
            tuple(
                Computation(
                    (
                        InBuffer(int(f_in(b)))
                        if isinstance(b, InBuffer)
                        else OutBuffer(int(f_out(b)))
                    )
                    for b in comp
                )
                for comp in self.computations
            )
        )

    def simplify(self) -> "TensorProductExecution":
        return self.map_buffers(
            lambda b: self.in_buffers.index(b),
            lambda b: self.out_buffers.index(b),
        )

    def jvp(self, has_tangent: list[bool]) -> "TensorProductExecution":
        assert self.max_in_buffer < len(has_tangent)

        bid = len(has_tangent)

        tangents_new_bid = []
        for has in has_tangent:
            if has:
                tangents_new_bid.append(bid)
                bid += 1
            else:
                tangents_new_bid.append(None)

        new_computations = []
        for computation in self.computations:
            for oid, bid in zip(
                computation.in_operands, computation.map_inputs(tangents_new_bid)
            ):
                if bid is None:
                    continue  # the tangent is zero

                c = list(computation)
                c[oid] = InBuffer(bid)
                new_computations.append(Computation(c))

        return TensorProductExecution(tuple(new_computations))

    def transpose(
        self,
        is_undefined_primal: list[bool],
        has_cotangent: list[bool],
    ) -> "TensorProductExecution":
        assert self.max_in_buffer < len(is_undefined_primal)
        assert self.max_out_buffer < len(has_cotangent)

        in_bid = 0
        out_bid = 0

        primals_new_bid = []
        for undef in is_undefined_primal:
            if undef:
                primals_new_bid.append(out_bid)
                out_bid += 1
            else:
                primals_new_bid.append(in_bid)
                in_bid += 1

        cotangents_new_bid = []
        for has in has_cotangent:
            if has:
                cotangents_new_bid.append(in_bid)
                in_bid += 1
            else:
                cotangents_new_bid.append(None)

        del in_bid, out_bid

        new_computations = []
        for comp in self.computations:
            if not has_cotangent[comp.out_buffer]:
                continue  # cotangent is zero

            for oid in comp.in_operands:
                if not is_undefined_primal[comp[oid]]:
                    continue  # nothing to transpose

                c = [None] * len(comp)
                # undefined primal -> output
                c[oid] = OutBuffer(primals_new_bid[comp[oid]])
                # output -> cotangent input
                c[comp.out_operand] = InBuffer(cotangents_new_bid[comp.out_buffer])
                # rest of inputs
                for i in range(comp.num_operands):
                    if i != oid and i != comp.out_operand:
                        c[i] = InBuffer(primals_new_bid[comp[i]])

                new_computations.append(Computation(c))

        return TensorProductExecution(tuple(new_computations))

    def group_by_symmetries(
        self, permutations: list[tuple[int, ...]]
    ) -> Generator[tuple[int, "TensorProductExecution"], None, None]:
        """Used in JVP. Allows to avoid redundant computations.

        Keeps only one computation per equivalence class of computations

        Args:
            permutations (list[tuple[int, ...]]): permutations of operands
                that define the equivalence relation
        """
        buckets: list[list[Computation]] = []
        for c in self.computations:
            found_bucket = False
            for bucket in buckets:
                rep = bucket[0]
                if any(Computation(rep[p] for p in perm) == c for perm in permutations):
                    bucket.append(c)
                    found_bucket = True
                    break
            if not found_bucket:
                buckets.append([c])
        by_mul = defaultdict(list)
        for bucket in buckets:
            by_mul[len(bucket)].append(bucket[0])
        for mul, reps in by_mul.items():
            yield mul, TensorProductExecution(tuple(reps))

    def group_by_identical_buffers(
        self,
    ) -> Generator[tuple[list[list[int]], "TensorProductExecution"], None, None]:
        """Used in the evaluation of the TP.
        If two inputs are identical we don't need to compute the assymetric part twice.
        """

        def partition(computation: Computation) -> list[list[int]]:
            bid_to_oid = defaultdict(list)
            for oid, b in enumerate(computation):
                b = (type(b), b)
                bid_to_oid[b].append(oid)
            return sorted(map(sorted, bid_to_oid.values()))

        for p, group in itertools.groupby(self.computations, key=partition):
            yield p, TensorProductExecution(tuple(group))

    def display(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        in_buffers: list[dict[int, Any]] = [{} for _ in range(self.num_operands)]
        out_buffers: list[dict[int, Any]] = [{} for _ in range(self.num_operands)]

        RADIUS = 0.06

        color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for c, color in zip(self.computations, color_cycle):
            assert isinstance(c, Computation)

            if c.out_buffer not in out_buffers[c.out_operand]:
                y = len(out_buffers[c.out_operand])
                art_out = plt.Circle(
                    (c.out_operand + 1 / 6, 0.5 * y + 0.25),
                    RADIUS,
                    color="black",
                    fill=False,
                )
                ax.add_artist(art_out)
                ax.annotate(
                    f"{int(c.out_buffer)}",
                    xy=art_out.center,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="red",
                )
                out_buffers[c.out_operand][c.out_buffer] = art_out
            art_out = out_buffers[c.out_operand][c.out_buffer]

            for oid, b in zip(c.in_operands, c.in_buffers):
                if b not in in_buffers[oid]:
                    y = self.in_buffers_per_operand[oid].index(b)
                    art = plt.Circle(
                        (oid - 1 / 6, 0.5 * y), RADIUS, color="black", fill=False
                    )
                    ax.add_artist(art)
                    ax.annotate(
                        f"{int(b)}",
                        xy=art.center,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                    in_buffers[oid][b] = art

                art = in_buffers[oid][b]

                start, end = np.array(art.center), np.array(art_out.center)
                rand_tr = np.random.randn(2) * RADIUS * 0.1
                start, end = start + rand_tr, end + rand_tr
                length = np.linalg.norm(end - start)
                if length > 2 * RADIUS:
                    ax.annotate(
                        "",
                        xytext=start + (end - start) * RADIUS / length,
                        xy=end - (end - start) * RADIUS / length,
                        arrowprops=dict(arrowstyle="->", color=color),
                    )

        y_min, y_max = 0, 0
        for art in ax.get_children():
            if not isinstance(art, plt.Circle):
                continue
            y_min = min(y_min, art.center[1])
            y_max = max(y_max, art.center[1])

        ax.set_ylim(y_min - 0.2, y_max + 0.2)
        ax.set_xlim(-0.5, self.num_operands - 0.5)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xticks(
            range(self.num_operands), [f"operand {i}" for i in range(self.num_operands)]
        )
        ax.set_yticks([])
        for i in range(self.num_operands + 1):
            ax.axvline(i - 0.5, color="black", lw=1)

        return ax
