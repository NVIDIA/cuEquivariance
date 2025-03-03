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
import itertools
from typing import Any, Sequence

import numpy as np

import cuequivariance as cue


def stable_unique(xs: Sequence[Any]) -> Sequence[Any]:
    seen = set()
    gen = (x for x in xs if not (x in seen or seen.add(x)))

    ty = type(xs)
    if isinstance(xs, str):
        return ty("".join(gen))
    return ty(gen)


def dot(
    x: cue.SegmentedTensorProduct,
    y: cue.SegmentedTensorProduct,
    *contraction: tuple[int, int],
) -> cue.SegmentedTensorProduct:
    """
    Compute the dot product of two segmented tensor products.

    Args:
        x: The first segmented tensor product.
        y: The second segmented tensor product.
        contraction: A tuple of two integers representing the indices of the operands to contract.

    Returns:
        The segmented tensor product resulting from the dot product.
    """
    for oidx, oidy in contraction:
        if x.operands_and_subscripts[oidx] != y.operands_and_subscripts[oidy]:
            raise ValueError("Operands to contract must be the same.")

    x_keep = [
        oid for oid in range(x.num_operands) if all(oid != i for i, _ in contraction)
    ]
    y_keep = [
        oid for oid in range(y.num_operands) if all(oid != j for _, j in contraction)
    ]

    d = cue.SegmentedTensorProduct(
        coefficient_subscripts=stable_unique(
            x.coefficient_subscripts + y.coefficient_subscripts
        )
    )
    d = dataclasses.replace(
        d,
        operands_and_subscripts=[x.operands_and_subscripts[i] for i in x_keep]
        + [y.operands_and_subscripts[i] for i in y_keep],
    )

    formula = f"{x.coefficient_subscripts} , {y.coefficient_subscripts} -> {d.coefficient_subscripts}"

    if len(contraction) == 0:
        for pathx, pathy in itertools.product(x.paths, y.paths):
            d.add_path(
                *pathx.indices,
                *pathy.indices,
                c=np.einsum(formula, pathx.coefficients, pathy.coefficients),
            )
        return d

    oidx, oidy = contraction[0]
    operand = x.operands[oidx]  # = y.operands[oidy]

    x, y = x.sort_paths(oidx), y.sort_paths(oidy)

    cx = x.compressed_path_segment(oidx)
    cy = y.compressed_path_segment(oidy)
    for sid in range(operand.num_segments):
        for pathx, pathy in itertools.product(
            x.paths[cx[sid] : cx[sid + 1]], y.paths[cy[sid] : cy[sid + 1]]
        ):
            assert pathx.indices[oidx] == pathy.indices[oidy]
            if all(
                pathx.indices[oidx_] == pathy.indices[oidy_]
                for oidx_, oidy_ in contraction
            ):
                d.add_path(
                    *[pathx.indices[i] for i in x_keep],
                    *[pathy.indices[i] for i in y_keep],
                    c=np.einsum(formula, pathx.coefficients, pathy.coefficients),
                )

    return d


def trace(
    d: cue.SegmentedTensorProduct, *contraction: tuple[int, int]
) -> cue.SegmentedTensorProduct:
    """
    Compute the trace of a segmented tensor product.

    Args:
        d: The segmented tensor product.
        contraction: A tuple of two integers representing the indices of the operands to contract.

    Returns:
        The segmented tensor product resulting from the trace.
    """
    for oidx, oidy in contraction:
        if d.operands[oidx].segments != d.operands[oidy].segments:
            raise ValueError("Operands to contract must be the same.")

    keep = [
        oid
        for oid in range(d.num_operands)
        if all(oid not in (i, j) for i, j in contraction)
    ]
    mapping = {
        chj: chi
        for i, j in contraction
        for chi, chj in zip(d.subscripts.operands[i], d.subscripts.operands[j])
    }
    f = lambda subscripts: "".join(mapping.get(ch, ch) for ch in subscripts)  # noqa

    coefficients_subscripts_renamed = f(d.coefficient_subscripts)
    coefficients_subscripts_compressed = stable_unique(coefficients_subscripts_renamed)

    dout = cue.SegmentedTensorProduct(
        coefficient_subscripts=coefficients_subscripts_compressed,
        operands_and_subscripts=[
            (ope, f(ss)) for ope, ss in (d.operands_and_subscripts[i] for i in keep)
        ],
    )

    formula = (
        f"{coefficients_subscripts_renamed} -> {coefficients_subscripts_compressed}"
    )
    for path in d.paths:
        if all(path.indices[i] == path.indices[j] for i, j in contraction):
            dout.add_path(
                *[path.indices[i] for i in keep],
                c=np.einsum(formula, path.coefficients),
            )

    return dout
