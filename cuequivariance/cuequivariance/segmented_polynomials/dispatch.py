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
import math
from typing import Generator, Tuple

# we cannot import cuequivariance as cue because of circular import
from cuequivariance.segmented_polynomials import SegmentedTensorProduct, Subscripts


def dispatch(
    descriptor: SegmentedTensorProduct,
    targets: list[Subscripts],
    permutation_mode: str,
) -> Generator[Tuple[SegmentedTensorProduct, Tuple[int, ...]], None, None]:
    """Dispatch a descriptor to a target subscripts.

    Args:
        descriptor (SegmentedTensorProduct): The descriptor to dispatch.
        targets (list of stp.Subscripts): A list of target subscripts.
        permutation_mode (str): The permutation mode. One of "permute_none", "permute_all", "permute_all_but_last".

    Yields:
        Tuple[stp.SegmentedTensorProduct, Tuple[int, ...]]:
            A tuple of the dispatched descriptor and the permutation of the operands.

    Note:
        The function tries to dispatch the descriptor to the target subscripts. If the descriptor
        is not dispatchable to the target subscripts, it will try to flatten the descriptor progressively
        and dispatch the flattened descriptor to the target subscripts. The function will yield all the
        possible dispatches found.
    """
    targets = [Subscripts(subscripts) for subscripts in targets]
    targets = [
        subscripts
        for subscripts in targets
        if subscripts.num_operands == descriptor.num_operands
    ]

    if permutation_mode == "permute_none":
        permutations = [tuple(range(descriptor.num_operands))]
    elif permutation_mode == "permute_all":
        permutations = list(itertools.permutations(range(descriptor.num_operands)))
    elif permutation_mode == "permute_all_but_last":
        permutations = [
            p + (descriptor.num_operands - 1,)
            for p in itertools.permutations(range(descriptor.num_operands - 1))
        ]
    else:
        raise ValueError(f"unknown permutation_mode: {permutation_mode}")

    # Squeeze all the channels of extent 1
    descriptor = descriptor.squeeze_modes()

    # Consolidate all the repeated channels e.g. ab -> a
    descriptor = descriptor.consolidate_modes()

    for subscripts in targets:
        for perm in permutations:
            d = descriptor.permute_operands(perm)

            for mapping in d.subscripts.is_subset_of(subscripts):
                d = d.add_or_rename_modes(subscripts, mapping=mapping)
                yield d, perm

    # Flatten one channel at a time, starting from the coefficients channel
    flattenable_powerset = descriptor.subscripts.flattenable_powerset()
    dims = descriptor.get_dimensions_dict()
    flattenable_powerset = sorted(
        flattenable_powerset,
        key=lambda channels: (
            len(channels),
            sum(-1 for ch in channels if ch in descriptor.subscripts.coefficients),
            math.prod(max(dims[ch]) for ch in channels),
        ),
    )

    for channels in flattenable_powerset:
        d = descriptor.flatten_modes(channels).remove_zero_paths()
        yield from dispatch(d, targets, permutation_mode)
