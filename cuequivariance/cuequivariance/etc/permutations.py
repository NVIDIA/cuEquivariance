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
from typing import Sequence


def compose_permutations(p1: tuple[int, ...], p2: tuple[int, ...]) -> tuple[int, ...]:
    """Compose two permutations"""
    return tuple(p1[p2[i]] for i in range(len(p1)))


def inverse_permutation(p: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse a permutation"""
    return tuple(p.index(i) for i in range(len(p)))


def generate_permutations_from(
    generators: Sequence[tuple[int, ...]],
) -> set[tuple[int, ...]]:
    """Generate all permutations from a list of generators"""
    result = set(generators)

    while True:
        n = len(result)
        new_result = result.copy()
        for g in result:
            for h in result:
                new_result.add(compose_permutations(g, h))
        if len(new_result) == n:
            break
        result = new_result

    return result
