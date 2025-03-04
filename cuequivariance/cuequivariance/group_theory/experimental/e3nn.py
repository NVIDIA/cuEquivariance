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
from typing import Iterator

import numpy as np

import cuequivariance as cue


class O3_e3nn(cue.O3):
    def __mul__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> Iterator["O3_e3nn"]:
        return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

    @classmethod
    def clebsch_gordan(
        cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
    ) -> np.ndarray:
        from e3nn import o3

        rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

        if rep1.p * rep2.p == rep3.p:
            return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                rep3.dim
            )
        else:
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    def __lt__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> bool:
        rep2 = rep1._from(rep2)
        return (rep1.l, rep1.p) < (rep2.l, rep2.p)

    @classmethod
    def iterator(cls) -> Iterator["O3_e3nn"]:
        for ell in itertools.count(0):
            yield O3_e3nn(l=ell, p=1 * (-1) ** ell)
            yield O3_e3nn(l=ell, p=-1 * (-1) ** ell)
