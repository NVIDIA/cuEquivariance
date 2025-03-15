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
from typing import Iterable, Type, Union

import cuequivariance as cue
from cuequivariance.group_theory import Irrep, irreps_array


def into_list_of_irrep(
    irrep_class: Type[Irrep],
    input: Union[
        str,
        Irrep,
        irreps_array.MulIrrep,
        Iterable[Union[str, Irrep, irreps_array.MulIrrep]],
    ],
) -> list[Irrep]:
    if isinstance(input, str):
        return [rep for _, rep in cue.Irreps(irrep_class, input)]
    if isinstance(input, Irrep):
        return [input]
    if isinstance(input, irreps_array.MulIrrep):
        return [input.ir]

    try:
        input = iter(input)
    except TypeError:
        return [irrep_class._from(input)]

    output = []
    for rep in input:
        if isinstance(rep, Irrep):
            output.append(rep)
        elif isinstance(rep, irreps_array.MulIrrep):
            output.append(rep.ir)
        else:
            output.append(irrep_class._from(rep))
    return output
