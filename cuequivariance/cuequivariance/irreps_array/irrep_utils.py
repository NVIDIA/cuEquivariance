# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from typing import *

import cuequivariance as cue
from cuequivariance import irreps_array


def into_list_of_irrep(
    irrep_class: Type[cue.Irrep],
    input: Union[
        str,
        cue.Irrep,
        irreps_array.MulIrrep,
        Iterable[Union[str, cue.Irrep, irreps_array.MulIrrep]],
    ],
) -> list[cue.Irrep]:
    if isinstance(input, str):
        return [rep for _, rep in cue.Irreps(irrep_class, input)]
    if isinstance(input, cue.Irrep):
        return [input]
    if isinstance(input, irreps_array.MulIrrep):
        return [input.ir]

    try:
        input = iter(input)
    except TypeError:
        return [irrep_class._from(input)]

    output = []
    for rep in input:
        if isinstance(rep, cue.Irrep):
            output.append(rep)
        elif isinstance(rep, irreps_array.MulIrrep):
            output.append(rep.ir)
        else:
            output.append(irrep_class._from(rep))
    return output
