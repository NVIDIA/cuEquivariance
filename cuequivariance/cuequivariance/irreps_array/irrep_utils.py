# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
