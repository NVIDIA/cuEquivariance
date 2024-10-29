# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

import warnings
from typing import *

import cuequivariance as cue


def default_layout(layout: Optional[cue.IrrepsLayout]) -> cue.IrrepsLayout:
    if layout is None:
        warnings.warn(
            "layout is not specified, defaulting to cue.mul_ir. This is the layout used in the e3nn library."
            " We use it as the default layout for compatibility with e3nn."
            " However, the cue.ir_mul layout is faster and more memory efficient."
            " Please specify the layout explicitly to avoid this warning."
        )
        return cue.mul_ir
    if isinstance(layout, str):
        return cue.IrrepsLayout[layout]
    return layout


def assert_same_group(*irreps_: cue.Irreps) -> None:
    group = irreps_[0].irrep_class
    for irreps in irreps_[1:]:
        if group != irreps.irrep_class:
            raise ValueError("The provided irreps are not of the same group.")


def default_irreps(
    *irreps_: Union[cue.Irreps, Any]
) -> Generator[cue.Irreps, None, None]:
    for irreps in irreps_:
        if isinstance(irreps, cue.Irreps):
            yield irreps
        else:
            warnings.warn(
                "irreps should be of type cue.Irreps, converting to cue.Irreps(cue.O3, ...) for compatibility with e3nn."
            )
            yield cue.Irreps(cue.O3, irreps)
