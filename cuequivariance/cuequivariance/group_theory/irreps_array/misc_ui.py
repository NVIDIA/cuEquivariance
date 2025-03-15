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

import warnings
from typing import Any, Generator, Optional, Union

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
    *irreps_: Union[cue.Irreps, Any],
) -> Generator[cue.Irreps, None, None]:
    for irreps in irreps_:
        if isinstance(irreps, cue.Irreps):
            yield irreps
        else:
            warnings.warn(
                "irreps should be of type cue.Irreps, converting to cue.Irreps(cue.O3, ...) for compatibility with e3nn."
            )
            yield cue.Irreps(cue.O3, irreps)
