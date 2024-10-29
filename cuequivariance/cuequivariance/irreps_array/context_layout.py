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

from cuequivariance.irreps_array import IrrepsLayout

_layout: Union[None, IrrepsLayout] = None


def get_layout_scope(raising: bool = True) -> IrrepsLayout:
    if raising and _layout is None:
        raise ValueError(
            "No layout set in the context. Please specify the layout explicitly or use ``with cue.assume(layout):``."
        )

    return _layout


def push_layout_scope(layout):
    global _layout
    old_layout = _layout
    _layout = layout
    return old_layout


def pop_layout_scope(old_layout):
    global _layout
    _layout = old_layout
