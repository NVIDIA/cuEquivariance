# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
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
