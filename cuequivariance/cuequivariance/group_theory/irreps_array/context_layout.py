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
from typing import Union

from cuequivariance.group_theory.irreps_array import IrrepsLayout

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
