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
from typing import Type, Union

import cuequivariance as cue
from cuequivariance.group_theory.representations import Irrep

_irrep_class: Union[None, str, Type[Irrep]] = None


def get_irrep_scope(raising: bool = True) -> Type[Irrep]:
    if raising and _irrep_class is None:
        raise ValueError(
            "No irrep class set in the context. Please specify the irrep class explicitly or use ``with cue.assume(irrep):``."
        )

    if isinstance(_irrep_class, str):
        return getattr(cue, _irrep_class)
    return _irrep_class


def push_irrep_scope(irrep_class):
    global _irrep_class
    old_irrep_class = _irrep_class
    _irrep_class = irrep_class
    return old_irrep_class


def pop_irrep_scope(old_irrep_class):
    global _irrep_class
    _irrep_class = old_irrep_class
