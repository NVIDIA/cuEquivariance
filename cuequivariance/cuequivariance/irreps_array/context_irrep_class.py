# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from typing import *

import cuequivariance as cue

_irrep_class: Union[None, str, Type[cue.Irrep]] = None


def get_irrep_scope(raising: bool = True) -> Type[cue.Irrep]:
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
