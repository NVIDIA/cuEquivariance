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
from functools import wraps
from typing import Optional, Type, Union

import cuequivariance as cue  # noqa: F401
import cuequivariance.group_theory.irreps_array as irreps_array
from cuequivariance.group_theory import Irrep
from cuequivariance.group_theory.irreps_array.context_irrep_class import (
    pop_irrep_scope,
    push_irrep_scope,
)
from cuequivariance.group_theory.irreps_array.context_layout import (
    pop_layout_scope,
    push_layout_scope,
)


class assume:
    """
    ``assume`` is a context manager or decorator to assume the irrep class and layout for a block of code or a function.

    Examples:
        As a context manager:

        >>> with cue.assume(cue.SO3, cue.mul_ir):
        ...     rep = cue.IrrepsAndLayout("2x1")
        >>> rep.irreps
        2x1
        >>> rep.layout
        (mul,irrep)

        As a decorator:

        >>> @cue.assume(cue.SO3, cue.mul_ir)
        ... def foo():
        ...     return cue.IrrepsAndLayout("2x1")
        >>> assert foo() == rep
    """

    def __init__(
        self,
        irrep_class: Optional[Union[str, Type[Irrep]]] = None,
        layout: Optional[irreps_array.IrrepsLayout] = None,
    ):
        if isinstance(irrep_class, irreps_array.IrrepsLayout) and layout is None:
            irrep_class, layout = None, irrep_class

        self.irrep_class = irrep_class
        self.layout = layout

    def __enter__(self):
        self.old_irrep_class = push_irrep_scope(self.irrep_class)
        self.old_layout = push_layout_scope(self.layout)
        return self

    def __exit__(self, *exc):
        pop_irrep_scope(self.old_irrep_class)
        pop_layout_scope(self.old_layout)
        return False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
