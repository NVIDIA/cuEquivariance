# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from functools import wraps
from typing import *

import cuequivariance as cue
from cuequivariance import irreps_array
from cuequivariance.irreps_array.context_irrep_class import (
    pop_irrep_scope,
    push_irrep_scope,
)
from cuequivariance.irreps_array.context_layout import (
    pop_layout_scope,
    push_layout_scope,
)


class assume:
    """Context manager / decorator to assume the irrep class and layout for a block of code.

    Examples:
    ```
    with cue.assume(irrep_class="SU2", layout=cue.mul_ir):
        ...
    ```

    ```
    @cue.assume(irrep_class="SU2", layout=cue.mul_ir)
    def my_function():
        ...
    ```
    """

    def __init__(
        self,
        irrep_class: Optional[Union[str, Type[cue.Irrep]]] = None,
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
