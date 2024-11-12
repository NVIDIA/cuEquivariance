# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from typing import *


def format_dimensions_dict(dims: dict[str, set[int]]) -> str:
    return " ".join(
        f"{m}={next(iter(dd))}" if len(dd) == 1 else f"{m}={dd}"
        for m, dd in sorted(dims.items())
    )
