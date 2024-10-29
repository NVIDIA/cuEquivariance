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


def format_dimensions_dict(dims: dict[str, set[int]]) -> str:
    return " ".join(
        f"{m}={next(iter(dd))}" if len(dd) == 1 else f"{m}={dd}"
        for m, dd in sorted(dims.items())
    )
