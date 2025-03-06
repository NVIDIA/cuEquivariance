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


def format_set(s: set[int]) -> str:
    if len(s) == 0:
        return ""
    if len(s) == 1:
        return str(next(iter(s)))
    return "{" + ", ".join(str(i) for i in sorted(s)) + "}"


def format_dimensions_dict(dims: dict[str, set[int]]) -> str:
    return " ".join(f"{m}={format_set(dd)}" for m, dd in sorted(dims.items()))
