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
from cuequivariance.group_theory.experimental.gatr import (
    gatr_geometric_product,
    gatr_linear,
    gatr_outer_product,
)


def test_geometric_product():
    gatr_geometric_product()


def test_outer_product():
    gatr_outer_product()


def test_linear():
    gatr_linear(32, 32)
