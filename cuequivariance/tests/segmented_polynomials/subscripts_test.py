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
import cuequivariance.segmented_polynomials as sp
import pytest


def test_subscripts():
    with pytest.raises(ValueError):
        sp.Subscripts("#$%@")

    with pytest.raises(ValueError):
        sp.Subscripts("Zu")  # uppercase not supported anymore

    with pytest.raises(ValueError):
        sp.Subscripts("uZ")  # uppercase after lowercase

    with pytest.raises(ValueError):
        sp.Subscripts("uZ+ij+kl")  # multiple + signs

    subscripts = sp.Subscripts("ui,vj,uvk+ijk")
    assert subscripts.canonicalize() == "ui,vj,uvk+ijk"

    assert subscripts.coefficients == "ijk"

    assert subscripts.num_operands == 3
    assert subscripts.operands[0] == "ui"
    assert subscripts.operands[1] == "vj"
    assert subscripts.operands[2] == "uvk"

    assert subscripts.is_subset_of("uwi,vwj,uvk+ijk")  # using w=1
    assert subscripts.is_equivalent("ui_vj_uvk+ijk")


def test_canonicalize():
    assert sp.Subscripts("ui").canonicalize() == "uv"
    assert sp.Subscripts("ab,ad+bd").canonicalize() == "ui,uj+ij"
    assert sp.Subscripts("i,j+ji").canonicalize() == "i,j+ji"
    assert sp.Subscripts("j,i+ij").canonicalize() == "i,j+ji"
