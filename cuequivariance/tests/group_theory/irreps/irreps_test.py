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
import pytest

import cuequivariance as cue


def test_su2():
    irreps = cue.Irreps("SU2", "0 + 2x1/2 + 1")
    assert len(irreps) == 3

    assert irreps[0].mul == 1
    assert irreps[0].ir.j == 0

    assert irreps[1].mul == 2
    assert irreps[1].ir.j == 1 / 2

    assert irreps[2].mul == 1
    assert irreps[2].ir.j == 1

    assert irreps.dim == 1 + 2 * 2 + 3

    irreps = cue.Irreps("SU2", [(3, 0), (2, 1 / 2), (1, 1)])
    assert len(irreps) == 3
    assert irreps[1].ir == cue.SU2(1 / 2)


def test_o3():
    irreps = cue.Irreps("O3", "0e + 1o + 2e")
    assert len(irreps) == 3

    assert irreps[0].mul == 1
    assert irreps[0].ir.l == 0
    assert irreps[0].ir.p == 1

    assert irreps[1].mul == 1
    assert irreps[1].ir.l == 1
    assert irreps[1].ir.p == -1

    assert irreps[2].mul == 1
    assert irreps[2].ir.l == 2
    assert irreps[2].ir.p == 1

    assert irreps.dim == 1 + 3 + 5


def test_so3():
    irreps = cue.Irreps("SO3", "0 + 1 + 2")
    assert len(irreps) == 3

    assert irreps[0].mul == 1
    assert irreps[0].ir.l == 0

    assert irreps[1].mul == 1
    assert irreps[1].ir.l == 1

    assert irreps[2].mul == 1
    assert irreps[2].ir.l == 2

    assert irreps.dim == 1 + 3 + 5


def test_iter():
    irreps = cue.Irreps("SU2", "1 + 3x2")

    ((mul1, rep1), (mul2, rep2)) = irreps

    assert mul1 == 1
    assert rep1.j == 1
    assert mul2 == 3
    assert rep2.j == 2


def test_getitem():
    irreps = cue.Irreps("SU2", "1/2 + 16x2")

    assert irreps[0].mul == 1
    assert irreps[0].ir.j == 1 / 2

    assert irreps[1].mul == 16
    assert irreps[1].ir.j == 2


def test_contains():
    irreps = cue.Irreps("SU2", "1/2 + 16x2")

    assert 1 / 2 in irreps
    assert 3 / 2 not in irreps
    assert cue.SU2(2) in irreps


def test_repr():
    irreps = cue.Irreps("O3", "0e + 1o + 3x2e")

    assert repr(irreps) == "0e+1o+3x2e"


def test_count():
    irreps = cue.Irreps("SU2", "1/2 + 16x2 + 1/2")

    assert irreps.count(1 / 2) == 2
    assert irreps.count(2) == 16
    assert irreps.count(cue.SU2(2)) == 16
    assert irreps.count(12) == 0


def test_dim():
    irreps = cue.Irreps("SU2", "1/2 + 16x2")
    assert irreps.dim == 2 + 16 * 5


def test_num_irreps():
    irreps = cue.Irreps("SU2", "1/2 + 16x2")
    assert irreps.num_irreps == 17


def test_slices():
    irreps = cue.Irreps("SU2", "1/2 + 16x2")
    assert irreps.slices() == [slice(0, 2), slice(2, 2 + 16 * 5)]


def test_is_scalar():
    assert cue.Irreps("O3", "0e + 1o").is_scalar() is False
    assert cue.Irreps("O3", "0e + 0x1o").is_scalar() is False
    assert cue.Irreps("O3", "0e + 0e").is_scalar() is True


def test_add():
    assert cue.Irreps("SU2", "1/2") + cue.Irreps("SU2", "1 + 2") == cue.Irreps(
        "SU2", "1/2 + 1 + 2"
    )
    assert "1/2" + cue.Irreps("SU2", "1") == cue.Irreps("SU2", "1/2 + 1")


def test_mul():
    assert 3 * cue.Irreps("SU2", "1/2") == cue.Irreps("SU2", "3x1/2")


def test_floordiv():
    assert cue.Irreps("SU2", "3x1/2") // 3 == cue.Irreps("SU2", "1/2")


def test_eq():
    assert cue.Irreps("SU2", "1/2") == "1/2"
    assert not (cue.Irreps("SU2", "1/2") == cue.Irreps("SU2", "1"))
    assert not (cue.Irreps("SU2", "1/2 + 0x2") == cue.Irreps("SU2", "1/2"))
    assert not (cue.Irreps("SU2", "1/2") == "sapristi !")


def test_merge_consecutive():
    assert cue.Irreps("SU2", "1+1+0x2").merge_consecutive() == cue.Irreps(
        "SU2", "2x1+0x2"
    )


def test_remove_zero_multiplicities():
    assert cue.Irreps("SU2", "1+1+0x2").remove_zero_multiplicities() == cue.Irreps(
        "SU2", "1+1"
    )


def test_simplify():
    assert cue.Irreps("SU2", "1+1+0x2+3+1").simplify() == cue.Irreps("SU2", "2x1+3+1")


def test_sort():
    irreps, perm, inv = cue.Irreps("SU2", "1+1+0x2+3+1").sort()

    assert irreps == cue.Irreps("SU2", "1+1+1+0x2+3")
    assert perm == (0, 1, 3, 4, 2)
    assert inv == (0, 1, 4, 2, 3)


def test_regroup():
    assert cue.Irreps("SU2", "1+1+0x2+3+1").regroup() == cue.Irreps("SU2", "3x1+3")


def test_set_mul():
    assert cue.Irreps("SU2", "1+1+0x2+3+1").set_mul(2) == "2x1+2x1+2x2+2x3+2x1"


def test_filter():
    assert cue.Irreps("SU2", "0 + 1/2 + 1 + 3x3/2").filter(
        keep=lambda mul_ir: mul_ir.ir.j > 1
    ) == cue.Irreps("SU2", "3x3/2")
    assert cue.Irreps("SU2", "0 + 1/2 + 1 + 3x3/2").filter(
        drop=lambda mul_ir: mul_ir.ir.j > 1
    ) == cue.Irreps("SU2", "0+1/2+1")
    assert cue.Irreps("SU2", "0 + 1/2 + 1 + 3x3/2").filter(drop=1) == cue.Irreps(
        "SU2", "0+1/2+3x3/2"
    )
    assert cue.Irreps("SU2", "0 + 1/2 + 1 + 3x3/2").filter(
        keep="2x1/2 + 1"
    ) == cue.Irreps("SU2", "1/2+1")


def test_init():
    irreps_su2 = cue.Irreps("SU2", "1 + 2")
    with pytest.raises(ValueError):
        cue.Irreps("SO3", irreps_su2)

    with pytest.raises(ValueError):
        cue.Irreps("SO3", [(12, "1", -1)])

    with pytest.raises(ValueError):
        cue.Irreps("SO3", [(12, "1/2")])
