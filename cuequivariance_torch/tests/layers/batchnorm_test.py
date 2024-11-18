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
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet


def test_equivariant() -> None:
    irreps = cue.Irreps("O3", "3x0e + 3x0o + 4x1e")
    m = cuet.layers.BatchNorm(irreps, layout=cue.mul_ir)
    m(torch.randn(16, irreps.dim))
    m(torch.randn(16, irreps.dim))
    m.train()


@pytest.mark.parametrize("layout", [cue.mul_ir, cue.ir_mul])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("instance", [True, False])
def test_modes(layout, affine, reduce, instance) -> None:
    irreps = cue.Irreps("O3", "10x0e + 5x1e")

    m = cuet.layers.BatchNorm(
        irreps,
        layout=layout,
        affine=affine,
        reduce=reduce,
        instance=instance,
    )
    repr(m)

    m.train()
    m(torch.randn(20, 20, irreps.dim))

    m.eval()
    m(torch.randn(20, 20, irreps.dim))


@pytest.mark.parametrize("instance", [True, False])
def test_normalization(instance) -> None:
    float_tolerance = 1e-5
    sqrt_float_tolerance = float_tolerance**0.5

    batch, n = 20, 20
    irreps = cue.Irreps("O3", "3x0e + 4x1e")

    m = cuet.layers.BatchNorm(irreps, layout=cue.mul_ir, instance=instance)

    x = torch.randn(batch, n, irreps.dim).mul(5.0).add(10.0)
    x = m(x)

    a = x[..., :3]  # [batch, space, mul]
    assert a.mean([0, 1]).abs().max() < float_tolerance
    assert a.pow(2).mean([0, 1]).sub(1).abs().max() < sqrt_float_tolerance

    a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
    assert a.pow(2).mean(3).mean([0, 1]).sub(1).abs().max() < sqrt_float_tolerance


def test_e3nn_compat():
    try:
        from e3nn import o3
    except ImportError:
        pytest.skip("e3nn not installed")

    irreps = o3.Irreps("3x0e + 4x1e")
    with pytest.warns(UserWarning):
        cuet.layers.BatchNorm(irreps, layout=cue.mul_ir)


def test_O3_default_warning():
    with pytest.warns(UserWarning):
        m = cuet.layers.BatchNorm("32x0e + 16x1o", layout=cue.ir_mul)
        assert m.irreps.irrep_class == cue.O3
