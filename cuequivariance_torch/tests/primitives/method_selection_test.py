# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from cuequivariance_torch.primitives import segmented_polynomial

import cuequivariance as cue
import cuequivariance_torch as cuet


@pytest.fixture
def no_backend(monkeypatch):
    monkeypatch.setattr(segmented_polynomial, "HAS_CUE_OPS", False)


def test_unstated_method_falls_back(no_backend):
    with pytest.warns(UserWarning, match="Falling back to naive"):
        m = cuet.SphericalHarmonics([0, 1, 2])
    assert m.method == "naive"
    assert m(torch.randn(4, 3)).shape == (4, 9)


@pytest.mark.parametrize("method", ["uniform_1d", "fused_tp", "indexed_linear"])
def test_stated_cuda_method_raises(no_backend, method):
    poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2]).polynomial
    with pytest.raises(ImportError, match="requires cuequivariance_ops_torch"):
        cuet.SegmentedPolynomial(poly, method=method)


def test_stated_cuda_method_raises_in_layer(no_backend):
    with pytest.raises(ImportError, match="requires cuequivariance_ops_torch"):
        cuet.SphericalHarmonics([0, 1, 2], method="uniform_1d")


def test_naive_method_needs_no_backend(no_backend):
    m = cuet.SphericalHarmonics([0, 1, 2], method="naive")
    assert m.method == "naive"


def test_transpose_method(no_backend):
    segments = [(2, 3), (4, 5)]
    with pytest.raises(ImportError, match="requires cuequivariance_ops_torch"):
        cuet.TransposeSegments(segments, method="segmented_transpose")
    with pytest.raises(ValueError, match="Invalid method"):
        cuet.TransposeSegments(segments, method="uniform_1d")

    x = torch.randn(3, 26)
    torch.testing.assert_close(
        cuet.TransposeSegments(segments)(x),
        cuet.TransposeSegments(segments, method="naive")(x),
    )
