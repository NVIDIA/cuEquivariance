# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import pytest
import torch

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import descriptors

list_of_irreps = [
    cue.Irreps("O3", "4x0e + 4x1o"),
    cue.Irreps("O3", "2x1o + 5x0e + 2e + 1e + 1o"),
    cue.Irreps("O3", "2e + 0x0e + 0o + 0x1e + 1e"),
]


@pytest.mark.parametrize("irreps1", list_of_irreps)
@pytest.mark.parametrize("irreps2", list_of_irreps)
@pytest.mark.parametrize("irreps3", list_of_irreps)
@pytest.mark.parametrize("layout", [cue.ir_mul, cue.mul_ir])
@pytest.mark.parametrize("use_fallback", [False, True])
def test_fully_connected(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3: cue.Irreps,
    layout: cue.IrrepsLayout,
    use_fallback: bool,
):
    m = cuet.FullyConnectedTensorProduct(
        irreps1,
        irreps2,
        irreps3,
        shared_weights=True,
        internal_weights=True,
        layout=layout,
        device="cuda",
        dtype=torch.float64,
    )

    x1 = torch.randn(32, irreps1.dim, dtype=torch.float64).cuda()
    x2 = torch.randn(32, irreps2.dim, dtype=torch.float64).cuda()

    out1 = m(x1, x2, use_fallback=use_fallback)

    d = descriptors.fully_connected_tensor_product(irreps1, irreps2, irreps3).d
    if layout == cue.mul_ir:
        d = d.add_or_transpose_modes("uvw,ui,vj,wk+ijk")
    mfx = cuet.TensorProduct(d, math_dtype=torch.float64).cuda()
    out2 = mfx(
        m.weight.to(torch.float64),
        x1.to(torch.float64),
        x2.to(torch.float64),
        use_fallback=True,
    ).to(out1.dtype)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)
