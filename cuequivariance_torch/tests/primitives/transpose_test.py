# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import torch
import cuequivariance_torch as cuet

import pytest

dtypes = [torch.float32, torch.float64]
if torch.cuda.get_device_capability()[0] >= 8:
    dtypes += [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("use_fallback", [False, True])
@pytest.mark.parametrize("dtype", dtypes)
def test_transpose(use_fallback: bool, dtype: torch.dtype):
    """
    1 2 3       1 4
    4 5 6       2 5
                3 6
        ------>
    10 11       10 12
    12 13       11 13
    """

    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10, 11, 12, 13]], dtype=dtype
    ).cuda()
    segments = [(2, 3), (2, 2)]
    xt = torch.tensor(
        [[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 10, 12, 11, 13]], dtype=dtype
    ).cuda()

    m = cuet.TransposeSegments(segments).cuda()
    torch.testing.assert_close(m(x, use_fallback=use_fallback), xt)
