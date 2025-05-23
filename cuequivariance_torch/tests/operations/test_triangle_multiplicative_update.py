# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 8, reason="Ampere+ GPU is needed!")
def test_api():
    """Actual test is in cuequivariance_ops_torch."""
    from cuequivariance_torch import triangle_multiplicative_update
