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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cuequivariance as cue
import cuequivariance.equivariant_tensor_product as etp
import cuequivariance.segmented_tensor_product as stp
import cuequivariance_jax as cuex

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("shape", [(2, 3), ()])
def test_spherical_harmonics(shape):
    x = cuex.IrrepsArray(
        cue.Irreps(cue.O3, "1o"), np.random.randn(*shape, 3), cue.ir_mul
    )
    y = cuex.spherical_harmonics([0, 1, 2], x)
    assert y.shape == shape + (9,)
    assert y.irreps() == cue.Irreps(cue.O3, "0e + 1o + 2e")


# def test_edge_case():
#     x = cuex.IrrepsArray(cue.Irreps(cue.O3, "1o"), np.random.randn(2, 2, 3), cue.ir_mul)
#     y = cuex.spherical_harmonics([0], x)
#     assert y.shape == (2, 2, 1)
#     assert y.irreps() == cue.Irreps(cue.O3, "0e")
