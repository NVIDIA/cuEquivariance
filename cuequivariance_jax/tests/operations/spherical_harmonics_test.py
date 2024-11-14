# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
from typing import *

import jax
import numpy as np
import pytest

import cuequivariance as cue
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
