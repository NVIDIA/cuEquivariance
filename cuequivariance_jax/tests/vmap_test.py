# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex


@cue.assume("SO3", cue.ir_mul)
def test_vmap():
    def f(x):
        return x

    x = cuex.IrrepsArray({0: "1"}, jnp.zeros((3, 2)))
    y = jax.jit(cuex.vmap(f, 1, 0))(x)
    assert y.shape == (2, 3)
    assert y.dirreps == {1: cue.Irreps("1")}
