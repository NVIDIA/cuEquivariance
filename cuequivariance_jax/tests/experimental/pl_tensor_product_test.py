# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import timeit
from typing import *

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cuequivariance as cue
import cuequivariance.equivariant_tensor_product as etp
import cuequivariance.segmented_tensor_product as stp
import cuequivariance_jax as cuex
from cuequivariance_jax.experimental.pl_tensor_product import pl_tensor_product


def list_of_stp() -> Generator[stp.SegmentedTensorProduct, None, None]:
    d = etp.channelwise_tensor_product(
        128 * cue.Irreps("O3", "0e + 1o + 2e"),
        cue.Irreps("O3", "0e + 1o + 2e + 3o"),
        cue.Irreps("O3", "0e + 1o + 2e + 3o"),
    ).d
    d = d.flatten_coefficient_modes()
    yield d
    yield d.move_operand_last(2)

    d = etp.fixed_axis_angle_rotation(
        cue.Irreps("O3", "16x0e + 16x1o"), np.array([1.0, 0.0, 0.0]), np.pi / 2
    ).d
    yield d
    yield d.move_operand_last(0)


@pytest.mark.parametrize("d", list(list_of_stp()))
@pytest.mark.parametrize(
    "dtype_io, dtype_math, tol",
    [
        (jnp.float32, jnp.float32, 1e-5),
        (jnp.float16, jnp.float32, 0.04),
        (jnp.float16, jnp.float16, 0.1),
    ],
)
def test_pl_tensor_product(
    d: stp.SegmentedTensorProduct,
    dtype_io: jnp.dtype,
    dtype_math: jnp.dtype,
    tol: float,
):
    if jnp.ones(()).device.platform != "gpu":
        pytest.skip("CUDA not available")

    inputs = [
        jax.random.normal(jax.random.PRNGKey(i), (ope.size,), dtype=dtype_io)
        for i, ope in enumerate(d.operands[:-1])
    ]

    fA = jax.jit(lambda *x: pl_tensor_product(d, *x, dtype_math=dtype_math))
    fB = jax.jit(lambda *x: cuex.tensor_product(d, *x, dtype_math=dtype_math))

    _run_tests(f"{d}", fA, fB, inputs, tol)


@pytest.mark.parametrize("d", list(list_of_stp()))
@pytest.mark.parametrize(
    "dtype_io, dtype_math, tol",
    [
        (jnp.float32, jnp.float32, 1e-4),
        (jnp.float16, jnp.float32, 0.1),
        (jnp.float16, jnp.float16, 0.1),
    ],
)
def test_pl_tensor_product_vmapped(
    d: stp.SegmentedTensorProduct,
    dtype_io: jnp.dtype,
    dtype_math: jnp.dtype,
    tol: float,
):
    if jnp.ones(()).device.platform != "gpu":
        pytest.skip("CUDA not available")

    inputs = [
        jax.random.normal(jax.random.PRNGKey(i), (128, ope.size), dtype=dtype_io)
        for i, ope in enumerate(d.operands[:-1])
    ]

    fA = jax.jit(jax.vmap(lambda *x: pl_tensor_product(d, *x, dtype_math=dtype_math)))
    fB = jax.jit(jax.vmap(lambda *x: cuex.tensor_product(d, *x, dtype_math=dtype_math)))

    _run_tests(f"{d}", fA, fB, inputs, tol)


def _run_tests(
    desc: str, fA: Callable, fB: Callable, inputs: Sequence[Any], tol: float
):
    A = fA(*inputs)
    B = fB(*inputs)

    np.testing.assert_allclose(A, B, atol=tol, rtol=tol)

    tA = timeit.timeit(lambda: jax.block_until_ready(fA(*inputs)), number=10)
    tB = timeit.timeit(lambda: jax.block_until_ready(fB(*inputs)), number=10)
    if tA < tB:
        x = 100 * (tB / tA - 1)
        print(f"{desc}\nPallas: {x:.0f}% faster")
    else:
        x = 100 * (tA / tB - 1)
        print(f"{desc}\nPure JAX: {x:.0f}% faster")
