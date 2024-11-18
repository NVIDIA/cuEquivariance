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
from typing import *

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cuequivariance as cue
from cuequivariance import descriptors
import cuequivariance.segmented_tensor_product as stp
import cuequivariance_jax as cuex

jax.config.update("jax_enable_x64", True)


def list_of_stp() -> Generator[stp.SegmentedTensorProduct, None, None]:
    d = descriptors.channelwise_tensor_product(
        32 * cue.Irreps("O3", "0e + 1o + 2e"),
        cue.Irreps("O3", "0e + 1o"),
        cue.Irreps("O3", "0e + 1o + 2e + 3o"),
    ).d.flatten_coefficient_modes()
    yield d
    yield d.move_operand_last(2)

    d = descriptors.fixed_axis_angle_rotation(
        cue.Irreps("O3", "64x0e + 64x1o"), np.array([1.0, 0.0, 0.0]), np.pi / 2
    ).d
    yield d
    yield d.move_operand_last(0)

    yield from descriptors.symmetric_contraction(
        32 * cue.Irreps("O3", "0e + 1o + 2e"),
        32 * cue.Irreps("O3", "0e + 1o"),
        [0, 1, 2],
    ).ds


def compare(fA: Callable, fB: Callable, inputs: Sequence[Any], tol: float):
    np.testing.assert_allclose(fA(*inputs), fB(*inputs), atol=tol, rtol=tol)


def make_inputs(d: stp.SegmentedTensorProduct, dtype):
    return [
        jax.random.normal(jax.random.key(i), (3 if i == 0 else 1, ope.size), dtype)
        for i, ope in enumerate(d.operands[:-1])
    ]


##########################################################################################
##########################################################################################
##########################################################################################


@pytest.mark.parametrize("d", list(list_of_stp()))
@pytest.mark.parametrize(
    "dtype_io, dtype_math, tol",
    [
        (jnp.float64, jnp.float64, 1e-7),
        (jnp.float32, jnp.float32, 1e-5),
        (jnp.float16, jnp.float32, 0.04),
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        "stacked",
        "compact_stacked",
        "indexed_vmap",
        "indexed_compact",
        "indexed_for_loop",
        "sliced",
    ],
)
def test_tensor_product_forward(
    d: stp.SegmentedTensorProduct,
    dtype_io: jnp.dtype,
    dtype_math: jnp.dtype,
    tol: float,
    algorithm: str,
):
    inputs = make_inputs(d, dtype_io)
    fA = jax.jit(cuex.tensor_product(d, dtype_math=dtype_math, algorithm=algorithm))
    fB = lambda *x: stp.compute_last_operand(d, *x)
    compare(fA, fB, inputs, tol)


@pytest.mark.parametrize("d", list(list_of_stp()))
@pytest.mark.parametrize(
    "algorithm",
    [
        "stacked",
        "compact_stacked",
        "indexed_vmap",
        "indexed_compact",
        "indexed_for_loop",
        "sliced",
    ],
)
def test_tensor_product_backward(d: stp.SegmentedTensorProduct, algorithm: str):
    x = make_inputs(d, jnp.float64)

    out = []
    for options in [
        dict(algorithm="sliced", use_custom_primitive=False),
        dict(algorithm=algorithm, use_custom_primitive=True),
    ]:
        f = lambda *x: cuex.tensor_product(d, *x, **options).sum() ** 2
        A = jax.grad(f)(*x)
        out.append(A)

    np.testing.assert_allclose(out[0], out[1], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("d", list(list_of_stp()))
@pytest.mark.parametrize(
    "algorithm",
    [
        "stacked",
        "compact_stacked",
        "indexed_vmap",
        "indexed_compact",
        "indexed_for_loop",
        "sliced",
    ],
)
def test_tensor_product_double_backward(d: stp.SegmentedTensorProduct, algorithm: str):
    x = make_inputs(d, jnp.float64)

    out = []
    for options in [
        dict(algorithm="sliced", use_custom_primitive=False),
        dict(algorithm=algorithm, use_custom_primitive=True),
    ]:
        f0 = lambda *x: cuex.tensor_product(d, *x, **options).sum()
        f1 = lambda *x: jax.grad(f0)(*x).sum()
        f2 = lambda *x: jax.grad(f1)(*x).sum()
        A = f2(*x)
        out.append(A)

    np.testing.assert_allclose(out[0], out[1], atol=1e-10, rtol=1e-10)


##########################################################################################


def test_edge_cases():
    d = stp.SegmentedTensorProduct.empty_segments([1])
    assert cuex.tensor_product(d).shape == (1,)

    d.add_path(0, c=123)
    assert cuex.tensor_product(d).shape == (1,)


def test_UnshapedArray_bug():
    e = cue.descriptors.symmetric_contraction(
        cue.Irreps("O3", "0e"), cue.Irreps("O3", "0e"), [0, 1]
    )
    w = jnp.ones((1, 2))
    x = jnp.ones((2, 1))

    def f(w, x):
        a = cuex.tensor_product(e.ds[0], w, use_custom_primitive=True)
        b = cuex.tensor_product(e.ds[1], w, x, use_custom_primitive=True)
        return jnp.sum(a) + jnp.sum(b)

    jax.jit(jax.grad(f, 0))(w, x)
