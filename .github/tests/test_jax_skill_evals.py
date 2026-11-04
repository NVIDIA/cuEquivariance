# SPDX-License-Identifier: Apache-2.0
"""CPU API and numerical contracts for the JAX skill evaluation prompts."""

import cuequivariance_jax.ir_dict as ir_dict
import cuequivariance_jax.nnx as cue_nnx
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import cuequivariance as cue
import cuequivariance_jax as cuex

jax.config.update("jax_enable_x64", True)


def test_broadcast_jit_and_finite_difference():
    e = cue.descriptors.channelwise_tensor_product(
        cue.Irreps("SO3", "2x0+2x1"),
        cue.Irreps("SO3", "0+1"),
        [cue.SO3(0), cue.SO3(1)],
    )
    rng = np.random.default_rng(21)
    w = rng.normal(size=e.inputs[0].dim)
    x = rng.normal(size=(2, 1, e.inputs[1].dim))
    y = rng.normal(size=(1, 3, e.inputs[2].dim))

    def run(weights):
        return cuex.segmented_polynomial(
            e.polynomial,
            [weights, jnp.asarray(x), jnp.asarray(y)],
            [jax.ShapeDtypeStruct((2, 3, e.outputs[0].dim), jnp.float64)],
            method="naive",
        )[0]

    expected = e(w, x, y)[0]
    np.testing.assert_allclose(run(jnp.asarray(w)), expected, atol=1e-12)
    np.testing.assert_allclose(jax.jit(run)(jnp.asarray(w)), expected, atol=1e-12)
    grad = jax.grad(lambda v: jnp.sum(run(v) ** 2))(jnp.asarray(w))
    eps = 1e-5
    wp, wm = w.copy(), w.copy()
    wp[0] += eps
    wm[0] -= eps
    finite_diff = (np.sum(e(wp, x, y)[0] ** 2) - np.sum(e(wm, x, y)[0] ** 2)) / (
        2 * eps
    )
    assert abs(grad[0]) > 1e-5
    np.testing.assert_allclose(grad[0], finite_diff, rtol=1e-7, atol=1e-7)


def test_dict_layout_conversion_is_not_a_flat_reshape():
    irreps = cue.Irreps("SO3", "2x0+3x1")
    flat = jnp.arange(2 * irreps.dim).reshape(2, -1)
    blocks = ir_dict.flat_to_dict(irreps, flat, layout="ir_mul")
    ir_dict.assert_mul_ir_dict(irreps, blocks)
    np.testing.assert_array_equal(
        blocks[cue.SO3(1)], np.asarray(flat[:, 2:]).reshape(2, 3, 3).swapaxes(-1, -2)
    )
    mul_ir = ir_dict.dict_to_flat(irreps, blocks)
    assert not np.array_equal(mul_ir, flat)
    restored = jnp.concatenate(
        [blocks[ir].swapaxes(-1, -2).reshape(2, -1) for _, ir in irreps], axis=-1
    )
    np.testing.assert_array_equal(restored, flat)


def test_nnx_linear_normalization_missing_irrep_and_equivariance():
    a, b = cue.Irreps("SO3", "2x0+3x1"), cue.Irreps("SO3", "4x0+2x1+2")
    layer = cue_nnx.IrrepsLinear(a, b, dtype=jnp.float64, rngs=nnx.Rngs(22))
    x = {
        ir: jnp.asarray(np.random.default_rng(23 + ir.l).normal(size=(2, mul, ir.dim)))
        for mul, ir in a
    }
    y = layer(x)
    ir_dict.assert_mul_ir_dict(b, y)
    np.testing.assert_array_equal(y[cue.SO3(2)], 0)
    for mul, ir in a:
        expected = np.einsum(
            "uv,bui->bvi", np.asarray(layer.w[str(ir)][...]), x[ir]
        ) / np.sqrt(mul)
        np.testing.assert_allclose(y[ir], expected, atol=1e-12)
    rotation = np.array([0.2, -0.3, 0.4])
    rotated = layer({ir: v @ ir.exp_map(rotation, []).T for ir, v in x.items()})
    for ir in y:
        np.testing.assert_allclose(
            rotated[ir], y[ir] @ ir.exp_map(rotation, []).T, atol=1e-12
        )


def test_ir_dict_naive_matches_numpy():
    e = cue.descriptors.channelwise_tensor_product_ir_dict(
        cue.Irreps("SO3", "2x0+2x1+2x2"),
        cue.Irreps("SO3", "0+1+2+3"),
        [cue.SO3(degree) for degree in range(3)],
    )
    rng = np.random.default_rng(24)
    inputs = [
        rng.normal(size=(2, mul, ir.dim)).swapaxes(-1, -2).reshape(2, -1)
        for group in e.input_irreps
        for mul, ir in group
    ]
    result = cuex.segmented_polynomial(
        e.polynomial,
        list(map(jnp.asarray, inputs)),
        [
            jax.ShapeDtypeStruct((2, op.size), jnp.float64)
            for op in e.polynomial.outputs
        ],
        method="naive",
    )
    expected = e.polynomial(*inputs)
    for actual, reference in zip(result, expected):
        np.testing.assert_allclose(actual, reference, atol=1e-12)


def test_nnx_spherical_harmonics_cpu():
    layer = cue_nnx.SphericalHarmonics(max_degree=2)
    output = layer(jnp.array([[1.0, 2.0, 3.0], [-0.4, 0.8, 0.2]]))
    assert {ir.l: value.shape for ir, value in output.items()} == {
        0: (2, 1, 1),
        1: (2, 1, 3),
        2: (2, 1, 5),
    }
    assert all(np.isfinite(value).all() for value in output.values())
