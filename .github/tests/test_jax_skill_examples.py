# SPDX-License-Identifier: Apache-2.0
"""Execute the published CPU recipes and exercise their layout helpers."""

import importlib
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cuequivariance as cue

SKILL = Path(__file__).resolve().parents[2] / "skills/cuequivariance-jax/SKILL.md"
EXAMPLES = [
    compile(code, f"{SKILL}:example-{index}", "exec")
    for index, code in enumerate(
        re.findall(r"```python\n(.*?)\n```", SKILL.read_text(), re.S)
    )
]


def run_example(index):
    namespace = {}
    exec(EXAMPLES[1], namespace)  # Common imports and precision configuration.
    exec(EXAMPLES[index], namespace)
    return namespace


@pytest.mark.parametrize("index", range(2, len(EXAMPLES)))
def test_published_cpu_example(index):
    # Run each example independently: accidental dependencies on earlier local
    # variables must fail here, just as they would for someone copying a recipe.
    run_example(index)


@pytest.mark.parametrize(
    "failed_module,error",
    [
        ("jax", ImportError("JAX cannot be imported")),
        ("cuequivariance", OSError("shared library cannot be loaded")),
        ("cuequivariance_jax", ModuleNotFoundError("frontend is absent")),
    ],
)
def test_environment_probe_preserves_partial_results(monkeypatch, failed_module, error):
    modules = {
        name: importlib.import_module(name)
        for name in ("jax", "cuequivariance", "cuequivariance_jax")
    }

    def import_module(name):
        if name == failed_module:
            raise error
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", import_module)
    namespace = {}
    exec(EXAMPLES[0], namespace)
    assert namespace["available"] == {
        name: module for name, module in modules.items() if name != failed_module
    }
    assert namespace["import_errors"] == {
        failed_module: f"{type(error).__name__}: {error}"
    }


@pytest.mark.parametrize("batch", [(), (2, 3)])
def test_layout_helpers_with_duplicate_irreps(batch):
    recipe = run_example(3)
    irreps = cue.Irreps("SO3", "2x1 + 0 + 1 + 2x2")
    flat = jnp.arange(int(np.prod(batch)) * irreps.dim).reshape(batch + (irreps.dim,))
    regrouped, blocks = recipe["ir_mul_to_dict"](irreps, flat)
    assert regrouped == irreps.regroup()
    # Combine separated l=1 blocks along the channel axis, keeping every value.
    expected_vectors = np.concatenate(
        [
            np.asarray(flat[..., :6]).reshape(batch + (3, 2)),
            np.asarray(flat[..., 7:10]).reshape(batch + (3, 1)),
        ],
        axis=-1,
    ).swapaxes(-1, -2)
    np.testing.assert_array_equal(blocks[cue.SO3(0)], flat[..., 6:7, None])
    np.testing.assert_array_equal(blocks[cue.SO3(1)], expected_vectors)
    np.testing.assert_array_equal(
        blocks[cue.SO3(2)],
        np.asarray(flat[..., 10:]).reshape(batch + (5, 2)).swapaxes(-1, -2),
    )
    expected_flat = np.concatenate(
        [
            flat[..., 6:7],
            expected_vectors.swapaxes(-1, -2).reshape(batch + (9,)),
            flat[..., 10:],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(
        recipe["dict_to_ir_mul_flat"](regrouped, blocks), expected_flat
    )


def test_layout_regroup_keeps_opposite_parities_separate():
    recipe = run_example(3)
    irreps = cue.Irreps("O3", "1o + 0e + 2x1e + 2x1o")
    flat = jnp.arange(2 * irreps.dim).reshape(2, irreps.dim)
    regrouped, blocks = recipe["ir_mul_to_dict"](irreps, flat)
    assert regrouped == irreps.regroup()
    assert set(blocks) == {cue.O3(0, 1), cue.O3(1, -1), cue.O3(1, 1)}
    odd_vectors = np.concatenate(
        [
            np.asarray(flat[:, :3]).reshape(2, 3, 1),
            np.asarray(flat[:, 10:]).reshape(2, 3, 2),
        ],
        axis=-1,
    ).swapaxes(-1, -2)
    np.testing.assert_array_equal(blocks[cue.O3(1, -1)], odd_vectors)
    np.testing.assert_array_equal(
        blocks[cue.O3(1, 1)],
        np.asarray(flat[:, 4:10]).reshape(2, 3, 2).swapaxes(-1, -2),
    )
    np.testing.assert_array_equal(blocks[cue.O3(0, 1)], flat[:, 3:4, None])


@pytest.mark.parametrize("batch_shapes", [((), (), ()), ((), (2, 1, 1), (1, 3, 2))])
def test_split_recipe_with_parity_higher_degrees_and_jit(batch_shapes):
    split_naive = run_example(5)["split_naive"]
    desc = cue.descriptors.channelwise_tensor_product_ir_dict(
        cue.Irreps("O3", "2x0e + 2x1o + 2x2e"),
        cue.Irreps("O3", "0e + 1o + 2e"),
        [cue.O3(degree, (-1) ** degree) for degree in range(4)],
    )
    rng = np.random.default_rng(41)
    groups = [
        {ir: jnp.asarray(rng.normal(size=batch + (mul, ir.dim))) for mul, ir in irreps}
        for irreps, batch in zip(desc.input_irreps, batch_shapes)
    ]
    operands = [
        np.asarray(group[ir]).swapaxes(-1, -2).reshape(batch + (mul * ir.dim,))
        for group, irreps, batch in zip(groups, desc.input_irreps, batch_shapes)
        for mul, ir in irreps
    ]
    expected = desc.polynomial(*operands)
    batch = np.broadcast_shapes(*batch_shapes)
    for compute in (
        lambda g: split_naive(desc, g),
        jax.jit(lambda g: split_naive(desc, g)),
    ):
        (result,) = compute(groups)
        assert set(result) == {ir for _, ir in desc.output_irreps[0]}
        for (mul, ir), reference in zip(desc.output_irreps[0], expected):
            assert result[ir].shape == batch + (mul, ir.dim)
            np.testing.assert_allclose(
                result[ir].swapaxes(-1, -2).reshape(batch + (mul * ir.dim,)),
                reference,
                rtol=1e-12,
                atol=1e-12,
            )
