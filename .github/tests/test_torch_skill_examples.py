# SPDX-License-Identifier: Apache-2.0
"""Run each published recipe independently and check the trainable GNN."""

import importlib
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import cuequivariance as cue

SKILL = Path(__file__).resolve().parents[2] / "skills/cuequivariance-torch/SKILL.md"
EXAMPLES = [
    compile(code, f"{SKILL}:example-{index}", "exec")
    for index, code in enumerate(
        re.findall(r"```python\n(.*?)\n```", SKILL.read_text(), re.S)
    )
]


@pytest.mark.parametrize("index", range(2, len(EXAMPLES) - 1))
def test_published_cpu_example(index):
    namespace = {}
    exec(EXAMPLES[1], namespace)
    exec(EXAMPLES[index], namespace)


@pytest.mark.parametrize(
    "failed_module, error",
    [
        ("cuequivariance_torch", ModuleNotFoundError("cuEquivariance is absent")),
        ("cuequivariance", OSError("shared library cannot be loaded")),
        ("torch", ImportError("PyTorch cannot be imported")),
    ],
)
def test_environment_probe_preserves_partial_results(monkeypatch, failed_module, error):
    modules = {
        name: importlib.import_module(name)
        for name in ("torch", "cuequivariance", "cuequivariance_torch")
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


def test_documented_gnn_is_trainable_and_equivariant():
    namespace = {}
    exec(EXAMPLES[1], namespace)
    exec(EXAMPLES[-1], namespace)
    torch.manual_seed(41)
    irreps_in = cue.Irreps("SO3", "2x0+2x1")
    irreps_out = cue.Irreps("SO3", "3x0+1")
    model = namespace["SimpleGNN"](irreps_in, irreps_out).double()
    features = torch.randn(5, irreps_in.dim, dtype=torch.float64, requires_grad=True)
    vectors = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    senders = torch.tensor([0, 1, 3, 0, 1, 2])
    receivers = torch.tensor([2, 0, 2, 1, 2, 0])
    output = model(features, vectors, senders, receivers, 4)
    assert output.shape == (4, irreps_out.dim)
    assert torch.count_nonzero(output[3]) == 0
    torch.testing.assert_close(
        output, model(features, vectors, senders, receivers, 4), rtol=0, atol=0
    )

    rotation = np.array([0.2, -0.3, 0.1])
    din, dout = [
        torch.from_numpy(cue.IrrepsAndLayout(r, cue.mul_ir).exp_map(rotation, []))
        for r in (irreps_in, irreps_out)
    ]
    dvector = torch.from_numpy(cue.SO3(1).exp_map(rotation, []))
    rotated = model(features @ din.T, vectors @ dvector.T, senders, receivers, 4)
    torch.testing.assert_close(rotated, output @ dout.T, rtol=1e-7, atol=1e-9)

    output.square().sum().backward()
    for name, tensor in [
        *model.named_parameters(),
        ("features", features),
        ("vectors", vectors),
    ]:
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name
