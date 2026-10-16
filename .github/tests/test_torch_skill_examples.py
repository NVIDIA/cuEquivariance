# SPDX-License-Identifier: Apache-2.0
"""Run the published skill examples and check the trainable GNN's invariants."""

import re
from pathlib import Path

import numpy as np
import pytest
import torch

import cuequivariance as cue


@pytest.fixture(scope="module")
def skill_namespace():
    skill = Path(__file__).resolve().parents[2] / "skills/cuequivariance-torch/SKILL.md"
    namespace = {}
    for index, code in enumerate(
        re.findall(r"```python\n(.*?)\n```", skill.read_text(), re.S)
    ):
        exec(compile(code, f"{skill}:example-{index}", "exec"), namespace)
    return namespace


def test_documented_gnn_is_trainable_and_equivariant(skill_namespace):
    torch.manual_seed(41)
    irreps_in = cue.Irreps("SO3", "2x0+2x1")
    irreps_out = cue.Irreps("SO3", "3x0+1")
    model = skill_namespace["SimpleGNN"](irreps_in, irreps_out).double()
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
