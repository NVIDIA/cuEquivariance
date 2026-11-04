# SPDX-License-Identifier: Apache-2.0
"""Shared checks of the challenge verifier, references, and deliberate defects."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

EVALS = Path(__file__).resolve().parents[1] / "evals"
spec = importlib.util.spec_from_file_location(
    "cueq_challenge_verifier", EVALS / "cueq_challenges.py"
)
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def reference(backend):
    return grader.load_solution(EVALS / f"reference_{backend}_challenges.py")


def run_mutation(backend, defect):
    original = reference(backend)
    if defect in ("overwrite_duplicate", "reshape_only", "merge_parity"):

        def broken(irreps, flat):
            if defect == "merge_parity":
                return {
                    ir.l: value
                    for ir, value in original.to_blocks(irreps, flat).items()
                }
            if defect == "reshape_only":
                return {
                    ir: value.swapaxes(-1, -2).reshape(value.shape)
                    for ir, value in original.to_blocks(irreps, flat).items()
                }
            blocks, offset = {}, 0
            for mul, ir in irreps:
                block = flat[..., offset : offset + mul * ir.dim]
                block = block.reshape(flat.shape[:-1] + (ir.dim, mul))
                block = block.swapaxes(-1, -2)
                blocks[ir] = block
                offset += mul * ir.dim
            return blocks

        mutant = SimpleNamespace(to_blocks=broken, from_blocks=original.from_blocks)
        with pytest.raises((AssertionError, KeyError, ValueError)):
            grader.check_layout(mutant, backend)
        return
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        def broken(params, x, v, s, r, n):
            if defect == "shared_radial":
                v = jnp.broadcast_to(v[0], v.shape)
            if defect == "detached_vectors":
                v = jax.lax.stop_gradient(v)
            output = original.apply(params, x, v, s, r, n)
            if defect == "mean_reduction":
                # Average repeated destinations instead of summing them.
                counts = jnp.bincount(r, length=n).clip(1)[:, None]
                output = output / counts
            if defect == "wrong_layout":
                output = original.apply(params, x[:, ::-1], v, s, r, n)
            return output

        mutant = SimpleNamespace(apply=broken)
    else:
        import torch

        class Broken(original.GraphLayer):
            def forward(self, x, v, s, r, n):
                if defect == "shared_radial":
                    v = v[0].expand_as(v)
                if defect == "detached_vectors":
                    v = v.detach()
                output = super().forward(x, v, s, r, n)
                if defect == "mean_reduction":
                    output = (
                        output / torch.bincount(r, minlength=n).clamp_min(1)[:, None]
                    )
                if defect == "wrong_layout":
                    output = super().forward(x.flip(-1), v, s, r, n)
                return output

        mutant = SimpleNamespace(GraphLayer=Broken)
    with pytest.raises((AssertionError, RuntimeError)):
        grader.check_graph(mutant, backend)


def check_routing(backend, tmp_path):
    path = tmp_path / "solution.py"
    path.write_text(
        "def mse_loss(predictions, targets):\n"
        "    return ((predictions[:, 0] - targets) ** 2).mean()\n"
    )
    grader.check_routing(grader.load_solution(path), backend, path)
    # Shape-valid broadcasting is precisely the bug this negative case asks to fix.
    path.write_text(
        "def mse_loss(predictions, targets):\n"
        "    return ((predictions - targets) ** 2).mean()\n"
    )
    with pytest.raises(AssertionError):
        grader.check_routing(grader.load_solution(path), backend, path)
