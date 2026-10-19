# SPDX-License-Identifier: Apache-2.0
"""Pre-agent environment failures must stop evaluation without replacing data."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("frontend", ["jax", "torch"])
def test_pre_agent_setup_preserves_existing_incompatible_environment(
    tmp_path, frontend
):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / f"cuequivariance-{frontend}"
        / "evals"
        / "config.yml"
    )
    config = yaml.safe_load(config_path.read_text())
    env_dir = tmp_path / ".venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(env_dir)],
        check=True,
        timeout=30,
    )
    sentinel = env_dir / "keep.txt"
    sentinel.write_text("existing environment")
    original_config = (env_dir / "pyvenv.cfg").read_bytes()
    command = "set -euo pipefail\n" + "\n".join(config["harbor"]["pre_agent_setup"])
    result = subprocess.run(
        ["bash", "-c", command],
        env={**os.environ, "CUEQ_EVAL_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "CPU environment ready" not in result.stdout
    assert sentinel.read_text() == "existing environment"
    assert (env_dir / "pyvenv.cfg").read_bytes() == original_config
    assert not list(tmp_path.glob(".cueq-env.*"))
