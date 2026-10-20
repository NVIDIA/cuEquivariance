# SPDX-License-Identifier: Apache-2.0
"""Pre-agent environment failures must stop evaluation without replacing data."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def setup_command(frontend):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / f"cuequivariance-{frontend}"
        / "evals"
        / "config.yml"
    )
    config = yaml.safe_load(config_path.read_text())
    return "set -euo pipefail\n" + "\n".join(config["harbor"]["pre_agent_setup"])


@pytest.fixture
def setup_sandbox(tmp_path):
    """Inject installer/smoke failures without downloading dependencies."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        """
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["CUEQ_EVAL_ROOT"])
args = sys.argv[1:]
if args[:2] == ["-m", "venv"]:
    stage = "venv"
    target = Path(args[-1])
    (target / "bin").mkdir()
    (target / "bin/python").symlink_to(Path(__file__).resolve())
elif args[:2] == ["-m", "pip"]:
    stage = "install"
else:
    stage = "smoke"
    sys.stdin.read()

with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps({
        "stage": stage, "args": args, "published": (root / ".venv").is_symlink()
    }) + "\\n")
if os.environ.get("CUEQ_FAIL_STAGE") == stage:
    print("injected " + stage + " failure", file=sys.stderr)
    sys.exit(23)
if stage == "smoke":
    print("CPU environment ready")
"""
    )
    python.chmod(0o755)
    return {
        **os.environ,
        "CUEQ_EVAL_ROOT": str(tmp_path),
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
    }


@pytest.mark.parametrize("frontend", ["jax", "torch"])
@pytest.mark.parametrize("failure_stage", ["venv", "install", "smoke"])
def test_failed_setup_is_not_published_and_can_be_retried(
    tmp_path, setup_sandbox, frontend, failure_stage
):
    def run(**extra):
        return subprocess.run(
            ["bash", "-c", setup_command(frontend)],
            env={**setup_sandbox, **extra},
            capture_output=True,
            text=True,
            timeout=30,
        )

    failed = run(CUEQ_FAIL_STAGE=failure_stage)
    assert failed.returncode == 23, failed.stderr
    assert f"injected {failure_stage} failure" in failed.stderr
    assert not os.path.lexists(tmp_path / ".venv")
    (partial,) = tmp_path.glob(".cueq-env.*")
    sentinel = partial / "keep.txt"
    sentinel.write_text("partial environment")

    retried = run(CUEQ_FAIL_STAGE="")
    assert retried.returncode == 0, retried.stderr
    assert "CPU environment ready" in retried.stdout
    published = tmp_path / ".venv"
    assert published.is_symlink()
    assert published.resolve() != partial
    assert sentinel.read_text() == "partial environment"
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert not any(call["published"] for call in calls)

    # Reuse checks the environment but does not reinstall or replace it.
    target = published.resolve()
    reused = run(CUEQ_FAIL_STAGE="install")
    assert reused.returncode == 0, reused.stderr
    assert published.resolve() == target
    after = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert [call["stage"] for call in after[len(calls) :]] == ["smoke"]


@pytest.mark.parametrize("frontend", ["jax", "torch"])
def test_pre_agent_setup_preserves_dangling_environment_link(
    tmp_path, setup_sandbox, frontend
):
    env_dir = tmp_path / ".venv"
    target = tmp_path / "missing-environment"
    env_dir.symlink_to(target)
    result = subprocess.run(
        ["bash", "-c", setup_command(frontend)],
        env=setup_sandbox,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert env_dir.is_symlink() and env_dir.readlink() == target
    assert not list(tmp_path.glob(".cueq-env.*"))
    assert not (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize("frontend", ["jax", "torch"])
def test_pre_agent_setup_preserves_existing_incompatible_environment(
    tmp_path, frontend
):
    env_dir = tmp_path / ".venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(env_dir)],
        check=True,
        timeout=30,
    )
    sentinel = env_dir / "keep.txt"
    sentinel.write_text("existing environment")
    original_config = (env_dir / "pyvenv.cfg").read_bytes()
    result = subprocess.run(
        ["bash", "-c", setup_command(frontend)],
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
