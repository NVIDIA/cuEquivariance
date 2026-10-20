# SPDX-License-Identifier: Apache-2.0
"""Pre-agent environment failures must stop evaluation without replacing data."""

import json
import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[2]
BUILDER = runpy.run_path(str(ROOT / ".github/scripts/prepare_skill_eval_tasks.py"))


def setup_command(frontend):
    config_path = (
        ROOT / "skills" / f"cuequivariance-{frontend}" / "evals" / "config.yml"
    )
    config = yaml.safe_load(config_path.read_text())
    if config["harbor"].get("task_source") == "native_harbor":
        return (config_path.parent / "setup_cpu.sh").read_text()
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


def test_native_torch_tasks_keep_regression_prompts_and_setup_budget():
    skill = ROOT / "skills/cuequivariance-torch"
    BUILDER["write_tasks"](skill, check=True)
    config = yaml.safe_load((skill / "evals/config.yml").read_text())
    assert config["harbor"]["task_source"] == "native_harbor"
    # Otherwise 1.5.6 rejects the existing native healthcheck as a collision.
    assert "pre_agent_setup" not in config["harbor"]
    assert config["grading"]["mode"] == "aces_default"
    entries = json.loads((skill / "evals/evals.json").read_text())["evals"]
    tasks = skill / "evals/harbor"
    assert {p.name for p in tasks.iterdir() if p.is_dir()} == {e["id"] for e in entries}
    manifest = tomllib.loads((tasks / "dataset.toml").read_text())
    assert [t["name"] for t in manifest["tasks"]] == [
        f"nvidia/{case_id}" for case_id in sorted(e["id"] for e in entries)
    ]
    healthchecks = []
    for entry in entries:
        task = tasks / entry["id"]
        assert (task / "instruction.md").read_text() == entry["prompt"] + "\n"
        data = tomllib.loads((task / "task.toml").read_text())
        assert data["metadata"]["entry_id"] == entry["id"]
        assert data["agent"]["timeout_sec"] == 300
        assert data["verifier"]["timeout_sec"] == 180
        healthchecks.append(data["environment"]["healthcheck"])
        assert healthchecks[-1]["timeout_sec"] == 600
        assert healthchecks[-1]["retries"] == 1
    assert all(h == healthchecks[0] for h in healthchecks)


@pytest.mark.parametrize("status", [0, 23])
def test_native_setup_retains_output_and_exit_status(tmp_path, status):
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps({"evals": [{"id": "smoke", "prompt": "Exercise setup"}]})
    )
    (evals / "setup_cpu.sh").write_text(
        f"printf 'installer output\\n'\nprintf 'installer diagnostic\\n' >&2\nexit {status}\n"
    )
    files = BUILDER["task_files"](tmp_path)
    data = tomllib.loads(files[Path("smoke/task.toml")])
    command = data["environment"]["healthcheck"]["command"]
    result = subprocess.run(
        shlex.split(command),
        env={**os.environ, "CUEQ_EVAL_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == status
    assert "installer output" in result.stdout
    assert "installer diagnostic" in result.stdout
    assert (tmp_path / "cueq-setup.log").read_text() == result.stdout
