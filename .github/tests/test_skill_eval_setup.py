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
failure = os.environ.get("CUEQ_FAIL_STAGE")
if failure == stage or (
    failure == "dependencies" and stage == "install" and "--no-deps" not in args
):
    print("injected " + failure + " failure", file=sys.stderr)
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


@pytest.mark.parametrize(
    ("failure", "reported_stage"),
    [
        ("venv", "create_environment"),
        ("install", "install_cpu_torch"),
        ("dependencies", "install_dependencies"),
        ("smoke", "smoke_check"),
    ],
)
def test_torch_setup_identifies_the_failed_stage(
    tmp_path, setup_sandbox, failure, reported_stage
):
    result = subprocess.run(
        ["bash", "-c", setup_command("torch")],
        env={**setup_sandbox, "CUEQ_FAIL_STAGE": failure},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 23
    assert f"injected {failure} failure" in result.stderr
    assert result.stderr.rstrip().endswith(
        f"CUEQ_SETUP_FAILED stage={reported_stage} exit_code=23"
    )
    assert not os.path.lexists(tmp_path / ".venv")


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
@pytest.mark.parametrize("logging_fails", [False, True])
def test_native_setup_retains_output_and_exit_status(tmp_path, status, logging_fails):
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
    # A previous successful attempt must not hide the next attempt's failure.
    status_file = tmp_path / "cueq-setup.status"
    status_file.write_text("0\n")
    log_file = tmp_path / "cueq-setup.log"
    if logging_fails:
        log_file.mkdir()
    env = {**os.environ, "CUEQ_EVAL_ROOT": str(tmp_path)}
    result = subprocess.run(
        shlex.split(command),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected_status = status or (1 if logging_fails else 0)
    assert result.returncode == expected_status
    assert "installer output" in result.stdout
    assert "installer diagnostic" in result.stdout
    if not logging_fails:
        assert log_file.read_text() == result.stdout
    assert status_file.read_text() == f"{expected_status}\n"

    (hook,) = data["verifier"]["collect"]
    diagnostics = subprocess.run(
        shlex.split(hook["command"]),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    # Harbor logs the stdout/stderr of a nonzero collection hook even when the
    # healthcheck failed before agent execution. Success produces no warning.
    assert diagnostics.returncode == expected_status
    if expected_status:
        assert (
            f"CUEQ_SETUP_DIAGNOSTICS exit_code={expected_status}" in diagnostics.stderr
        )
        if logging_fails:
            assert "No setup log was produced" in diagnostics.stderr
        else:
            assert "installer diagnostic" in diagnostics.stderr
    else:
        assert not diagnostics.stdout and not diagnostics.stderr


@pytest.mark.parametrize("saved_status", [None, "125\n", "invalid\n"])
def test_setup_diagnostics_handle_missing_or_incomplete_status(tmp_path, saved_status):
    if saved_status is not None:
        (tmp_path / "cueq-setup.status").write_text(saved_status)
    result = subprocess.run(
        shlex.split(BUILDER["SETUP_DIAGNOSTICS_COMMAND"]),
        env={**os.environ, "CUEQ_EVAL_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 125
    assert "CUEQ_SETUP_DIAGNOSTICS exit_code=125" in result.stderr
    assert "No setup log was produced" in result.stderr


def test_setup_diagnostics_keep_the_error_at_the_end_of_long_logs(tmp_path):
    (tmp_path / "cueq-setup.status").write_text("23\n")
    (tmp_path / "cueq-setup.log").write_text(
        "irrelevant download progress\n" * 1000
        + "ERROR: injected dependency failure\n"
        + "CUEQ_SETUP_FAILED stage=install_dependencies exit_code=23\n"
    )
    result = subprocess.run(
        shlex.split(BUILDER["SETUP_DIAGNOSTICS_COMMAND"]),
        env={**os.environ, "CUEQ_EVAL_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 23
    assert "ERROR: injected dependency failure" in result.stderr
    assert "CUEQ_SETUP_FAILED stage=install_dependencies exit_code=23" in result.stderr
    assert len(result.stderr.splitlines()) <= 13
