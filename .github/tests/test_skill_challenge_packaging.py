# SPDX-License-Identifier: Apache-2.0
"""Preserve regression inputs, isolate reference answers, and bound verification."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import tomllib
import yaml
from challenge_checks import EVALS, grader

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prepare_skill_challenges", ROOT / ".github/scripts/prepare_skill_challenges.py"
)
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


def test_pilot_keeps_skill_and_environment_fixed_and_excludes_answers(tmp_path):
    original = {}
    for backend in ("jax", "torch"):
        source = ROOT / "skills" / f"cuequivariance-{backend}"
        for relative in ("SKILL.md", "evals/config.yml", "evals/evals.json"):
            original[source / relative] = (source / relative).read_bytes()
    manifest = preparer.prepare(tmp_path / "pilot", ("jax", "torch"))
    assert manifest["attempts_per_case_per_arm"] == 3
    assert manifest["stop_on_pass"] is False
    for backend in ("jax", "torch"):
        name = f"cuequivariance-{backend}"
        record = manifest["skills"][name]
        source, staged = ROOT / "skills" / name, Path(record["path"])
        assert (staged / "SKILL.md").read_bytes() == (source / "SKILL.md").read_bytes()
        config = yaml.safe_load((staged / "evals/config.yml").read_text())
        original_config = yaml.safe_load((source / "evals/config.yml").read_text())
        assert config["harbor"] == original_config["harbor"]
        assert config["grading"]["mode"] == "aces_plus_custom"
        entries = json.loads((staged / "evals/evals.json").read_text())["evals"]
        assert len(entries) == 3
        assert len({e["id"] for e in entries}) == len(entries)
        assert sum(e["expected_skill"] == name for e in entries) == 2
        assert sum(e["expected_skill"] is None for e in entries) == 1
        assert all("v1" in e["id"] for e in entries)
        assert not list(staged.rglob("reference_*"))
        assert preparer.digest(staged / "evals/grader.py") == record["grader_digest"]
        assert record["command"][record["command"].index("--n-attempts") + 1] == "3"
        assert "--skip-baseline" not in record["command"]
        for agent, model in manifest["agent_models"].items():
            assert f"{agent}={model}" in record["command"]
        if backend == "torch":
            assert (staged / "evals/setup_cpu.sh").read_bytes() == (
                source / "evals/setup_cpu.sh"
            ).read_bytes()
            tasks = staged / "evals/harbor"
            assert {p.name for p in tasks.iterdir() if p.is_dir()} == {
                e["id"] for e in entries
            }
            dataset = tomllib.loads((tasks / "dataset.toml").read_text())
            assert {t["name"] for t in dataset["tasks"]} == {
                f"nvidia/{e['id']}" for e in entries
            }
            for entry in entries:
                task = tasks / entry["id"]
                assert (task / "instruction.md").read_text() == entry["prompt"] + "\n"
                data = tomllib.loads((task / "task.toml").read_text())
                assert data["environment"]["healthcheck"]["timeout_sec"] == 600
            assert record["setup_digest"] == preparer.digest(
                staged / "evals/setup_cpu.sh"
            )
            assert record["native_task_digests"]
    for path, content in original.items():
        assert path.read_bytes() == content
    with pytest.raises(FileExistsError):
        preparer.prepare(tmp_path / "pilot", ("jax",))


def test_missing_submission_and_timeout_are_failures(tmp_path):
    missing = grader.evaluate(
        "jax", "layout", tmp_path / "missing.py", python=sys.executable
    )
    assert not missing["passed"] and "FileNotFoundError" in missing["error"]
    hanging = tmp_path / "solution.py"
    hanging.write_text("while True: pass\n")
    timed_out = grader.evaluate("jax", "layout", hanging, timeout=0.2)
    assert not timed_out["passed"] and "exceeded" in timed_out["error"]


def test_submission_import_supports_sibling_helpers_and_dataclasses(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    (tmp_path / "challenge_import_helper.py").write_text("VALUE = 7\n")
    solution = tmp_path / "solution.py"
    solution.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from challenge_import_helper import VALUE\n"
        "@dataclass\n"
        "class Result:\n"
        "    value: int = VALUE\n"
    )
    assert grader.load_solution(solution).Result().value == 7


@pytest.mark.parametrize("leak", ("evals/evals.json", "grader.py", "reference_jax.py"))
def test_export_rejects_evaluation_material_in_agent_bundle(tmp_path, leak):
    task = tmp_path / "task"
    path = task / "environment" / leak
    path.parent.mkdir(parents=True)
    path.write_text("must not be visible to the agent")
    with pytest.raises(AssertionError):
        preparer.verify_environment(task, "jax", False)


def test_harbor_dispatch_writes_a_real_failure_reward(tmp_path, monkeypatch):
    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps({"id": "jax-challenge-layout-v1"}))
    reward = tmp_path / "reward.json"
    monkeypatch.setenv("HARBOR_ENTRY_JSON", str(entry))
    monkeypatch.setenv("HARBOR_REWARD_JSON", str(reward))
    monkeypatch.setattr(sys, "argv", [str(EVALS / "cueq_challenges.py")])
    observed = {}

    def failed(backend, case, solution, **kwargs):
        observed.update(backend=backend, case=case, solution=str(solution), **kwargs)
        return {"passed": False, "error": "numerical mismatch"}

    monkeypatch.setattr(grader, "evaluate", failed)
    grader.main()
    assert observed["backend"] == "jax" and observed["case"] == "layout"
    assert observed["solution"] == "/workspace/solution.py"
    assert observed["python"] == "/workspace/.venv/bin/python"
    result = json.loads(reward.read_text())
    assert result["custom_metrics"]["numerical_success"] == 0
    assert "overall" not in result and "goal_accuracy" not in result
