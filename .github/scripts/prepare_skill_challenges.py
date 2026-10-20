# SPDX-License-Identifier: Apache-2.0
"""Stage challenge-v1 pilots without changing canonical skills or regression evals.

Run with Python + PyYAML. --export-harbor additionally needs SkillEvaluator 1.5.6
or a compatible builder that separates verifier material from runtime skills.
All output is created in a fresh directory; this command never launches agents.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shlex
import shutil
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GRADER = ROOT / ".github/evals/cueq_challenges.py"
# Match the models resolved in the user's 2026-10-20 comparison.
MODELS = {
    "claude-code": "aws/anthropic/bedrock-claude-opus-4-8",
    "codex": "openai/openai/gpt-5.5",
}


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verify_environment(task, backend, with_skill):
    """Fail the export if evaluator or reference material reaches the agent."""
    environment = task / "environment"
    for path in environment.rglob("*"):
        relative = path.relative_to(environment)
        assert "evals" not in relative.parts, relative
        assert path.name != "grader.py", relative
        assert not path.name.startswith("reference_"), relative
        assert path.name not in ("entry.json", "challenges.json"), relative
    skill = environment / "skills" / f"cuequivariance-{backend}" / "SKILL.md"
    assert skill.exists() == with_skill
    if with_skill:
        assert digest(skill) == digest(ROOT / "skills" / skill.parent.name / "SKILL.md")
    assert (task / "tests/grader.py").read_bytes() == GRADER.read_bytes()
    entry = json.loads((task / "tests/entry.json").read_text())
    assert entry["has_skill"] == with_skill


def prepare(destination, backends, *, export_harbor=False):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    if export_harbor:
        from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

        if (
            "evaluator_skill_path"
            not in inspect.signature(generate_harbor_tasks).parameters
        ):
            raise RuntimeError(
                "Use the SkillEvaluator 1.5.6 builder with private evaluator-source support"
            )
    manifest = {
        "benchmark": "cuequivariance-challenges-v1",
        "attempts_per_case_per_arm": 3,
        "stop_on_pass": False,
        "agents": ["claude-code", "codex"],
        "agent_models": MODELS,
        "grading": "aces_plus_custom",
        "custom_metric": "numerical_success",
        "custom_metric_changes_aces_gate": False,
        "skills": {},
    }
    for backend in backends:
        name = f"cuequivariance-{backend}"
        source, staged = ROOT / "skills" / name, destination / "skills" / name
        staged.mkdir(parents=True)
        shutil.copy2(source / "SKILL.md", staged / "SKILL.md")
        # Only runtime resources and the challenge evaluator are staged.
        for directory in ("scripts", "references", "assets", "agents"):
            if (source / directory).is_dir():
                shutil.copytree(source / directory, staged / directory)
        evals = staged / "evals"
        evals.mkdir()
        shutil.copy2(source / "evals/challenges.json", evals / "evals.json")
        shutil.copy2(GRADER, evals / "grader.py")
        config = yaml.safe_load((source / "evals/config.yml").read_text())
        config["grading"]["mode"] = "aces_plus_custom"
        (evals / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        command = [
            "skillevaluator",
            "tier3",
            "evaluate",
            str(staged),
            "--env-mode",
            "k8s-sandbox",
            "--agents",
            "claude-code,codex",
            "--agent-model",
            f"claude-code={MODELS['claude-code']}",
            "--agent-model",
            f"codex={MODELS['codex']}",
            "--n-attempts",
            "3",
            "--no-stop-on-pass",
            "--n-concurrent",
            "1",
            "--max-agents",
            "1",
            "--harbor-keep-jobs",
            "--results-dir",
            str(destination / "results" / name),
        ]
        record = {
            "path": str(staged),
            "skill_digest": digest(staged / "SKILL.md"),
            "dataset_digest": digest(evals / "evals.json"),
            "config_digest": digest(evals / "config.yml"),
            "grader_digest": digest(evals / "grader.py"),
            "case_ids": [
                e["id"] for e in json.loads((evals / "evals.json").read_text())["evals"]
            ],
            "command": command,
        }
        if export_harbor:
            tasks_by_arm = {}
            for arm, with_skill in (("with_skill", True), ("baseline", False)):
                task_root = destination / "harbor" / backend / arm
                tasks = generate_harbor_tasks(
                    staged,
                    task_root,
                    evaluator_skill_path=staged,
                    with_skill=with_skill,
                    workspace_mode="isolated",
                    grading_mode="aces_plus_custom",
                    env_mode="astra-sandbox",
                    runtime_env=config["harbor"]["runtime_env"],
                    pre_agent_setup=config["harbor"]["pre_agent_setup"],
                    agent_workdir="/workspace",
                    copy_repo=False,
                    repo_context_exclude_paths=[ROOT / ".github/evals"],
                )
                for task in tasks:
                    verify_environment(task, backend, with_skill)
                tasks_by_arm[arm] = tasks
            # Inputs and the verifier must be identical in the two conditions.
            for with_task, without_task in zip(
                tasks_by_arm["with_skill"], tasks_by_arm["baseline"], strict=True
            ):
                assert (with_task / "instruction.md").read_bytes() == (
                    without_task / "instruction.md"
                ).read_bytes()
                assert (with_task / "tests/grader.py").read_bytes() == (
                    without_task / "tests/grader.py"
                ).read_bytes()
            record["verified_harbor_tasks"] = sum(map(len, tasks_by_arm.values()))
        manifest["skills"][name] = record
    (destination / "pilot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "torch", "both"), default="both")
    parser.add_argument(
        "--output", type=Path, help="New directory; existing paths are refused."
    )
    parser.add_argument("--export-harbor", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        root = ROOT / ".venv"
        root.mkdir(exist_ok=True)
        # The parent is unique; prepare owns a new, initially absent child.
        args.output = (
            Path(tempfile.mkdtemp(prefix="cueq-challenges-v1-", dir=root)) / "pilot"
        )
    backends = ("jax", "torch") if args.backend == "both" else (args.backend,)
    manifest = prepare(args.output, backends, export_harbor=args.export_harbor)
    print(f"Pilot manifest: {args.output.resolve() / 'pilot.json'}")
    for record in manifest["skills"].values():
        print(shlex.join(record["command"]))


if __name__ == "__main__":
    main()
