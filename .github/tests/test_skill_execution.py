# SPDX-License-Identifier: Apache-2.0
"""Exercise the documented completion commands, including failures and live runs."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(params=["cuequivariance-jax", "cuequivariance-torch"])
def completion_command(request):
    skill = Path(__file__).resolve().parents[2] / "skills" / request.param / "SKILL.md"
    section = skill.read_text().split("### Complete and inspect a run\n", 1)[1]
    return re.search(r"```bash\n(.*?)\n```", section, re.S).group(1)


def run_environment(tmp_path):
    return {**os.environ, "TMPDIR": str(tmp_path), "cueq_python": sys.executable}


def test_completion_records_success_and_failure_without_reusing_logs(
    completion_command, tmp_path
):
    project = tmp_path / "project with spaces"
    project.mkdir()
    sentinel = tmp_path / ".venv" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("existing environment")
    records = []
    for status in (0, 7):
        (project / "check.py").write_text(
            f"print('observed result for exit {status}')\nraise SystemExit({status})\n"
        )
        result = subprocess.run(
            ["bash", "-c", "set -e\n" + completion_command],
            cwd=project,
            env=run_environment(tmp_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (result.returncode == 0) == (status == 0), result
        record = Path(result.stdout.splitlines()[0].removeprefix("run directory: "))
        assert (record / "exit-code.txt").read_text().strip() == str(status)
        assert (
            record / "output.txt"
        ).read_text() == f"observed result for exit {status}\n"
        assert f"observed result for exit {status}" in result.stdout
        assert f"exit code: {status}" in result.stdout
        records.append(record)
    assert records[0] != records[1]
    assert (records[0] / "exit-code.txt").read_text().strip() == "0"
    assert sentinel.read_text() == "existing environment"


def test_live_run_has_no_completion_record(completion_command, tmp_path):
    (tmp_path / "check.py").write_text(
        "import time\nfrom pathlib import Path\n"
        "print('check started', flush=True)\n"
        "while not Path('release').exists(): time.sleep(0.01)\n"
        "print('check completed')\n"
    )
    with subprocess.Popen(
        ["bash", "-c", "set -e\n" + completion_command],
        cwd=tmp_path,
        env=run_environment(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            first_line = process.stdout.readline().strip()
            record = Path(first_line.removeprefix("run directory: "))
            assert record.is_dir(), first_line
            assert process.poll() is None
            assert not (record / "exit-code.txt").exists()
        finally:
            (tmp_path / "release").touch()
            output, errors = process.communicate(timeout=10)
    assert process.returncode == 0, errors
    assert (record / "exit-code.txt").read_text().strip() == "0"
    assert "check completed" in output
    assert "exit code: 0" in output


def test_missing_interpreter_is_a_failed_run(completion_command, tmp_path):
    env = run_environment(tmp_path)
    env["cueq_python"] = str(tmp_path / "missing-python")
    result = subprocess.run(
        ["bash", "-c", "set -e\n" + completion_command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    record = Path(result.stdout.splitlines()[0].removeprefix("run directory: "))
    assert result.returncode != 0
    assert (record / "exit-code.txt").read_text().strip() == "127"
    assert "missing-python" in (record / "output.txt").read_text()
