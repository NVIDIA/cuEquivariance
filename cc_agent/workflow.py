"""Isolated CPU/GPU build, pytest selection, and coverage for SWQA CC CLI.

Run from the repository root. Dependencies and caches stay inside .venv.
The host Python used to launch this wrapper needs only the standard library.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PACKAGES = ("cuequivariance", "cuequivariance_jax", "cuequivariance_torch")
VENV = Path(".venv")
PYTHON = str(VENV / "bin" / "python")
CPU_VENV = VENV / "cpu"
CPU_PYTHON = str(CPU_VENV / "bin" / "python")
SELECTORS_VARIABLE = "SWQA_PYTEST_SELECTORS"
SOURCE_PATTERNS = ",".join(f"{name}/{name}/*" for name in PACKAGES)
REPORT = VENV / "coverage.xml"


def run(argv, env):
    print("+ " + shlex.join(argv), flush=True)
    subprocess.run(argv, env=env, check=True)


def environment():
    env = os.environ.copy()
    cache = (VENV / ".cache").resolve()
    env.update(
        UV_CACHE_DIR=str(cache / "uv"),
        UV_HTTP_TIMEOUT="180",
        UV_PYTHON_INSTALL_DIR=str((VENV / ".python").resolve()),
        XDG_CACHE_HOME=str(cache),
        JAX_COMPILATION_CACHE_DIR=str(cache / "jax"),
        TORCH_HOME=str(cache / "torch"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        COVERAGE_FILE=str((VENV / ".coverage").resolve()),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    return env


def build(env):
    # CI uses uv editable installs; all installs target our venv explicitly.
    run(["uv", "venv", "--allow-existing", "--python", "3.12", str(VENV)], env)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            PYTHON,
            "-e",
            "./cuequivariance[dev]",
            "-e",
            "./cuequivariance_jax",
            "-e",
            "./cuequivariance_torch",
            "jax[cuda12]<0.11.2",
            "flax>=0.12.0",
            "equinox>=0.11.11",
            "torch",
            "e3nn",
            "triton",
            "coverage",
            "nvidia-cusolver-cu12==11.7.3.90",
            "nvidia-cublas-cu12",
            "cuequivariance-ops-jax-cu12",
            "cuequivariance-ops-torch-cu12",
        ],
        env,
    )
    # CPU CI deliberately omits CUDA ops. Merely hiding the GPU while those
    # packages remain importable makes some modules select a CUDA-only method.
    run(["uv", "venv", "--allow-existing", "--python", PYTHON, str(CPU_VENV)], env)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            CPU_PYTHON,
            "-e",
            "./cuequivariance[dev]",
            "-e",
            "./cuequivariance_jax",
            "-e",
            "./cuequivariance_torch",
            "jax<0.11.2",
            "flax>=0.12.0",
            "equinox>=0.11.11",
            "torch",
            "e3nn",
            "triton",
            "coverage",
        ],
        env,
    )


def device_environment(env, device):
    selected = env.copy()
    # GPU tests explicitly exercise CPU reference implementations as well.
    # Keep that backend available while making CUDA the default device.
    selected["JAX_PLATFORMS"] = "cpu" if device == "cpu" else "cuda,cpu"
    if device == "cpu":
        selected["CUDA_VISIBLE_DEVICES"] = ""
    return selected


def require_gpu(package, env):
    # Fail explicitly instead of allowing a CPU-only test run to look successful.
    if package == "cuequivariance_jax":
        code = (
            "import jax, cuequivariance_ops_jax; "
            "devices = jax.devices('gpu'); "
            "assert devices, 'JAX GPU unavailable'; print(devices)"
        )
    else:
        code = (
            "import torch, cuequivariance_ops_torch; "
            "assert torch.cuda.is_available(), 'PyTorch GPU unavailable'; "
            "print(torch.cuda.get_device_name(0))"
        )
    run([PYTHON, "-c", code], env)


def selected_tests():
    selectors = os.environ.get(SELECTORS_VARIABLE, "")
    grouped = {name: [] for name in PACKAGES}
    for selector in dict.fromkeys(selectors.split("\n")):
        if not selector:
            continue
        path = Path(selector.split("::", 1)[0])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Expected repository-relative pytest node ID: {selector}")
        if (
            len(path.parts) < 2
            or path.parts[0] not in grouped
            or path.parts[1] != "tests"
        ):
            raise ValueError(
                f"Selector must be inside a configured test directory: {selector}"
            )
        grouped[path.parts[0]].append(selector)
    return grouped


def tests(action, env):
    focused = action == "focused"
    coverage = action == "coverage"
    selections = selected_tests() if focused else {}
    has_selections = any(selections.values())
    if coverage:
        # Remove the old report before collecting fresh, non-appended baseline data.
        REPORT.unlink(missing_ok=True)
        run([PYTHON, "-m", "coverage", "erase"], env)

    for package in PACKAGES:
        if focused and has_selections and not selections[package]:
            continue
        targets = (
            selections[package]
            if has_selections
            else [f"{package}/tests" if focused else package]
        )
        devices = ("cpu",) if package == "cuequivariance" else ("cpu", "gpu")
        for device in devices:
            print(f"Running {package}: {device} ({action})", flush=True)
            selected_env = device_environment(env, device)
            python = CPU_PYTHON if device == "cpu" else PYTHON
            if device == "gpu":
                require_gpu(package, selected_env)
            if coverage:
                command = [
                    python,
                    "-m",
                    "coverage",
                    "run",
                    "--append",
                    "--branch",
                    "--source=.",
                    "-m",
                    "pytest",
                ]
            else:
                command = [python, "-m", "pytest"]
            command += ["-x"]
            if focused:
                command += ["-m", "not slow"]
            else:
                command += ["--doctest-modules"]
            run(command + targets, selected_env)

    if coverage:
        run(
            [
                PYTHON,
                "-m",
                "coverage",
                "xml",
                "--include=" + SOURCE_PATTERNS,
                "-o",
                str(REPORT),
            ],
            env,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "focused", "test", "coverage"))
    args = parser.parse_args()
    if not all(Path(name, "pyproject.toml").is_file() for name in PACKAGES):
        parser.error("Run this wrapper from the cuEquivariance repository root")
    env = environment()
    if args.action == "build":
        build(env)
    else:
        if not all(Path(p).is_file() for p in (PYTHON, CPU_PYTHON)):
            parser.error(
                "Repository CPU/GPU venv is missing; run the build action first"
            )
        tests(args.action, env)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
