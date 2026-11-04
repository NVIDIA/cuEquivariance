# SPDX-License-Identifier: Apache-2.0
"""Check skill resources and CLI through the same build path as publish.yml."""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("cuequivariance", "cuequivariance_jax", "cuequivariance_torch")


def check_cli(package, import_root, expected, cwd):
    env = {**os.environ, "PYTHONPATH": str(import_root)}
    # Verify import provenance too: an editable install must not hide a missing
    # resource or module in the wheel being checked.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, pathlib, sys; "
            "m = importlib.import_module(sys.argv[1]); "
            "assert pathlib.Path(m.__file__).is_relative_to(sys.argv[2]), m.__file__",
            package,
            str(import_root),
        ],
        cwd=cwd,
        env=env,
        check=True,
    )
    actual = subprocess.check_output(
        [sys.executable, "-m", package, "skill"], cwd=cwd, env=env
    )
    assert actual == expected + b"\n", (
        f"{package}: CLI output differs from canonical skill"
    )


def build(project, kind, destination):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hatchling",
            "build",
            "-t",
            kind,
            "-d",
            str(destination),
        ],
        cwd=project,
        check=True,
    )


def check_package(package, temp):
    canonical = ROOT / "skills" / package.replace("_", "-") / "SKILL.md"
    expected = canonical.read_bytes()
    assert not canonical.is_symlink(), f"{canonical}: nv-carps needs a real file"
    assert (ROOT / package / package / "SKILL.md").read_bytes() == expected
    check_cli(package, ROOT / package, expected, temp)

    # Publishing already dereferences LICENSE, README, VERSION and SKILL links.
    # Copy with dereferencing to exercise that behavior without modifying the
    # developer's checkout. Direct sdists from the linked tree are not releases.
    project = temp / "source" / package
    shutil.copytree(
        ROOT / package,
        project,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", "dist", "*.egg-info"
        ),
    )
    wheel_dir, sdist_dir = temp / "wheel", temp / "sdist"
    build(project, "wheel", wheel_dir)
    build(project, "sdist", sdist_dir)
    archive = next(sdist_dir.glob("*.tar.gz"))
    with tarfile.open(archive) as source:
        members = [
            m for m in source.getmembers() if m.name.endswith(f"/{package}/SKILL.md")
        ]
        assert len(members) == 1 and members[0].isfile(), (
            "sdist must contain a regular SKILL.md"
        )
        assert source.extractfile(members[0]).read() == expected
        source.extractall(temp / "unpacked", filter="data")
    rebuilt = temp / "rebuilt"
    build(next((temp / "unpacked").iterdir()), "wheel", rebuilt)
    for label, directory in (("wheel", wheel_dir), ("rebuilt", rebuilt)):
        wheel = next(directory.glob("*.whl"))
        import_root = temp / f"import-{label}"
        with zipfile.ZipFile(wheel) as contents:
            assert contents.read(f"{package}/SKILL.md") == expected
            contents.extractall(import_root)
        check_cli(package, import_root, expected, temp)
    print(f"{package}: source CLI, wheel, sdist, and rebuilt wheel passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=PACKAGES)
    args = parser.parse_args()
    for package in (args.package,) if args.package else PACKAGES:
        with tempfile.TemporaryDirectory(prefix=f"{package}-skill-packaging-") as temp:
            check_package(package, Path(temp))


if __name__ == "__main__":
    main()
