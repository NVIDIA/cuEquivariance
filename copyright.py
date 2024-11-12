"""Scan all the files in the directory and check if they contain a Copyright notice."""

import os
import subprocess
import sys
from typing import Generator


def scan_files(directory: str) -> Generator[str, None, None]:
    for entry in os.scandir(directory):
        if is_ignored_by_git(entry.path):
            continue

        if entry.is_dir():
            yield from scan_files(entry.path)

        if entry.is_file():
            yield entry.path


def is_ignored_by_git(file: str) -> bool:
    if ".git/" in file:
        return True

    result = subprocess.run(["git", "check-ignore", file], capture_output=True)
    return result.returncode == 0


def check_copyright(file: str) -> bool:
    with open(file, "rb") as f:
        try:
            first_line = f.readline().decode("utf-8").strip()
            second_line = f.readline().decode("utf-8").strip()
        except UnicodeDecodeError:
            return False
        except Exception:
            return False

        if not first_line.endswith(
            "SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES"
        ):
            return False
        if not second_line.endswith("SPDX-License-Identifier: Apache-2.0"):
            return False
    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python copyright.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    for file in scan_files(directory):
        print(file, end=" ")

        if not check_copyright(file):
            print("\U0001f641")
        else:
            print("\U00002705")


if __name__ == "__main__":
    main()
