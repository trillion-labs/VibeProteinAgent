#!/usr/bin/env python3
"""Check that release branches do not contain generated runtime artifacts."""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

NOISY_PATTERNS = (
    "job_*",
    "workspace*",
    "slurm-*.out",
    "slurm-*.err",
    "*.out",
    "*.err",
    "codex_notes/phase20_matrix*.json",
    "codex_notes/*_json_*.json",
    "vendor/**/checkpoints/**",
    "vendor/**/ckpts/**",
    "vendor/**/params/**",
    "vendor/**/tool_weights/**",
    "vendor/**/release_data/**",
    "vendor/protenix-pxd/protenix/model/layer_norm/*.o",
    "vendor/protenix-pxd/protenix/model/layer_norm/*.so",
    "vendor/protenix-pxd/protenix/model/layer_norm/*.cuda.*",
    "vendor/protenix-pxd/protenix/model/layer_norm/build.ninja",
    "vendor/protenix-pxd/protenix/model/layer_norm/.ninja_*",
    "vendor/protenix-pxd/protenix/model/layer_norm/lock",
)

IGNORED_SCRATCH_PATTERNS = (
    "job_*",
    "workspace*",
    "slurm-*.out",
    "slurm-*.err",
    "*.out",
    "*.err",
    "codex_notes/phase20_matrix*.json",
    "codex_notes/*_json_*.json",
)


def _git_lines(args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _matches(path: str, *, patterns: tuple[str, ...] = NOISY_PATTERNS) -> bool:
    name = Path(path).name
    if name in {".gitignore", "UPSTREAM.md"} or name.startswith("LICENSE"):
        return False
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _untracked_paths(*, include_ignored: bool = False) -> tuple[list[str], list[str]]:
    untracked: list[str] = []
    ignored: list[str] = []
    args = ["status", "--porcelain", "--untracked-files=all"]
    if include_ignored:
        args.append("--ignored=matching")
    for line in _git_lines(args):
        if line.startswith("?? "):
            untracked.append(line[3:])
        elif include_ignored and line.startswith("!! "):
            ignored.append(line[3:])
    return untracked, ignored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="Also fail on untracked generated artifacts in the working tree.",
    )
    args = parser.parse_args()

    tracked = [path for path in _git_lines(["ls-files"]) if _matches(path)]
    noisy = list(tracked)
    if args.include_untracked:
        untracked, ignored = _untracked_paths(include_ignored=True)
        noisy.extend(path for path in untracked if _matches(path))
        noisy.extend(
            path
            for path in ignored
            if _matches(path, patterns=IGNORED_SCRATCH_PATTERNS)
        )

    if noisy:
        print("Generated/runtime artifacts found in release-sensitive paths:")
        for path in sorted(set(noisy)):
            print(f"  - {path}")
        print("\nRemove them or add a narrower ignore rule before release.")
        return 1

    root = Path.cwd()
    print(f"release hygiene ok: no generated artifacts matched under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
