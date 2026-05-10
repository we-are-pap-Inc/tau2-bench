#!/usr/bin/env python3
"""Guard StageGate changes from touching prohibited benchmark files."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import PurePosixPath

PROHIBITED_PREFIXES = (
    "data/tau2/domains/",
    "src/tau2/domains/",
    "src/tau2/evaluator/",
    "src/tau2/metrics/",
    "src/tau2/user/",
)
logger = logging.getLogger(__name__)

PROHIBITED_EXACT_PATHS = {
    "src/tau2/user_simulation_voice_presets.py",
    "src/tau2/scripts/recombine_rewards.py",
}

PROHIBITED_NAMES = {
    "policy.md",
    "tasks.json",
    "tasks_voice.json",
    "split_tasks.json",
}


def is_prohibited_stagegate_path(path: str) -> bool:
    """Return whether a path changes benchmark-validity-controlled files."""
    normalized = path.strip().lstrip("./")
    if normalized in PROHIBITED_EXACT_PATHS:
        return True
    if normalized.startswith(PROHIBITED_PREFIXES):
        return True
    return PurePosixPath(normalized).name in PROHIBITED_NAMES


def prohibited_paths(paths: list[str]) -> list[str]:
    """Return prohibited paths from a changed-path list."""
    return [path for path in paths if is_prohibited_stagegate_path(path)]


def changed_paths(base: str, head: str) -> list[str]:
    """Return changed paths for a git revision range."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()

    blocked = prohibited_paths(changed_paths(args.base, args.head))
    if blocked:
        logger.error("StageGate branch changes prohibited benchmark files:")
        for path in blocked:
            logger.error(f"  {path}")
        return 1

    logger.info("StageGate prohibited-path guard passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
