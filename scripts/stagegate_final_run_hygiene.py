#!/usr/bin/env python3
"""Guard final StageGate result manifests against invalid benchmark hygiene."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Any

REQUIRED_CONDITIONS = ("baseline", "stage_only", "stagegate")
REQUIRED_DOMAINS = ("retail", "airline", "telecom")
INVARIANT_FIELDS = {
    "model": (("model", "agent_model", "agent_llm"), ("--model", "--agent-llm")),
    "provider": (("provider", "audio_provider"), ("--provider", "--audio-provider")),
    "reasoning_effort": (
        ("reasoning_effort", "agent_reasoning_effort"),
        ("--reasoning-effort", "--agent-reasoning-effort"),
    ),
    "timeout": (("timeout", "task_timeout"), ("--timeout", "--task-timeout")),
    "seed": (("seed",), ("--seed",)),
    "concurrency": (
        ("concurrency", "max_concurrency"),
        ("--concurrency", "--max-concurrency"),
    ),
}

logger = logging.getLogger(__name__)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a final-run manifest as either a list or an object with runs."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        runs = data
    elif isinstance(data, dict) and isinstance(data.get("runs"), list):
        runs = data["runs"]
    else:
        raise ValueError("Manifest must be a list or an object with a 'runs' list.")
    if not all(isinstance(run, dict) for run in runs):
        raise ValueError("Every manifest run must be a JSON object.")
    return runs


def validate_final_run_manifest(runs: list[dict[str, Any]]) -> list[str]:
    """Return hygiene violations for a StageGate final-run manifest."""
    errors: list[str] = []
    by_domain_condition: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for index, run in enumerate(runs):
        label = run_label(index, run)
        condition = normalize_token(
            metadata_value(run, ("condition",), ("--condition",))
        )
        domain = normalize_token(metadata_value(run, ("domain",), ("--domain",)))
        if condition is None:
            errors.append(f"{label}: missing condition")
        elif condition not in REQUIRED_CONDITIONS:
            errors.append(f"{label}: unsupported condition {condition!r}")
        if domain is None:
            errors.append(f"{label}: missing domain")
        elif domain not in REQUIRED_DOMAINS:
            errors.append(f"{label}: unsupported domain {domain!r}")
        if condition in REQUIRED_CONDITIONS and domain in REQUIRED_DOMAINS:
            by_domain_condition.setdefault((domain, condition), []).append(run)

        if uses_forbidden_task_filter(run):
            errors.append(f"{label}: final runs must not use task filters")

        speech_complexity = normalize_token(
            metadata_value(
                run,
                ("speech_complexity", "complexity"),
                ("--speech-complexity", "--complexity"),
            )
        )
        if speech_complexity != "regular":
            errors.append(
                f"{label}: speech_complexity must be 'regular', "
                f"got {speech_complexity!r}"
            )

        for field, (keys, flags) in INVARIANT_FIELDS.items():
            if metadata_value(run, keys, flags) is None:
                errors.append(f"{label}: missing invariant field {field!r}")

    for domain in REQUIRED_DOMAINS:
        missing = [
            condition
            for condition in REQUIRED_CONDITIONS
            if (domain, condition) not in by_domain_condition
        ]
        if missing:
            errors.append(f"{domain}: missing required conditions {missing}")

        for field, (keys, flags) in INVARIANT_FIELDS.items():
            values_by_condition: dict[str, tuple[str, ...]] = {}
            for condition in REQUIRED_CONDITIONS:
                group = by_domain_condition.get((domain, condition), [])
                if not group:
                    continue
                values = {
                    normalize_for_compare(metadata_value(run, keys, flags))
                    for run in group
                }
                values.discard(None)
                values_by_condition[condition] = tuple(sorted(values))
            if len({values for values in values_by_condition.values()}) > 1:
                errors.append(
                    f"{domain}: inconsistent {field} across conditions "
                    f"{values_by_condition}"
                )

    return errors


def uses_forbidden_task_filter(run: dict[str, Any]) -> bool:
    """Return whether a run selected a non-final subset of benchmark tasks."""
    if run.get("num_tasks") not in (None, "", []):
        return True
    task_ids = run.get("task_ids")
    if task_ids not in (None, "", []):
        return True
    args = run_args(run)
    return has_flag(args, "--num-tasks") or has_flag(args, "--task-ids")


def metadata_value(
    run: dict[str, Any],
    keys: tuple[str, ...],
    flags: tuple[str, ...],
) -> Any:
    """Read a value from structured run metadata or command args."""
    for key in keys:
        if run.get(key) not in (None, ""):
            return run[key]
    args = run_args(run)
    for flag in flags:
        value = flag_value(args, flag)
        if value not in (None, ""):
            return value
    return None


def run_args(run: dict[str, Any]) -> list[str]:
    """Return command arguments from a manifest run."""
    raw = run.get("args", run.get("command", []))
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def has_flag(args: list[str], flag: str) -> bool:
    """Return whether an argv list contains a flag, including --flag=value."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def flag_value(args: list[str], flag: str) -> Any:
    """Return a CLI flag value from --flag value or --flag=value."""
    for index, arg in enumerate(args):
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
    return None


def normalize_token(value: Any) -> str | None:
    """Normalize enum-like manifest values."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def normalize_for_compare(value: Any) -> str | None:
    """Normalize invariant values for cross-condition comparison."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return json.dumps(sorted(str(item) for item in value), separators=(",", ":"))
    return str(value).strip()


def run_label(index: int, run: dict[str, Any]) -> str:
    """Return a stable human-readable label for an invalid run."""
    condition = metadata_value(run, ("condition",), ("--condition",)) or "?"
    domain = metadata_value(run, ("domain",), ("--domain",)) or "?"
    return f"run[{index}] {domain}/{condition}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON final-run manifest with a top-level runs list.",
    )
    args = parser.parse_args()

    errors = validate_final_run_manifest(load_manifest(args.manifest))
    if errors:
        logger.error("StageGate final-run hygiene guard failed:")
        for error in errors:
            logger.error(f"  {error}")
        return 1

    logger.info("StageGate final-run hygiene guard passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
