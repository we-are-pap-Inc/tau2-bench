#!/usr/bin/env python3
"""Build posthoc StageGate oracle outcome rows from completed result files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "stagegate.oracle_analysis.v1"
logger = logging.getLogger(__name__)


def load_result_file(path: Path) -> list[dict[str, Any]]:
    """Load simulations from a tau2 results JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("simulations"), list):
        simulations = data["simulations"]
    elif isinstance(data, dict):
        simulations = [data]
    else:
        raise ValueError(f"Unsupported results file shape: {path}")

    return [
        row
        for row in simulations
        if isinstance(row, dict) and row.get("reward_info") is not None
    ]


def outcome_row(simulation: dict[str, Any], *, source_file: Path) -> dict[str, Any]:
    """Convert one completed simulation row into an oracle analysis event."""
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")
    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": "final_outcome",
        "source": "posthoc_oracle_analysis",
        "source_file": str(source_file),
        "run_id": simulation.get("id"),
        "sim_id": simulation.get("id"),
        "benchmark_task_id": simulation.get("task_id"),
        "trial": simulation.get("trial"),
        "domain": simulation.get("domain"),
        "termination_reason": simulation.get("termination_reason"),
        "reward": reward,
        "passed": reward == 1.0 if reward is not None else None,
        "reward_breakdown": reward_info.get("reward_breakdown"),
    }


def build_outcome_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Build posthoc rows from one or more result files."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        for simulation in load_result_file(path):
            rows.append(outcome_row(simulation, source_file=path))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="One or more tau2 result JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oracle_analysis.jsonl"),
        help="Output oracle analysis JSONL path.",
    )
    args = parser.parse_args()

    rows = build_outcome_rows(args.results)
    write_jsonl(rows, args.output)
    logger.info(f"Wrote {len(rows)} posthoc outcome row(s) to {args.output}")


if __name__ == "__main__":
    main()
