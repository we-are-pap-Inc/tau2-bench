#!/usr/bin/env python3
"""Summarize StageGate τ-Voice results.

This is a scaffold. Codex must inspect the actual `data/simulations` output format and implement robust parsing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Codex should implement actual parsing after inspecting tau2 simulation output.
    summary_path = args.output_dir / "summary_by_condition_domain.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "domain",
                "num_tasks",
                "num_pass",
                "num_fail",
                "pass_1",
                "mean_stage_calls",
                "mean_validator_blocks",
            ],
        )
        writer.writeheader()

    print(f"Wrote scaffold summary to {summary_path}")
    print("TODO: implement parsing for the actual tau2-bench output format.")


if __name__ == "__main__":
    main()
