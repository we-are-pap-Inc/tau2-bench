#!/usr/bin/env python3
"""Streamlit viewer for StageGate JSONL traces.

Run:
    streamlit run scripts/stagegate_trace_viewer.py -- --trace-glob '/path/to/**/trace_events.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "stagegate.trace.v1"


@dataclass
class TraceLoadResult:
    events: pd.DataFrame
    invalid_rows: pd.DataFrame


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


def _iter_paths(pattern: str) -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(pattern, recursive=True))]


def load_trace_data(pattern: str) -> TraceLoadResult:
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for path in _iter_paths(pattern):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    invalid_rows.append(
                        {
                            "trace_file": str(path),
                            "line_number": line_number,
                            "error": f"invalid_json: {exc}",
                        }
                    )
                    continue
                if not isinstance(row, dict):
                    invalid_rows.append(
                        {
                            "trace_file": str(path),
                            "line_number": line_number,
                            "error": "invalid_json_object",
                        }
                    )
                    continue
                row["trace_file"] = str(path)
                row["line_number"] = line_number
                if row.get("schema_version") != SCHEMA_VERSION:
                    invalid_rows.append(
                        {
                            "trace_file": str(path),
                            "line_number": line_number,
                            "schema_version": row.get("schema_version"),
                            "event_type": row.get("event_type"),
                            "error": "unsupported_schema_version",
                        }
                    )
                    continue
                rows.append(row)
    return TraceLoadResult(
        events=pd.json_normalize(rows) if rows else _empty_frame(),
        invalid_rows=(pd.DataFrame(invalid_rows) if invalid_rows else _empty_frame()),
    )


def load_traces(pattern: str) -> pd.DataFrame:
    return load_trace_data(pattern).events


def _sorted_values(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    return sorted(str(v) for v in df[col].dropna().unique())


def multiselect_if_present(
    st: Any, df: pd.DataFrame, col: str, label: str
) -> list[str]:
    values = _sorted_values(df, col)
    if not values:
        return []
    return st.sidebar.multiselect(label, values)


def _event_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "event_type" not in df.columns:
        return _empty_frame()
    group_cols = [
        col for col in ["condition", "domain", "event_type"] if col in df.columns
    ]
    return df.groupby(group_cols).size().reset_index(name="count")


def _timeline(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ["ts", "tick_index", "turn_index"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    cols = [
        col
        for col in [
            "ts",
            "run_id",
            "condition",
            "domain",
            "benchmark_task_id",
            "sim_id",
            "event_type",
            "stage",
            "tick_index",
            "turn_index",
            "tool_name",
            "latency_ms",
            "leakage_risk",
            "payload.tool_call_id",
        ]
        if col in df.columns
    ]
    return df[cols] if cols else df


def main() -> None:
    import streamlit as st

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-glob", default="**/trace_events.jsonl")
    parser.add_argument(
        "--paired-outcomes",
        default="results/paired_delta_by_domain.csv",
        help="Optional paired outcome CSV for baseline-vs-StageGate comparison.",
    )
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="StageGate Trace Viewer", layout="wide")
    st.title("StageGate Trace Viewer")

    load_result = load_trace_data(args.trace_glob)
    df = load_result.events
    if df.empty:
        st.warning(
            "No trace events found. Provide --trace-glob or run a StageGate smoke test."
        )
        if not load_result.invalid_rows.empty:
            st.subheader("Invalid rows")
            st.dataframe(load_result.invalid_rows, use_container_width=True)
        return
    if not load_result.invalid_rows.empty:
        st.warning(f"Ignored {len(load_result.invalid_rows)} invalid trace row(s).")
        with st.expander("Invalid rows"):
            st.dataframe(load_result.invalid_rows, use_container_width=True)

    filtered = df.copy()
    for col, label in [
        ("run_id", "Run"),
        ("condition", "Condition"),
        ("domain", "Domain"),
        ("benchmark_task_id", "Task"),
        ("stage", "Stage"),
        ("event_type", "Event type"),
        ("tool_name", "Tool"),
        ("leakage_risk", "Leakage risk"),
        ("passed", "Pass/fail"),
        ("failure_type", "Failure type"),
        ("validator_decision", "Validator decision"),
        ("validator_reason", "Validator reason"),
    ]:
        selected = multiselect_if_present(st, filtered, col, label)
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]

    st.subheader("Chronological timeline")
    st.dataframe(_timeline(filtered), use_container_width=True)

    st.subheader("Event counts")
    counts = _event_counts(filtered)
    if not counts.empty:
        st.dataframe(counts, use_container_width=True)

    st.subheader("Tool calls and results")
    if "event_type" in filtered.columns:
        tool_events = filtered[
            filtered["event_type"]
            .astype(str)
            .isin(["model_function_call", "domain_tool_call", "domain_tool_result"])
        ]
        st.dataframe(tool_events, use_container_width=True)

    st.subheader("Stage packets")
    if "event_type" in filtered.columns:
        packets = filtered[
            filtered["event_type"].astype(str) == "stage_packet_returned"
        ]
        st.dataframe(packets, use_container_width=True)

    st.subheader("Ledger updates")
    if "event_type" in filtered.columns:
        ledger = filtered[filtered["event_type"].astype(str) == "ledger_update"]
        if ledger.empty:
            st.info("No ledger_update events in this trace.")
        else:
            st.dataframe(ledger, use_container_width=True)

    st.subheader("Validator decisions")
    if "validator_decision" in filtered.columns:
        decisions = filtered[filtered["validator_decision"].notna()]
        if decisions.empty:
            st.info("No validator decision events in this trace.")
        else:
            st.dataframe(decisions, use_container_width=True)
    else:
        st.info("No validator decision fields in this trace.")

    st.subheader("Final outcome")
    if "event_type" in filtered.columns:
        outcomes = filtered[filtered["event_type"].astype(str) == "final_outcome"]
        st.dataframe(outcomes, use_container_width=True)

    st.subheader("Baseline vs StageGate")
    paired_path = Path(args.paired_outcomes)
    if paired_path.exists():
        st.dataframe(pd.read_csv(paired_path), use_container_width=True)
    else:
        st.info(f"No paired outcome CSV found at {paired_path}.")


if __name__ == "__main__":
    main()
