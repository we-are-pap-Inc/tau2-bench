#!/usr/bin/env python3
"""Streamlit viewer for StageGate JSONL traces.

Run:
    streamlit run scripts/stagegate_trace_viewer.py -- --trace-glob '/path/to/**/trace_events.jsonl'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def load_traces(pattern: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in Path("/").glob(pattern.lstrip("/")) if pattern.startswith("/") else Path(".").glob(pattern):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["trace_file"] = str(path)
                    rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def multiselect_if_present(df: pd.DataFrame, col: str, label: str) -> list[str]:
    if col not in df.columns:
        return []
    values = sorted(str(v) for v in df[col].dropna().unique())
    return st.sidebar.multiselect(label, values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-glob", default="**/trace_events.jsonl")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="StageGate Trace Viewer", layout="wide")
    st.title("StageGate Trace Viewer")

    df = load_traces(args.trace_glob)
    if df.empty:
        st.warning("No trace events found. Provide --trace-glob or run a StageGate smoke test.")
        return

    filtered = df.copy()
    for col, label in [
        ("run_id", "Run"),
        ("condition", "Condition"),
        ("domain", "Domain"),
        ("task_id", "Task"),
        ("event_type", "Event type"),
        ("stage", "Stage"),
        ("validator_reason", "Validator reason"),
    ]:
        selected = multiselect_if_present(filtered, col, label)
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]

    st.subheader("Events")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Event counts")
    if "event_type" in filtered.columns:
        st.dataframe(filtered.groupby("event_type").size().reset_index(name="count"), use_container_width=True)

    st.subheader("Validator blocks")
    if "validator_decision" in filtered.columns:
        blocks = filtered[filtered["validator_decision"].astype(str) == "block"]
        st.dataframe(blocks, use_container_width=True)

    st.subheader("Ledger updates")
    if "event_type" in filtered.columns:
        ledger = filtered[filtered["event_type"].astype(str) == "ledger_update"]
        st.dataframe(ledger, use_container_width=True)


if __name__ == "__main__":
    main()
