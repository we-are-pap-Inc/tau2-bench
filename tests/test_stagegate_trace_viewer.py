import json
import re
from pathlib import Path

from scripts.stagegate_posthoc_outcomes import (
    SCHEMA_VERSION as ORACLE_SCHEMA_VERSION,
)
from scripts.stagegate_posthoc_outcomes import (
    build_outcome_rows,
    write_jsonl,
)
from scripts.stagegate_trace_viewer import SCHEMA_VERSION, load_trace_data, load_traces


def test_load_traces_accepts_valid_schema(tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "ts": "2026-05-08T19:30:00.000Z",
                "run_id": "run_1",
                "condition": "stage_only",
                "domain": "mock",
                "benchmark_task_id": "task_1",
                "sim_id": "sim_1",
                "event_type": "domain_tool_call",
                "stage": "inspect_state_with_read_tools",
                "tick_index": 12,
                "tool_name": "get_account",
                "leakage_risk": "none",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    df = load_traces(str(tmp_path / "**" / "trace_events.jsonl"))

    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "run_1"
    assert df.iloc[0]["tool_name"] == "get_account"
    assert df.iloc[0]["line_number"] == 1


def test_load_trace_data_reports_invalid_rows(tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "ts": "2026-05-08T19:30:00.000Z",
                        "condition": "baseline",
                        "event_type": "run_start",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "stagegate.trace_event.v1",
                        "condition": "stage_only",
                        "event_type": "advance_stage_call",
                    }
                ),
                json.dumps(["not", "an", "object"]),
                "{not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_trace_data(str(trace_path))

    assert len(result.events) == 1
    assert len(result.invalid_rows) == 3
    errors = list(result.invalid_rows["error"])
    assert "unsupported_schema_version" in errors
    assert "invalid_json_object" in errors
    assert any(error.startswith("invalid_json:") for error in errors)


def test_posthoc_outcome_writer_uses_oracle_analysis_schema(tmp_path):
    results_path = tmp_path / "results.json"
    output_path = tmp_path / "oracle_analysis.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "id": "sim_1",
                        "task_id": "task_1",
                        "trial": 2,
                        "domain": "mock",
                        "termination_reason": "agent_stop",
                        "reward_info": {
                            "reward": 1.0,
                            "reward_breakdown": {"db": 1.0},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = build_outcome_rows([results_path])
    write_jsonl(rows, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["schema_version"] == ORACLE_SCHEMA_VERSION
    assert row["event_type"] == "final_outcome"
    assert row["source"] == "posthoc_oracle_analysis"
    assert row["benchmark_task_id"] == "task_1"
    assert row["sim_id"] == "sim_1"
    assert row["reward"] == 1.0
    assert row["passed"] is True
    assert row["reward_breakdown"] == {"db": 1.0}


def test_trace_queries_use_benchmark_task_id():
    repo_root = Path(__file__).resolve().parents[1]
    query_text = (repo_root / "scripts/trace_queries.sql").read_text(encoding="utf-8")

    assert "benchmark_task_id" in query_text
    assert re.search(r"\btask_id\b", query_text) is None
