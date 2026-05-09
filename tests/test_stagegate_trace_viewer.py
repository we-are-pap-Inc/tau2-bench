import json

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
                "task_id": "task_1",
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
                "{not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_trace_data(str(trace_path))

    assert len(result.events) == 1
    assert len(result.invalid_rows) == 2
    errors = list(result.invalid_rows["error"])
    assert "unsupported_schema_version" in errors
    assert any(error.startswith("invalid_json:") for error in errors)
