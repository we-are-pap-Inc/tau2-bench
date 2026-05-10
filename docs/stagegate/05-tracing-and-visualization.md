# Tracing and Visualization

## Decision

Use JSONL as the source of truth.

Start with DuckDB + Streamlit for visualization. Phoenix is optional. Raindrop is not needed for the first benchmark run.

## Why JSONL is enough

This is a controlled benchmark, not production monitoring. The key requirement is reproducible, queryable traces that can support paired task-level analysis and manual failure review.

## Trace file

Each task should emit StageGate events to:

    trace_events.jsonl

Each event must include:

    {
      "schema_version": "stagegate.trace.v1",
      "ts": "2026-05-08T19:30:00.000Z",
      "run_id": "tau3voice_2026_05_08_gptrt2",
      "condition": "stagegate",
      "domain": "retail",
      "task_id": "retail_001",
      "sim_id": "sim_abc123",
      "trial": 1,
      "event_type": "validator_block",
      "stage": "propose_action_and_confirm",
      "turn_index": 12,
      "tick_index": 1844,
      "span_id": "span_123",
      "parent_span_id": "span_122",
      "visible_to_agent": true,
      "source": "domain_tool_call",
      "tool_name": "refund_order",
      "tool_args": {"order_id": "O-12345"},
      "ledger_delta": {"confirmation.status": "missing"},
      "validator_decision": "block",
      "validator_reason": "missing_confirmation",
      "latency_ms": 14,
      "leakage_risk": "none"
    }

## Event types

- `run_start`
- `run_end`
- `tick`
- `user_audio_event`
- `user_transcript_event`
- `assistant_audio_event`
- `model_function_call`
- `domain_tool_call`
- `domain_tool_result`
- `advance_stage_call`
- `stage_packet_returned`
- `ledger_update`
- `validator_check`
- `validator_allow`
- `validator_block`

Runtime trace events must not include evaluator-derived outcomes, reward
signals, reward breakdowns, or pass/fail labels.

## Separate posthoc analysis file

Use a separate file for labels that rely on evaluator output:

    oracle_analysis.jsonl

The harness must never read this file during a run.
Generate it only after completed result files exist, for example with
`scripts/stagegate_posthoc_outcomes.py`.

## Streamlit viewer requirements

Build `scripts/stagegate_trace_viewer.py` with filters:

- run_id
- condition
- domain
- task_id
- pass/fail
- failure_type
- stage
- validator decision

Main panels:

1. chronological timeline
2. user/assistant transcript and tool calls
3. stage packets
4. ledger state over time
5. validator decisions
6. final outcome
7. baseline vs StageGate diff

## DuckDB query examples

See `scripts/trace_queries.sql`.

## Why not Raindrop first

Raindrop may be useful for production agent monitoring, alerting, semantic search, and hosted demos. For this benchmark, it creates extra integration work without improving the core evidence. The first version should stay repo-local and reproducible.
