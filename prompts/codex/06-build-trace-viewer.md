# Codex Prompt 06 — Build Trace Viewer

Goal: Implement the JSONL trace viewer using DuckDB/Pandas + Streamlit.

Context to read:

- `docs/stagegate/05-tracing-and-visualization.md`
- active ExecPlan 03
- `scripts/stagegate_trace_viewer.py`
- `scripts/trace_queries.sql`

Tasks:

1. Implement JSONL loading.
2. Validate `schema_version`.
3. Add filters for run_id, condition, domain, task_id, stage, event_type, validator_reason.
4. Build timeline view.
5. Build ledger panel.
6. Build validator events panel.
7. Build baseline-vs-StageGate comparison panel if paired outcome CSV exists.
8. Add a minimal sample trace fixture or loader test.

Done when:

- The viewer starts with `streamlit run scripts/stagegate_trace_viewer.py`.
- It loads sample or real trace files.
