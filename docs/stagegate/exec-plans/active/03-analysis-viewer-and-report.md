# ExecPlan 03 — Analysis Viewer and Report

## Purpose

Build the analysis scripts and trace viewer needed to turn τ-Voice results into a credible empirical contribution. After this plan is complete, a human should be able to compare baseline, StageOnly, and StageGate runs at the domain level and task level, inspect trace timelines, and produce the tables for a public writeup.

## Current State

Template files exist:

- `scripts/stagegate_summarize_results.py`
- `scripts/stagegate_trace_viewer.py`
- `scripts/trace_queries.sql`

Codex must inspect actual τ-bench simulation output format before implementing robust parsing.

## Target State

Scripts produce:

- `results/summary_by_condition_domain.csv`
- `results/per_task_outcomes.csv`
- `results/paired_delta_by_domain.csv`
- `results/failure_taxonomy.csv`
- `results/stagegate_events_summary.csv`

Viewer supports filtering by condition, domain, task, pass/fail, stage, and validator reason.

## Implementation Milestones

### Milestone 1 — Parse simulation results

Inspect `data/simulations/` output and implement robust parsing of task outcomes and metrics.

Acceptance: script summarizes a smoke run.

### Milestone 2 — Parse trace events

Load JSONL traces, validate schema version, and aggregate event counts.

Acceptance: script summarizes stage calls, ledger updates, validator blocks/allows.

### Milestone 3 — Paired delta analysis

Join baseline, StageOnly, and StageGate task outcomes by domain/task ID.

Acceptance: produce paired delta CSV.

### Milestone 4 — Streamlit trace viewer

Implement the viewer with timeline, ledger, stage, validator, and final outcome panels.

Acceptance: viewer starts locally and loads at least one trace file.

### Milestone 5 — Report skeleton

Create `docs/stagegate/results-report-template.md` with tables and narrative slots.

Acceptance: report template is ready to fill after final runs.

## Progress

- [ ] Milestone 1 pending.
- [x] Milestone 2 JSONL trace loading/schema validation implemented for `stagegate.trace.v1`.
- [ ] Milestone 3 pending.
- [x] Milestone 4 minimal trace viewer implemented for JSONL loading, filters, timeline, stage packets, ledger placeholder, validator placeholder, final outcome, and optional paired outcome CSV.
- [ ] Milestone 5 pending.

## Surprises & Discoveries

- 2026-05-08: `scripts/stagegate_trace_viewer.py` imported Streamlit at module import time, which made loader tests depend on the viewer runtime dependency. The loader now keeps Streamlit import inside `main()` so JSONL parsing and schema validation are unit-testable without launching the app.
- 2026-05-08: The freshly created `.venv` initially resolved `uv run pytest` to a global Python 3.13 pytest because the dev extra was not synced. `uv sync --extra voice --extra dev --extra experiments` fixed command resolution to `.venv/bin/pytest`.

## Decision Log

- 2026-05-08: Canonical trace rows use `schema_version="stagegate.trace.v1"` and retain nullable top-level fields for stable DataFrame columns across event types.
- 2026-05-08: Viewer schema validation rejects unsupported trace schema rows into an invalid-row table instead of mixing them with current traces.
- 2026-05-08: The minimal viewer remains Pandas-based for this implementation; DuckDB query examples remain in `scripts/trace_queries.sql` for later analysis work.

## Validation Evidence

- `uv sync --extra voice --extra dev --extra experiments`
  result: completed successfully and installed the dev/voice/experiments environment.
- `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py -q`
  result: `12 passed, 2 warnings in 0.04s`.
- `make test`
  result: `164 passed, 17 failed, 1 xfailed, 14 warnings`; failures are LLM-backed core tests failing with `litellm.AuthenticationError` because `OPENAI_API_KEY` is not set in this environment.
- `make test-voice`
  result: `255 passed, 3 skipped, 83 deselected, 2 warnings in 0.72s`.
- `make check-all`
  result: Ruff check passed and Ruff format reformatted 3 files.
- `git diff --check`
  result: passed with no whitespace errors.
- `uv run ruff check .`
  result: `All checks passed!`.

## Outcomes & Retrospective

Minimal JSONL trace loading and filtering are implemented. Full simulation result parsing, paired delta CSV generation, failure taxonomy, and report filling remain future milestones.
