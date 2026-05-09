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
- [ ] Milestone 2 pending.
- [ ] Milestone 3 pending.
- [ ] Milestone 4 pending.
- [ ] Milestone 5 pending.

## Surprises & Discoveries

Record discoveries here.

## Decision Log

Record decisions here.

## Validation Evidence

Record commands and outputs here.

## Outcomes & Retrospective

Complete after implementation.
