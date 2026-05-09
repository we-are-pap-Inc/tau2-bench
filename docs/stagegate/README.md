# τ³ / τ-Voice StageGate Codex Handoff Pack

This folder is a repo-local knowledge base for delegating the StageGate research project to Codex.

The project is intentionally framed as an empirical scaffold study, not a leaderboard shortcut:

> Same realtime model, same τ-Voice tasks, same speech complexity, same timeout, same evaluator. Change only the voice-agent scaffold and measure the paired task-level delta.

## What to copy into the repository

Copy the contents of this pack into the root of your fork of `sierra-research/tau2-bench`.

Recommended repository layout after copying:

    tau2-bench/
      AGENTS.md
      .codex/
        config.example.toml
        agents/
          explorer.toml
          implementer.toml
          reviewer.toml
          experiment-runner.toml
      .agent/
        PLANS.md
      .agents/
        skills/
          tau3-stagegate/
            SKILL.md
      docs/
        stagegate/
          00-project-brief.md
          01-research-thesis.md
          02-architecture.md
          03-experiment-protocol.md
          04-modal-execution.md
          05-tracing-and-visualization.md
          06-validity-and-leakage-rules.md
          07-sierra-submission.md
          08-codex-operating-model.md
          09-reference-bibliography.md
          exec-plans/
            active/
              01-stagegate-implementation.md
              02-modal-runner-and-experiment-jobs.md
              03-analysis-viewer-and-report.md
      prompts/
        codex/
          00-bootstrap-repo.md
          01-codebase-map.md
          02-implement-stage-only.md
          03-implement-entity-ledger.md
          04-implement-prewrite-validator.md
          05-build-modal-runner.md
          06-build-trace-viewer.md
          07-run-analysis-and-report.md
      modal_tau3_voice_stagegate.py
      scripts/
        stagegate_summarize_results.py
        stagegate_trace_viewer.py
        launch_modal_matrix.sh
        trace_queries.sql

## Recommended Codex workflow

1. Open the fork in Codex App or Codex CLI.
2. Ask Codex to read `AGENTS.md`, `.agent/PLANS.md`, and `docs/stagegate/00-project-brief.md`.
3. Start in Plan mode and ask Codex to produce or refine the active ExecPlan.
4. Use one coherent Codex thread per milestone. Fork only when a branch of work is genuinely independent.
5. Use subagents only for bounded work: codebase exploration, test creation, review, or trace-analysis scripts.
6. Make Codex update the active ExecPlan after every meaningful implementation step.
7. Require tests and evidence before merging any generated change.

## Core deliverable

The final project should produce:

- A standard GPT-Realtime-2 τ-Voice baseline, if the baseline run is unmodified.
- A custom StageOnly scaffold run.
- A custom StageGate scaffold run: staged workflow packets + typed entity ledger + pre-write validator.
- JSONL traces, result CSVs, failure taxonomy, and a trace viewer.
- Sierra-ready custom submission metadata.

## One-sentence project description

StageGate is a hybrid realtime voice-agent scaffold where persistent identity/global rules stay in session instructions, procedural workflow arrives through staged tool outputs, a typed entity ledger tracks exact facts, and a pre-write validator blocks premature or malformed side-effecting tool calls.
