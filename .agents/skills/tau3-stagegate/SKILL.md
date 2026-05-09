---
name: tau3-stagegate
description: Use this skill when working on the StageGate τ³ / τ-Voice research project, including scaffold implementation, experiment execution, trace analysis, or Sierra submission prep.
---

# StageGate Skill

## Purpose

Help Codex reliably work on the StageGate τ-Voice project.

## Always read first

- `AGENTS.md`
- `docs/stagegate/00-project-brief.md`
- the active ExecPlan for the task

## Validity rules

Never modify benchmark tasks, evaluator, user simulator, domain policy files, domain tools, or scoring.

Never use hidden task objective, expected final database state, evaluator output, or user simulator private plan.

## Implementation pattern

1. Map code path.
2. Make additive, localized changes.
3. Add tests.
4. Emit JSONL traces for new behavior.
5. Run relevant checks.
6. Update the active ExecPlan.

## Experiment constants

Final runs use:

- model: `gpt-realtime-2`
- reasoning effort: `high`
- speech complexity: `regular`
- tick duration: `0.2`
- max steps seconds: `1200`
- max concurrency: `1`
- seed: `300`
- no task filters

## Deliverables

- working StageOnly and StageGate conditions
- Modal runner
- JSONL traces
- analysis CSVs
- Streamlit trace viewer
- Sierra custom submission metadata
