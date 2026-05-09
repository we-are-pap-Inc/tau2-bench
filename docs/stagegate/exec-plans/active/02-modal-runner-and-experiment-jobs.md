# ExecPlan 02 — Modal Runner and Experiment Jobs

## Purpose

Build a Modal runner that executes the final τ-Voice experiment matrix in a reproducible way. After this plan is complete, a human should be able to launch all condition/domain jobs with one command and collect artifacts from a Modal Volume.

## Current State

A template runner exists at `modal_tau3_voice_stagegate.py`. Codex must verify it against the actual repository and Modal SDK behavior.

## Target State

A working Modal app that:

- builds a Python 3.12 image with system dependencies;
- clones the fork at a fixed commit SHA;
- installs with `uv sync --extra voice --extra dev`;
- runs `tau2 check-data`;
- executes one full-domain τ-Voice run per function call;
- writes outputs to a Modal Volume;
- exposes a local entrypoint that launches the 9-job final matrix.

## Non-Negotiable Constraints

Final jobs must use no task filters, regular speech complexity, tick duration 0.2, max steps 1200, max concurrency 1, seed 300, model GPT-Realtime-2, reasoning effort high.

## Implementation Milestones

### Milestone 1 — Verify Modal SDK and image

Confirm package names and image build steps. Adjust the template runner if Modal APIs changed.

Acceptance: `modal run modal_tau3_voice_stagegate.py --help` or equivalent local validation succeeds.

### Milestone 2 — Implement artifact copy and commit

Ensure all simulation outputs and StageGate JSONL traces are copied to `/runs/<batch_id>/<condition>/<domain>/` and persisted with `volume.commit()`.

Acceptance: a smoke run writes expected files to the Modal volume.

### Milestone 3 — Launch matrix

Implement local entrypoint to launch all jobs or a subset of conditions/domains for smoke testing.

Acceptance: can launch one condition/domain and then the full 9-job matrix.

### Milestone 4 — Collection instructions

Document how to download or inspect artifacts from Modal.

Acceptance: `docs/stagegate/04-modal-execution.md` is updated with exact working commands.

## Progress

- [ ] Milestone 1 pending.
- [ ] Milestone 2 pending.
- [ ] Milestone 3 pending.
- [ ] Milestone 4 pending.

## Surprises & Discoveries

Record discoveries here.

## Decision Log

Record decisions here.

## Validation Evidence

Record commands and outputs here.

## Outcomes & Retrospective

Complete after implementation.
