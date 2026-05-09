# ExecPlan 01 — StageGate Implementation

## Purpose

Implement the StageGate scaffold for τ-Voice so that the same GPT-Realtime-2 model can be evaluated under three conditions: baseline, StageOnly, and StageGate. After this plan is complete, a human should be able to run a one-task smoke test and observe StageGate emitting stage packets, ledger updates, validator checks, and JSONL traces without changing the benchmark evaluator or tasks.

## Current State

The repository is a fork of `sierra-research/tau2-bench`. τ³ voice evaluation lives in the existing voice/audio-native framework. Codex must inspect the actual repo before editing, especially:

- `src/tau2/voice/README.md`
- `src/tau2/voice/audio_native/README.md`
- OpenAI provider files under `src/tau2/voice/audio_native/openai/`
- tests under `tests/`
- `docs/leaderboard-submission.md`

## Target State

Add a StageGate module that can be enabled via:

    TAU2_STAGEGATE_CONDITION=baseline|stage_only|stagegate

Behavior:

- `baseline`: no StageGate behavior.
- `stage_only`: model has `advance_stage` orchestration tool and receives stage packets.
- `stagegate`: StageOnly plus entity ledger plus pre-write validator.

The code must emit JSONL trace events to `TAU2_TRACE_JSONL` when that environment variable is set.

## Non-Negotiable Constraints

Do not modify tasks, evaluator, user simulator, domain policies, domain tools, or scoring.

The harness must not read hidden task objectives, expected final DB state, evaluator output, user simulator private plan, or clean simulator text not visible to the agent path.

## Implementation Milestones

### Milestone 1 — Codebase map

Inspect the audio-native OpenAI provider architecture and identify the precise insertion points for:

- adding `advance_stage` to the session tool list;
- intercepting `advance_stage` function calls;
- observing domain tool calls and results;
- optionally blocking side-effecting domain tool calls before execution;
- emitting trace events.

Acceptance: update this plan with exact file paths and a short architecture map.

### Milestone 2 — StageGate data models and trace writer

Create StageGate module files. Define typed models for:

- `StagePacket`
- `LedgerSlot`
- `EntityLedger`
- `ValidatorDecision`
- `TraceEvent`

Implement a JSONL trace writer that is no-op unless `TAU2_TRACE_JSONL` is set.

Acceptance: unit tests validate schema serialization and JSONL writing.

### Milestone 3 — StageOnly orchestration

Add the `advance_stage` tool and implement a simple deterministic orchestrator that returns compact stage packets from public domain/tool context and visible state.

Acceptance: retail control smoke test shows at least one `advance_stage_call` and one `stage_packet_returned` event.

### Milestone 4 — Entity ledger

Implement typed ledger updates from visible model tool calls and official tool outputs. Include statuses: missing, hypothesized, heard_not_confirmed, repeated_back, user_confirmed, tool_verified, contradicted, stale.

Acceptance: tests cover slot creation, confirmation, contradiction, tool verification, and trace emission.

### Milestone 5 — Pre-write validator

Identify side-effecting tools by schema/name/policy heuristics, with a conservative allowlist or denylist reviewed by the human. Intercept proposed write tools, check identity, exact identifiers, policy-state inspection, and confirmation, then allow or block.

Acceptance: tests show blocked calls do not mutate domain state and return corrective packets.

### Milestone 6 — Integration smoke test

Run one control task in retail for `stage_only` and `stagegate`.

Acceptance: `tau2 view` works, trace file exists, and the run completes or fails for benchmark reasons rather than integration errors.

## Progress

- [ ] Milestone 1 pending.
- [ ] Milestone 2 pending.
- [ ] Milestone 3 pending.
- [ ] Milestone 4 pending.
- [ ] Milestone 5 pending.
- [ ] Milestone 6 pending.

## Surprises & Discoveries

Record discoveries here.

## Decision Log

Record design decisions here.

## Validation Evidence

Record commands and outputs here.

## Outcomes & Retrospective

Complete after implementation.
