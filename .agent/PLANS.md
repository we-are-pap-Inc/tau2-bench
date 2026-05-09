# Codex Execution Plans for StageGate

Use an ExecPlan for complex features, research milestones, significant refactors, or benchmark execution work. An ExecPlan is a living, self-contained implementation document that allows another Codex session or human engineer to restart the task from the current repository state.

## When to use an ExecPlan

Use an ExecPlan when the work touches any of the following:

- OpenAI audio-native adapter changes
- StageGate orchestrator, ledger, validator, or traces
- Modal execution infrastructure
- final benchmark runs
- Sierra submission artifacts
- analysis/reporting scripts

Do not use an ExecPlan for tiny one-file edits or typo fixes.

## Rules for ExecPlans

Every ExecPlan must:

1. Be self-contained enough for a new Codex session to continue without chat history.
2. Define the user-visible/research-visible outcome first.
3. Name exact files to inspect, edit, create, or avoid.
4. State benchmark validity constraints explicitly.
5. Decompose work into independently verifiable milestones.
6. Include exact commands to run and expected evidence of success.
7. Maintain these sections throughout implementation:
   - Progress
   - Surprises & Discoveries
   - Decision Log
   - Validation Evidence
   - Outcomes & Retrospective
8. Be updated whenever implementation direction changes.
9. Avoid vague tasks such as “make it work.” Each milestone must produce observable behavior.

## ExecPlan skeleton

Copy this skeleton into `docs/stagegate/exec-plans/active/<NN>-<name>.md` and fill it in.

# <Plan Title>

## Purpose

Explain in plain language what will be possible after this plan is implemented and how a human can see that it works.

## Current State

Describe what exists in the repository now. Include exact files, modules, commands, and relevant observations.

## Target State

Describe the intended architecture and behavior. Include the exact behavior boundaries and validity constraints.

## Non-Negotiable Constraints

List benchmark validity, safety, and repository constraints. Include forbidden files or hidden-state access if relevant.

## Implementation Milestones

### Milestone 1 — <name>

Explain the work, files, commands, and observable acceptance criteria.

### Milestone 2 — <name>

Explain the work, files, commands, and observable acceptance criteria.

## Progress

- [ ] Milestone 1 pending.
- [ ] Milestone 2 pending.

## Surprises & Discoveries

Record unexpected repo behavior, test failures, API edge cases, or benchmark constraints discovered during work.

## Decision Log

Record decisions with dates. Include why one design was chosen over alternatives.

## Validation Evidence

Record commands run and concise output evidence.

## Outcomes & Retrospective

Summarize what was completed, what remains, and what future Codex sessions should know.
