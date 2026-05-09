# Codex Operating Model for This Project

## Strategy

Codex should work as an implementation teammate, not a magic one-shot writer.

Use a repository-local system of record:

- `AGENTS.md` as a concise map and rules file.
- `docs/stagegate/` as the source of truth for project context.
- `.agent/PLANS.md` as the planning method.
- `docs/stagegate/exec-plans/active/` for living milestone plans.
- JSONL traces and CSVs as empirical evidence.

## Recommended Codex surfaces

Use Codex App or CLI for iterative local work.

Use Codex Cloud for PR-sized tasks and reviewable diffs.

Use Modal for actual τ-Voice benchmark runs. Codex writes and tests the Modal runner; Modal executes the benchmark jobs.

## Recommended model

Use the strongest available Codex model in the account for planning and implementation. As of the docs used for this pack, OpenAI recommends starting with `gpt-5.5` in Codex when available, using `gpt-5.4` during rollout if not available, and using smaller/faster models for lighter coding or subagent work.

## Threads

Use one main thread per milestone:

1. codebase map
2. StageOnly
3. entity ledger
4. pre-write validator
5. Modal runner
6. trace viewer
7. analysis/reporting
8. Sierra submission

Do not keep one endless mega-thread for the entire project. Do not split into many live threads touching the same files unless using worktrees.

## Subagents

Use subagents for bounded tasks:

- explorer: read-heavy codebase exploration
- implementer: focused coding task
- reviewer: PR/diff review
- experiment-runner: Modal/runbook work

Do not ask subagents to recursively spawn more agents.

## Prompt pattern

Every Codex task should include:

- Goal
- Context files
- Constraints
- Done when
- Exact validation commands

Example:

    Goal: Implement the typed entity ledger for StageGate.
    Context: Read AGENTS.md, docs/stagegate/02-architecture.md, and the active ExecPlan.
    Constraints: Do not modify task/evaluator/user simulator/domain tool files. Do not read hidden task state.
    Done when: ledger model exists, tests cover slot transitions, trace events include ledger_update, and pytest passes.

## Review loop

Before accepting a Codex change, require:

- diff review
- tests run
- benchmark-validity check
- trace/schema consistency check
- update to ExecPlan Decision Log

## Reusable skill

The repository includes `.agents/skills/tau3-stagegate/SKILL.md`. Once the workflow stabilizes, ask Codex to use this skill for StageGate-related tasks.
