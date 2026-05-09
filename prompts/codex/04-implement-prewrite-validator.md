# Codex Prompt 04 — Implement Pre-Write Validator

Goal: Implement the StageGate pre-write validator that blocks unsafe or premature side-effecting domain tool calls.

Context to read:

- `docs/stagegate/02-architecture.md`
- `docs/stagegate/06-validity-and-leakage-rules.md`
- active ExecPlan 01

Tasks:

1. Identify read-only vs side-effecting domain tools from public tool schemas and names.
2. Create a reviewed, transparent side-effecting tool list per domain if heuristics are insufficient.
3. Implement `ValidatorDecision` and validator checks.
4. Block write/action tool calls when identity, exact identifiers, policy-state inspection, or confirmation are missing.
5. Return corrective packets to the model.
6. Emit `validator_check`, `validator_allow`, and `validator_block` trace events.
7. Add tests proving blocked calls do not mutate domain state.
8. Add leakage tests.

Constraints:

- Validator must not mutate domain state.
- Validator must not infer expected outcome from hidden data.
- Validator must not call domain tools itself unless original model call passed validation.

Done when:

- Tests prove allow/block behavior.
- A retail control smoke run produces validator events.
