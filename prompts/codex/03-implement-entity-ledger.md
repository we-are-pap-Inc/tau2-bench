# Codex Prompt 03 — Implement Entity Ledger

Goal: Implement the StageGate typed entity ledger.

Context to read:

- `docs/stagegate/02-architecture.md`
- `docs/stagegate/05-tracing-and-visualization.md`
- active ExecPlan 01

Tasks:

1. Add `LedgerSlot` and `EntityLedger` models.
2. Implement domain slot definitions for retail, airline, telecom.
3. Update ledger from visible model tool arguments and official domain tool outputs.
4. Add statuses: missing, hypothesized, heard_not_confirmed, repeated_back, user_confirmed, tool_verified, contradicted, stale.
5. Emit `ledger_update` trace events.
6. Add tests for slot transitions and JSON serialization.

Constraints:

- Do not use clean simulator text unavailable to the agent.
- Do not use task ID as a rule selector.
- Do not read expected final database state.

Done when:

- Unit tests cover ledger updates and trace events.
- StageOnly still works.
