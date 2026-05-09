# Codex Prompt 02 — Implement StageOnly

Goal: Implement the V1 StageOnly condition: compact persistent prompt plus `advance_stage` tool returning stage packets.

Context to read:

- `AGENTS.md`
- `docs/stagegate/02-architecture.md`
- `docs/stagegate/05-tracing-and-visualization.md`
- active ExecPlan 01

Tasks:

1. Add StageGate module skeleton.
2. Add typed `StagePacket` model.
3. Add trace writer.
4. Add `advance_stage` schema to the OpenAI realtime session only when `TAU2_STAGEGATE_CONDITION=stage_only` or `stagegate`.
5. Handle `advance_stage` function calls and return JSON stage packets.
6. Emit trace events.
7. Add unit tests for packet schema and trace writer.
8. Run the smallest relevant tests.

Constraints:

- Do not modify tasks, evaluator, user simulator, domain tools, domain policies, or scoring.
- Do not read hidden task data.
- Do not implement entity ledger or validator yet except stubs needed for clean architecture.

Done when:

- A retail control one-task smoke run can complete far enough to emit `advance_stage_call` and `stage_packet_returned` events.
