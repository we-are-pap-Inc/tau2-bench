# Codex Prompt 01 — Map the OpenAI Audio-Native Code Path

Goal: Build a precise implementation map for StageGate insertion points.

Context to read:

- `AGENTS.md`
- `docs/stagegate/02-architecture.md`
- `docs/stagegate/06-validity-and-leakage-rules.md`
- `docs/stagegate/exec-plans/active/01-stagegate-implementation.md`
- `src/tau2/voice/README.md`
- `src/tau2/voice/audio_native/README.md`
- OpenAI audio-native provider files

Tasks:

1. Trace how `tau2 run --audio-native --audio-native-provider openai` creates a realtime session.
2. Identify where session instructions are built.
3. Identify where tool schemas are attached.
4. Identify where model function calls are received.
5. Identify where domain tool calls are executed.
6. Identify whether a write tool can be blocked before domain mutation.
7. Identify where to emit JSONL traces.
8. Update the active ExecPlan with exact paths and insertion strategy.

Constraints:

- No code changes unless needed to create notes.
- Do not modify benchmark behavior.

Done when:

- The active ExecPlan contains exact file paths and a feasible insertion strategy.
