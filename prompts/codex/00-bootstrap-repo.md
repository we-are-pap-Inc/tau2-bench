# Codex Prompt 00 — Bootstrap the StageGate Repo

Goal: Prepare this fork of `sierra-research/tau2-bench` for the StageGate research project.

Context to read first:

- `AGENTS.md`
- `.agent/PLANS.md`
- `docs/stagegate/00-project-brief.md`
- `docs/stagegate/08-codex-operating-model.md`

Constraints:

- Do not modify benchmark tasks, evaluator, user simulator, domain policies, domain tools, or scoring.
- Do not implement anything yet unless you find a tiny broken reference in the project pack.
- Focus on understanding and planning.

Tasks:

1. Inspect the repository layout.
2. Verify that the StageGate handoff docs are present and coherent.
3. Verify that `uv sync --extra voice --extra dev` is the right install command for this repo.
4. Create or update the first active ExecPlan if needed.
5. Report the exact next implementation milestone.

Done when:

- You have a concise repo map.
- You have identified the OpenAI audio-native files to inspect next.
- You have not made speculative code changes.
