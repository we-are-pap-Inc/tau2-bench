# Codex Prompt 05 — Build Modal Runner

Goal: Make `modal_tau3_voice_stagegate.py` work for smoke and final experiment jobs.

Context to read:

- `docs/stagegate/04-modal-execution.md`
- active ExecPlan 02
- `modal_tau3_voice_stagegate.py`

Tasks:

1. Verify Modal SDK syntax.
2. Ensure image installs system dependencies: git, curl, build-essential, pkg-config, ffmpeg, portaudio19-dev, libsndfile, jq.
3. Ensure the function clones the repo at a fixed commit.
4. Ensure `uv sync --extra voice --extra dev` runs.
5. Ensure final command uses correct τ-Voice constants.
6. Ensure artifacts and traces are copied to the Modal Volume.
7. Add local entrypoint options for subset smoke tests and full matrix.
8. Update docs with exact launch commands.

Constraints:

- Do not hardcode secrets.
- Do not use task filters in final matrix.

Done when:

- A one-domain smoke Modal run can launch.
- The full matrix command is ready but not run unless explicitly requested.
