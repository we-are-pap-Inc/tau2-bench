# ExecPlan 02 — Modal Runner and Experiment Jobs

## Purpose

Build a Modal runner that executes the final τ-Voice experiment matrix in a reproducible way. After this plan is complete, a human should be able to launch all condition/domain jobs with one command and collect artifacts from a Modal Volume.

## Current State

The template runner has been replaced by a production-oriented runner split
between pure configuration helpers and Modal-specific execution.

## Target State

A working Modal app that:

- builds a Python 3.12 image with system dependencies;
- clones the fork at a fixed commit SHA;
- installs with `uv sync --extra voice --extra dev`;
- runs `tau2 check-data`;
- executes one full-domain τ-Voice run per function call;
- writes outputs to a Modal Volume;
- exposes a local entrypoint that launches the 9-job final matrix;
- provides a plan-only dry-run path that does not import Modal;
- provides a Modal preflight path for CLI/auth/Secret checks;
- separates final mode from smoke mode;
- writes per-job manifests only from Modal jobs;
- builds completed batch manifests in a separate posthoc collector step.

## Non-Negotiable Constraints

Final jobs must use no task filters, regular speech complexity, tick duration
0.2, max steps 1200, max concurrency 1, seed 300, model GPT-Realtime-2,
provider OpenAI, reasoning effort high, a full 40-character commit SHA, and
the exact 9-job condition/domain matrix.

## Implementation Milestones

### Milestone 1 — Verify Modal SDK and image

Confirm package names and image build steps. Keep Modal as a runner-only
dependency checked through `uv run --with modal`.

Acceptance: `modal run modal_tau3_voice_stagegate.py --help` or equivalent local validation succeeds.

### Milestone 2 — Implement artifact copy and commit

Ensure all simulation outputs and StageGate JSONL traces are copied to
`/runs/<batch_id>/<condition>/<domain>/` and persisted with `volume.commit()`.
Each job writes only its own `job_manifest.json` and `command_metadata.json`.

Acceptance: a smoke run writes expected files to the Modal volume.

### Milestone 3 — Launch matrix

Implement local entrypoint to launch exact final jobs or a subset only in
explicit smoke mode.

Acceptance: can launch one condition/domain and then the full 9-job matrix.

### Milestone 4 — Collection instructions

Document final/smoke launch commands, manifest collection, artifact paths, and
hygiene validation.

Acceptance: `docs/stagegate/04-modal-execution.md` is updated with exact working commands.

## Progress

- [x] Milestone 1 implemented and validated against Modal client `1.4.2`.
- [x] Milestone 2 implemented in code; live smoke artifact validation pending.
- [x] Milestone 3 implemented in code; live matrix not launched.
- [x] Milestone 4 documented in `docs/stagegate/04-modal-execution.md`.

## Surprises & Discoveries

- Local repo environment did not include Modal; Modal remains runner-only and is
  checked with `uv run --with modal`.
- Concurrent writes to one batch manifest are avoided by design. The local
  launcher writes `batch_manifest_planned.json`, each job writes only under its
  own condition/domain directory, and the collector creates
  `batch_manifest_completed.json`.
- External τ-Voice runs need persona voice ID overrides. Because this runner
  always uses `speech_complexity: regular`, the Modal Secret must include the
  five regular `TAU2_VOICE_ID_*` keys in addition to provider API keys. The two
  control persona IDs are only needed for separate control-speech experiments.

## Decision Log

- Final mode requires a full 40-character commit SHA and rejects condition/domain
  subsets.
- Smoke mode may accept condition/domain subsets and non-SHA refs, but every
  smoke manifest entry is marked `mode: smoke` and fails final hygiene.
- Modal is not added to `pyproject.toml`; runner-specific checks use
  `uv run --with modal`.
- `scripts/stagegate_modal_runner_config.py` owns pure matrix, command,
  metadata, and manifest logic so most tests do not import Modal.
- Plan-only dry-run is implemented as
  `uv run python scripts/stagegate_modal_runner_config.py plan ...`; it does
  not use `modal run`, because `modal run` validates configured remote objects
  such as Secrets before the local entrypoint can return.
- Modal preflight is implemented as
  `uv run --with modal python scripts/stagegate_modal_runner_config.py preflight`
  and checks `modal --version` plus `modal secret list --json`.

## Modal Docs Review

Reviewed sources:

- Context7 resolved Modal as `/websites/modal` with trust score 9.9.
- Official Modal API reference for `modal.App`, `modal.Function`,
  `modal.Volume`, `modal.Image`, and `modal.Secret`.
- Official Modal CLI reference for `modal run` and `modal secret`.
- Official Modal guides for timeouts, Volumes, and Secrets.
- Installed Modal client `1.4.2` introspection for signatures of `App.function`,
  `App.local_entrypoint`, `Secret.from_name`, `Volume.from_name`,
  `Volume.commit`, `Volume.reload`, `Image.apt_install`,
  `Image.run_commands`, and `Image.env`.

Verdict:

- `modal.App`, `@app.function`, `@app.local_entrypoint`,
  `modal.Secret.from_name`, `modal.Volume.from_name(create_if_missing=True)`,
  `volume.commit()`, `Function.spawn`, `timeout=24 * 60 * 60`,
  `Image.apt_install`, `Image.run_commands`, and `Image.env` are valid for
  Modal client `1.4.2`.
- `modal.Secret.from_name` supports `required_keys`; the runner now requires
  `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `DEEPGRAM_API_KEY`, and the five
  regular persona voice ID keys in `tau3-voice-secrets`.
- Modal Volumes have explicit commit/reload semantics and last-write-wins
  behavior for shared files, so the per-job manifest design avoids concurrent
  shared-file writes.

## Validation Evidence

Validation run on 2026-05-10:

- `uv sync --extra voice --extra dev` completed successfully.
- `uv run tau2 check-data` passed.
- `uv run ruff check .` passed.
- `make check-all` passed; Ruff formatting touched three files.
- `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py -q` passed: 43 tests.
- `uv run pytest tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q` passed: 20 tests.
- `uv run --with modal python -m py_compile modal_tau3_voice_stagegate.py` passed.
- `uv run --with modal modal --version` passed with Modal client `1.4.2`.
- `uv run python scripts/stagegate_modal_runner_config.py plan --batch-id dryrun --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git --repo-ref 1910fe2998f230bda6f6ad1b69edef4124275d63 --mode final --output /private/tmp/stagegate_plan_only_manifest.json --print-matrix` passed and ran final hygiene against the planned manifest without importing Modal.
- `uv run --with modal python scripts/stagegate_modal_runner_config.py preflight` correctly failed because Secret `tau3-voice-secrets` is absent.
- `uv run --with modal modal secret list --json` succeeded and confirmed that
  `tau3-voice-secrets` is absent in the current Modal environment; no secret
  values were printed.

Voice persona environment review on 2026-05-10:

- `src/tau2/data_model/voice_personas.py`, `.env.example`,
  `docs/voice-personas.md`, `src/tau2/voice/scripts/setup_voices.py`,
  `src/tau2/voice/README.md`, `src/tau2/config.py`, and OpenAI/ElevenLabs/
  Deepgram setup paths were inspected.
- Final/smoke Modal runs use regular speech complexity, which samples
  `mildred_kaplan`, `arjun_roy`, `wei_lin`, `mamadou_diallo`, and
  `priya_patil`; external runs require
  `TAU2_VOICE_ID_MILDRED_KAPLAN`, `TAU2_VOICE_ID_ARJUN_ROY`,
  `TAU2_VOICE_ID_WEI_LIN`, `TAU2_VOICE_ID_MAMADOU_DIALLO`, and
  `TAU2_VOICE_ID_PRIYA_PATIL`.
- The Modal runner now exposes `--verify-secret-keys-only`, which validates
  exact Secret key presence through Modal without launching benchmark jobs or
  logging secret values.

## Outcomes & Retrospective

Runner implementation, Modal API review, and local validation are complete. Live
smoke jobs and the live final matrix remain intentionally unlaunched because the
required provider Secret is not present.
