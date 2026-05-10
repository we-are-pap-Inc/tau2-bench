# Modal Execution Guide

## Purpose

Use `modal_tau3_voice_stagegate.py` to launch reproducible τ-Voice StageGate
jobs on Modal. Final reported results must come only from final mode: the fixed
9-job matrix, pinned to one 40-character commit SHA, with no task filters.

Smoke mode is only for infrastructure checks. Never report smoke-mode results as
final results.

## Runner Dependency Policy

Modal is a runner-only dependency. It is not part of core `tau2-bench`, and
`make test`, `make test-voice`, and `make check-all` must not require Modal.

Use `uv --with modal` for runner checks:

    uv run --with modal python -m py_compile modal_tau3_voice_stagegate.py
    uv run --with modal modal --version

Plan-only dry-runs do not import Modal and do not require Modal auth or the
`tau3-voice-secrets` Secret:

    uv run python scripts/stagegate_modal_runner_config.py plan \
      --batch-id tau3voice_gptrt2_stagegate_YYYY_MM_DD \
      --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git \
      --repo-ref FINAL_40_CHAR_COMMIT_SHA \
      --mode final \
      --print-matrix

Base repository validation still starts with:

    uv sync --extra voice --extra dev
    uv run tau2 check-data

## Required Modal Resources

Create one Modal Secret for provider credentials and voice persona IDs. These
are not Modal auth keys:

    modal secret create tau3-voice-secrets \
      OPENAI_API_KEY=... \
      ELEVENLABS_API_KEY=... \
      DEEPGRAM_API_KEY=... \
      TAU2_VOICE_ID_MILDRED_KAPLAN=... \
      TAU2_VOICE_ID_ARJUN_ROY=... \
      TAU2_VOICE_ID_WEI_LIN=... \
      TAU2_VOICE_ID_MAMADOU_DIALLO=... \
      TAU2_VOICE_ID_PRIYA_PATIL=...

Final and smoke runs in this runner use `speech_complexity: regular`, so
external τ-Voice runs need the five regular ElevenLabs persona voice IDs above.
The control-only persona IDs `TAU2_VOICE_ID_MATT_DELANEY` and
`TAU2_VOICE_ID_LISA_BRENNER` are only needed for separate control-speech local
experiments, not for this final-matrix runner.

Create or reuse one Modal Volume:

    tau3-voice-runs

Modal auth and project setup are operator-side prerequisites. Do not pass Modal
auth tokens into the container and do not print secret values.

Run preflight before any live Modal launch:

    uv run --with modal python scripts/stagegate_modal_runner_config.py preflight

Preflight checks the Modal CLI, Modal auth/config through `modal secret list`,
and whether the `tau3-voice-secrets` Secret exists. It does not print secret
values and reports the required key names only. The live Modal functions require
`OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `DEEPGRAM_API_KEY`, and the five regular
`TAU2_VOICE_ID_*` keys via
`modal.Secret.from_name(..., required_keys=...)`.

To validate exact Secret key presence without launching benchmark jobs, run the
remote Secret-key preflight:

    uv run --with modal modal run modal_tau3_voice_stagegate.py \
      --batch-id secret-preflight \
      --verify-secret-keys-only

This uses Modal auth and the Secret, but it does not run `tau2`, read `.env`, or
print any secret values.

## Plan-Only Dry-Run

Final plan-only dry-run:

    uv run python scripts/stagegate_modal_runner_config.py plan \
      --batch-id tau3voice_gptrt2_stagegate_YYYY_MM_DD \
      --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git \
      --repo-ref FINAL_40_CHAR_COMMIT_SHA \
      --mode final \
      --print-matrix

This writes `batch_manifest_planned.json`, prints the exact 9 planned commands,
and runs `scripts/stagegate_final_run_hygiene.py` against the planned final
manifest.

Smoke plan-only dry-run:

    uv run python scripts/stagegate_modal_runner_config.py plan \
      --batch-id dryrun \
      --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git \
      --repo-ref stagegate \
      --mode smoke \
      --condition baseline \
      --domain retail \
      --print-matrix \
      --skip-hygiene

Smoke manifests are marked `mode: smoke` and intentionally fail final hygiene.

## Final Mode

Final mode:

- requires `--repo-ref` to be a full 40-character commit SHA;
- launches exactly 9 jobs: every `baseline`, `stage_only`, and `stagegate`
  condition across `retail`, `airline`, and `telecom`;
- rejects `--condition` and `--domain` subset flags;
- uses no `--num-tasks` or `--task-ids`;
- uses fixed constants:

      model: gpt-realtime-2
      provider: openai
      reasoning_effort: high
      speech_complexity: regular
      tick_duration: 0.2
      max_steps_seconds: 1200
      max_concurrency: 1
      seed: 300

Launch command:

    uv run --with modal modal run modal_tau3_voice_stagegate.py \
      --batch-id tau3voice_gptrt2_stagegate_YYYY_MM_DD \
      --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git \
      --repo-ref FINAL_40_CHAR_COMMIT_SHA \
      --mode final

Do not run the final matrix until the operator explicitly approves the live
9-job run.

## Modal Preflight

Run:

    uv run --with modal python scripts/stagegate_modal_runner_config.py preflight

If this reports that `tau3-voice-secrets` is missing, create it before running
live smoke or final jobs. The preflight command uses `modal secret list --json`,
which lists Secret names and metadata but not secret values.

After the Secret exists, run the remote Secret-key preflight to make Modal
enforce the exact required keys:

    uv run --with modal modal run modal_tau3_voice_stagegate.py \
      --batch-id secret-preflight \
      --verify-secret-keys-only

## Live Smoke Mode

Smoke mode may use a branch or other non-SHA repo ref only because it is
explicitly marked `--mode smoke`. Smoke runs are recorded as smoke in metadata
and are invalid for final reporting.

Modal smoke dry-run command. This imports the Modal app and therefore still
requires Modal auth and the configured Secret:

    uv run --with modal modal run modal_tau3_voice_stagegate.py \
      --batch-id dryrun \
      --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git \
      --repo-ref FINAL_40_CHAR_COMMIT_SHA_OR_BRANCH \
      --mode smoke \
      --condition baseline \
      --domain retail \
      --dry-run

Live one-job smoke command:

    uv run --with modal modal run modal_tau3_voice_stagegate.py \
      --batch-id smoke_YYYY_MM_DD \
      --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git \
      --repo-ref FINAL_40_CHAR_COMMIT_SHA_OR_BRANCH \
      --mode smoke \
      --condition baseline \
      --domain retail

## Live Final Mode

Only run this after plan-only dry-run, Modal preflight, and explicit operator
approval:

    uv run --with modal modal run modal_tau3_voice_stagegate.py \
      --batch-id tau3voice_gptrt2_stagegate_YYYY_MM_DD \
      --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git \
      --repo-ref FINAL_40_CHAR_COMMIT_SHA \
      --mode final

## Manifest Flow

The plan-only dry-run or live local launcher writes `batch_manifest_planned.json`
before spawning jobs. Modal jobs do not update shared batch files.

Each Modal job writes only its own directory:

    /runs/<batch_id>/<condition>/<domain>/job_manifest.json
    /runs/<batch_id>/<condition>/<domain>/command_metadata.json
    /runs/<batch_id>/<condition>/<domain>/trace_events.jsonl
    /runs/<batch_id>/<condition>/<domain>/simulation_output/

After jobs finish, build the completed manifest from per-job manifests:

    uv run --with modal modal run modal_tau3_voice_stagegate.py \
      --batch-id tau3voice_gptrt2_stagegate_YYYY_MM_DD \
      --collect-completed

The collector writes:

    /runs/<batch_id>/batch_manifest_completed.json

Validate final-run hygiene before reporting any result:

    uv run scripts/stagegate_final_run_hygiene.py \
      --manifest /path/to/batch_manifest_completed.json

The hygiene guard rejects task filters, smoke entries, non-SHA final refs,
non-regular speech, inconsistent constants, and missing matrix entries.

## Runtime Environment

The Modal function sets only runner-control environment variables:

    TAU2_STAGEGATE_CONDITION=baseline|stage_only|stagegate
    TAU2_TRACE_JSONL=/runs/<batch_id>/<condition>/<domain>/trace_events.jsonl
    TAU2_TRACE_RUN_ID=<batch_id>:<condition>:<domain>

Provider credentials come from the `tau3-voice-secrets` Modal Secret. The runner
logs sanitized commands only and never logs environment variables, `.env`
contents, or secret values.

## Reliability Notes

- Modal function timeout is `24 * 60 * 60`.
- Jobs use CPU and memory only; remote APIs perform model inference.
- `tau2 run` uses `--auto-resume`, so completed non-infrastructure runs are
  skipped if the same save directory is resumed.
- Keep one final commit SHA fixed across all 9 jobs.
- Do not mix final results from different commits or smoke runs.

## Modal References

- Job processing: https://modal.com/docs/guide/job-queue
- Batch processing: https://modal.com/docs/guide/batch-processing
- Timeouts: https://modal.com/docs/guide/timeouts
- Developing Modal code with LLMs: https://modal.com/docs/guide/developing-with-llms
- Volumes: https://modal.com/docs/guide/volumes
- Secrets: https://modal.com/docs/guide/secrets
