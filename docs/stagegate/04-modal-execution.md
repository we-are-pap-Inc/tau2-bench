# Modal Execution Guide

## Why Modal

τ-Voice final runs are long-running, network-sensitive, API-driven Python jobs that generate logs, audio, trajectories, and JSONL traces. Modal is a good fit because it supports Python functions, secrets, persistent volumes, job-style `.spawn()` execution, batch processing, and long function timeouts.

## Required Modal resources

Create one Modal secret:

    modal secret create tau3-voice-secrets \
      OPENAI_API_KEY=... \
      ELEVENLABS_API_KEY=... \
      DEEPGRAM_API_KEY=... \
      TAU2_VOICE_ID_ALLOY=... \
      TAU2_VOICE_ID_ASH=...

Create/use one Modal volume:

    tau3-voice-runs

## Runner file

Use `modal_tau3_voice_stagegate.py` at repo root.

## Launch command

    modal run modal_tau3_voice_stagegate.py \
      --batch-id tau3voice_gptrt2_stagegate_YYYY_MM_DD \
      --repo-ref FINAL_COMMIT_SHA

## Final job structure

One Modal function call per condition/domain:

- 9 function calls total for baseline, stage_only, stagegate across retail, airline, telecom.
- Each call performs a full-domain run, not a task-filtered run.
- Each call writes artifacts under `/runs/<batch_id>/<condition>/<domain>/`.

## Environment variables inside the function

    TAU2_STAGEGATE_CONDITION=baseline|stage_only|stagegate
    TAU2_TRACE_JSONL=/runs/<batch_id>/<condition>/<domain>/trace_events.jsonl

## Modal reliability notes

- Use `timeout=24 * 60 * 60` for the function.
- Use CPU and memory, not GPU. The models are remote APIs.
- Use `max_concurrency=1` in `tau2 run` for the first final pass.
- Call `volume.commit()` after copying artifacts to the mounted volume.
- Keep final commit SHA fixed across all jobs.
- Do not mix final runs from different code commits.

## Modal docs to hand to the executor

- Job processing: https://modal.com/docs/guide/job-queue
- Batch processing: https://modal.com/docs/guide/batch-processing
- Timeouts: https://modal.com/docs/guide/timeouts
- Developing Modal code with LLMs: https://modal.com/docs/guide/developing-with-llms
- Volumes: https://modal.com/docs/guide/volumes
- Secrets: https://modal.com/docs/guide/secrets
