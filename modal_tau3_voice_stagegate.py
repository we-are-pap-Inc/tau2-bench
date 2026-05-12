"""Modal runner for Sierra-grade τ³ / τ-Voice StageGate experiments."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import modal

from scripts.stagegate_modal_runner_config import (
    DEFAULT_REPO_URL,
    DEFAULT_SECRET_NAME,
    REQUIRED_PROVIDER_SECRET_KEYS,
    SpawnedStageGateCall,
    StageGateJob,
    artifact_dir,
    collect_completed_manifest,
    command_metadata,
    command_to_log,
    job_manifest_base,
    planned_jobs,
    planned_manifest,
    require_full_commit_sha,
    resolve_modal_job_concurrency,
    save_name,
    simulation_output_dir,
    trace_jsonl_path,
    trace_run_id,
    utc_now_iso,
    validate_condition,
    validate_domain,
    wait_for_stagegate_calls,
    write_json,
)

APP_NAME = "tau3-voice-stagegate"
SECRET_NAME = DEFAULT_SECRET_NAME
VOLUME_NAME = "tau3-voice-runs"

logger = logging.getLogger(__name__)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "build-essential",
        "pkg-config",
        "ffmpeg",
        "portaudio19-dev",
        "libsndfile1",
        "libsndfile1-dev",
        "jq",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        }
    )
    .add_local_python_source("scripts")
)


def _run_logged(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    logger.info("Running command: %s", command_to_log(argv))
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _modal_container_id() -> str | None:
    return (
        os.environ.get("MODAL_TASK_ID")
        or os.environ.get("MODAL_CONTAINER_ID")
        or os.environ.get("MODAL_FUNCTION_ID")
    )


def _modal_function_call_id() -> str | None:
    try:
        return modal.current_function_call_id()
    except Exception:
        return None


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name(
            SECRET_NAME,
            required_keys=list(REQUIRED_PROVIDER_SECRET_KEYS),
        )
    ],
    volumes={"/runs": volume},
    timeout=24 * 60 * 60,
    cpu=4,
    memory=16_384,
)
def run_domain(
    condition: str,
    domain: str,
    repo_ref: str,
    batch_id: str,
    repo_url: str = DEFAULT_REPO_URL,
    mode: Literal["final", "smoke", "dev"] = "final",
) -> dict[str, str | None]:
    """Run one condition/domain pair in Modal."""

    if mode not in {"final", "smoke", "dev"}:
        raise ValueError("mode must be 'final', 'smoke', or 'dev'")
    require_full_commit_sha(repo_ref, mode=mode)
    job = StageGateJob(
        condition=validate_condition(condition),
        domain=validate_domain(domain),
        mode=mode,
    )
    start_timestamp = utc_now_iso()
    workdir = Path("/tmp/tau2-bench")
    artifacts = Path(artifact_dir(batch_id, job))
    artifacts.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting StageGate Modal job condition=%s domain=%s batch_id=%s "
        "repo_ref=%s artifact_dir=%s",
        job.condition,
        job.domain,
        batch_id,
        repo_ref,
        artifacts,
    )

    resolved_commit_sha: str | None = None
    status = "running"
    error_message: str | None = None
    manifest_path = artifacts / "job_manifest.json"
    metadata_path = artifacts / "command_metadata.json"

    try:
        if workdir.exists():
            shutil.rmtree(workdir)
        _run_logged(["git", "clone", repo_url, str(workdir)])
        _run_logged(["git", "checkout", repo_ref], cwd=workdir)
        rev_parse = _run_logged(
            ["git", "rev-parse", "HEAD"], cwd=workdir, capture_output=True
        )
        resolved_commit_sha = rev_parse.stdout.strip()

        write_json(
            metadata_path,
            command_metadata(
                batch_id=batch_id,
                repo_url=repo_url,
                repo_ref=repo_ref,
                resolved_commit_sha=resolved_commit_sha,
                job=job,
            ),
        )

        _run_logged(["uv", "sync", "--extra", "voice", "--extra", "dev"], cwd=workdir)
        _run_logged(["uv", "run", "tau2", "check-data"], cwd=workdir)

        env = os.environ.copy()
        env["TAU2_STAGEGATE_CONDITION"] = condition
        env["TAU2_TRACE_JSONL"] = trace_jsonl_path(batch_id, job)
        env["TAU2_TRACE_RUN_ID"] = trace_run_id(batch_id, job)
        logger.info(
            "Trace configured condition=%s domain=%s trace_jsonl=%s trace_run_id=%s",
            job.condition,
            job.domain,
            env["TAU2_TRACE_JSONL"],
            env["TAU2_TRACE_RUN_ID"],
        )

        tau2_argv = command_metadata(
            batch_id=batch_id,
            repo_url=repo_url,
            repo_ref=repo_ref,
            resolved_commit_sha=resolved_commit_sha,
            job=job,
        )["sanitized_command_argv"]
        logger.info(
            "Generated tau2 command condition=%s domain=%s command=%s",
            job.condition,
            job.domain,
            command_to_log(tau2_argv),
        )
        _run_logged(tau2_argv, cwd=workdir, env=env)

        source_simulation_dir = (
            workdir / "data" / "simulations" / save_name(batch_id, job)
        )
        if not source_simulation_dir.exists():
            raise FileNotFoundError(
                f"Expected simulation dir not found: {source_simulation_dir}"
            )
        copied_simulation_dir = Path(simulation_output_dir(batch_id, job))
        if copied_simulation_dir.exists():
            shutil.rmtree(copied_simulation_dir)
        shutil.copytree(source_simulation_dir, copied_simulation_dir)

        status = "succeeded"
        return {
            "condition": condition,
            "domain": domain,
            "mode": mode,
            "status": status,
            "artifact_dir": str(artifacts),
            "job_manifest": str(manifest_path),
            "resolved_commit_sha": resolved_commit_sha,
        }
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        logger.exception("StageGate Modal job failed for %s/%s", condition, domain)
        raise
    finally:
        manifest = job_manifest_base(
            batch_id=batch_id,
            repo_url=repo_url,
            repo_ref=repo_ref,
            resolved_commit_sha=resolved_commit_sha,
            job=job,
            start_timestamp=start_timestamp,
            end_timestamp=utc_now_iso(),
            status=status,
            modal_function_call_id=_modal_function_call_id(),
            modal_container_id=_modal_container_id(),
        )
        if error_message is not None:
            manifest["error"] = error_message
        write_json(manifest_path, manifest)
        volume.commit()


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name(
            SECRET_NAME,
            required_keys=list(REQUIRED_PROVIDER_SECRET_KEYS),
        )
    ],
    timeout=60,
    cpu=1,
    memory=512,
)
def verify_secret_keys() -> dict[str, list[str] | str]:
    """Validate required Secret keys without reading or logging values."""

    return {
        "secret_name": SECRET_NAME,
        "required_keys": list(REQUIRED_PROVIDER_SECRET_KEYS),
    }


@app.function(
    image=image,
    volumes={"/runs": volume},
    timeout=60 * 30,
    cpu=1,
    memory=1024,
)
def collect_batch_manifests(batch_id: str) -> dict[str, str | int]:
    """Build batch_manifest_completed.json from per-job manifests."""

    batch_dir = Path("/runs") / batch_id
    completed = collect_completed_manifest(batch_dir)
    output_path = batch_dir / "batch_manifest_completed.json"
    write_json(output_path, completed)
    volume.commit()
    return {
        "batch_id": batch_id,
        "runs": len(completed["runs"]),
        "path": str(output_path),
    }


@app.local_entrypoint()
def launch(
    batch_id: str,
    repo_ref: str = "",
    repo_url: str = DEFAULT_REPO_URL,
    mode: Literal["final", "smoke", "dev"] = "final",
    condition: str | None = None,
    domain: str | None = None,
    conditions: str | None = None,
    domains: str | None = None,
    dry_run: bool = False,
    allow_dev_smoke_domain: bool = False,
    collect_completed: bool = False,
    verify_secret_keys_only: bool = False,
    modal_job_concurrency: int = 0,
) -> None:
    """Launch final/smoke jobs or collect completed job manifests."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if verify_secret_keys_only:
        result = verify_secret_keys.remote()
        logger.info(
            "Verified required Secret keys for %s: %s",
            result["secret_name"],
            ", ".join(result["required_keys"]),
        )
        return

    if collect_completed:
        result = collect_batch_manifests.remote(batch_id)
        logger.info("Collected completed manifest: %s", result)
        return

    if mode not in {"final", "smoke", "dev"}:
        raise ValueError("mode must be 'final', 'smoke', or 'dev'")
    if not repo_ref:
        raise ValueError("repo_ref is required unless --collect-completed is set")
    require_full_commit_sha(repo_ref, mode=mode)
    jobs = planned_jobs(
        mode=mode,
        condition=condition,
        domain=domain,
        conditions=conditions,
        domains=domains,
        allow_dev_smoke_domain=allow_dev_smoke_domain,
    )
    manifest = planned_manifest(
        batch_id=batch_id,
        repo_url=repo_url,
        repo_ref=repo_ref,
        mode=mode,
        jobs=jobs,
    )
    planned_path = Path("batch_manifest_planned.json")
    write_json(planned_path, manifest)
    logger.info("Wrote planned manifest to %s", planned_path)
    logger.info("Prepared %s %s job(s)", len(jobs), mode)
    job_concurrency = resolve_modal_job_concurrency(
        mode=mode,
        jobs=jobs,
        override=modal_job_concurrency,
    )
    logger.info("Modal job concurrency: %s", job_concurrency)

    if dry_run:
        logger.info(
            "Dry run only; live execution would use blocking/waiting mode "
            "via Modal FunctionCall.get() for every scheduled job"
        )
        for job in jobs:
            logger.info(
                "Dry run job %s/%s: %s",
                job.condition,
                job.domain,
                command_to_log(
                    command_metadata(
                        batch_id=batch_id,
                        repo_url=repo_url,
                        repo_ref=repo_ref,
                        resolved_commit_sha=None,
                        job=job,
                    )["sanitized_command_argv"]
                ),
            )
        return

    completed_jobs = 0
    for batch_index, start in enumerate(range(0, len(jobs), job_concurrency), start=1):
        chunk = jobs[start : start + job_concurrency]
        calls: list[SpawnedStageGateCall] = []
        logger.info(
            "Launching Modal job batch %s with %s job(s)", batch_index, len(chunk)
        )
        for job in chunk:
            call = run_domain.spawn(
                job.condition,
                job.domain,
                repo_ref,
                batch_id,
                repo_url,
                mode,
            )
            function_call_id = getattr(call, "object_id", None)
            calls.append(
                SpawnedStageGateCall(
                    job=job,
                    function_call_id=function_call_id,
                    call=call,
                )
            )
            logger.info(
                "Spawned %s/%s function_call_id=%s",
                job.condition,
                job.domain,
                function_call_id,
            )

        logger.info(
            "Launched %s job(s); blocking until this Modal batch completes",
            len(calls),
        )
        wait_for_stagegate_calls(calls, log=logger)
        completed_jobs += len(calls)
        logger.info(
            "Completed Modal job batch %s; %s/%s job(s) done",
            batch_index,
            completed_jobs,
            len(jobs),
        )
    logger.info("All %s StageGate Modal job(s) completed", len(jobs))
