#!/usr/bin/env python3
"""Pure configuration helpers for the StageGate Modal runner."""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Literal, Protocol

Condition = Literal["baseline", "stage_only", "stagegate"]
Domain = Literal["retail", "airline", "telecom"]
RunMode = Literal["final", "smoke", "dev"]

CONDITIONS: tuple[Condition, ...] = ("baseline", "stage_only", "stagegate")
DOMAINS: tuple[Domain, ...] = ("retail", "airline", "telecom")
DEFAULT_REPO_URL = "https://github.com/we-are-pap-Inc/tau2-bench.git"
DEFAULT_SECRET_NAME = "tau3-voice-secrets"
REQUIRED_API_SECRET_KEYS = ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "DEEPGRAM_API_KEY")
REQUIRED_REGULAR_VOICE_ID_KEYS = (
    "TAU2_VOICE_ID_MILDRED_KAPLAN",
    "TAU2_VOICE_ID_ARJUN_ROY",
    "TAU2_VOICE_ID_WEI_LIN",
    "TAU2_VOICE_ID_MAMADOU_DIALLO",
    "TAU2_VOICE_ID_PRIYA_PATIL",
)
OPTIONAL_CONTROL_VOICE_ID_KEYS = (
    "TAU2_VOICE_ID_MATT_DELANEY",
    "TAU2_VOICE_ID_LISA_BRENNER",
)
REQUIRED_PROVIDER_SECRET_KEYS = (
    *REQUIRED_API_SECRET_KEYS,
    *OPTIONAL_CONTROL_VOICE_ID_KEYS,
    *REQUIRED_REGULAR_VOICE_ID_KEYS,
)
SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
logger = logging.getLogger(__name__)

RUN_CONSTANT_VALUE = str | int | float | bool

FINAL_CONSTANTS: dict[str, str | int | float] = {
    "model": "gpt-realtime-2",
    "provider": "openai",
    "reasoning_effort": "high",
    "speech_complexity": "regular",
    "tick_duration": "0.2",
    "max_steps_seconds": "1200",
    "max_concurrency": "1",
    "seed": "300",
}

SMOKE_CONSTANTS: dict[str, RUN_CONSTANT_VALUE] = {
    "model": "gpt-realtime-2",
    "provider": "openai",
    "reasoning_effort": "high",
    "speech_complexity": "control",
    "tick_duration": "0.2",
    "max_steps_seconds": "300",
    "max_concurrency": "1",
    "seed": "300",
    "num_tasks": "1",
    "audio_taps": True,
    "auto_resume": False,
}

DEV_CONSTANTS: dict[str, RUN_CONSTANT_VALUE] = {
    "model": "gpt-realtime-2",
    "provider": "openai",
    "reasoning_effort": "high",
    "tick_duration": "0.2",
    "max_steps_seconds": "600",
    "max_concurrency": "1",
    "seed": "300",
    "num_tasks": "10",
    "audio_taps": False,
    "auto_resume": True,
}

DEV_SPEECH_COMPLEXITY_BY_DOMAIN: dict[Domain, str] = {
    "retail": "regular",
    "airline": "control",
    "telecom": "control",
}

FORBIDDEN_TASK_FILTER_FLAGS = ("--num-tasks", "--task-ids")
DEFAULT_DEV_MODAL_JOB_CONCURRENCY = 4


@dataclass(frozen=True)
class StageGateJob:
    """One StageGate condition/domain run."""

    condition: Condition
    domain: Domain
    mode: RunMode


class BlockingModalCall(Protocol):
    """Minimal Modal FunctionCall surface needed by the local launcher."""

    def get(self, timeout: float | None = None) -> Any:
        """Block until the remote call finishes and return its result."""


@dataclass(frozen=True)
class SpawnedStageGateCall:
    """A spawned Modal call plus the StageGate job it represents."""

    job: StageGateJob
    function_call_id: str | None
    call: BlockingModalCall


def utc_now_iso() -> str:
    """Return a UTC ISO timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_full_commit_sha(repo_ref: str) -> bool:
    """Return whether repo_ref is a full 40-character git SHA."""
    return bool(SHA_PATTERN.fullmatch(repo_ref.strip()))


def require_full_commit_sha(repo_ref: str, *, mode: RunMode) -> None:
    """Require a full commit SHA unless this is an explicit smoke run."""
    if mode in {"final", "dev"} and not is_full_commit_sha(repo_ref):
        raise ValueError(f"{mode} mode requires repo_ref to be a full 40-character SHA")


def validate_condition(condition: str) -> Condition:
    """Validate and return a StageGate condition token."""
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition {condition!r}")
    return condition  # type: ignore[return-value]


def validate_domain(domain: str) -> Domain:
    """Validate and return a StageGate domain token."""
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain {domain!r}")
    return domain  # type: ignore[return-value]


def final_matrix() -> list[StageGateJob]:
    """Return the exact final 9-job matrix."""
    return [
        StageGateJob(condition=condition, domain=domain, mode="final")
        for condition in CONDITIONS
        for domain in DOMAINS
    ]


def smoke_jobs(
    condition: str | None,
    domain: str | None,
    *,
    allow_dev_smoke_domain: bool = False,
) -> list[StageGateJob]:
    """Return the paid smoke matrix, with an explicit non-retail dev escape hatch."""
    if condition is not None:
        raise ValueError("smoke mode always runs all StageGate conditions")
    smoke_domain = validate_domain(domain or "retail")
    if smoke_domain != "retail" and not allow_dev_smoke_domain:
        raise ValueError(
            "smoke mode defaults to retail only; pass --allow-dev-smoke-domain "
            "for non-retail development smoke runs"
        )
    return [
        StageGateJob(condition=condition_name, domain=smoke_domain, mode="smoke")
        for condition_name in CONDITIONS
    ]


def _parse_selector_csv(
    raw_value: str,
    *,
    allowed_values: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    """Parse comma-separated selector values in canonical allowed-value order."""
    requested = [value.strip() for value in raw_value.split(",")]
    if any(not value for value in requested):
        raise ValueError(f"{label} selectors must not contain empty values")
    invalid = sorted({value for value in requested if value not in allowed_values})
    if invalid:
        raise ValueError(f"unsupported {label} selector(s): {', '.join(invalid)}")
    if len(set(requested)) != len(requested):
        raise ValueError(f"{label} selectors must not contain duplicates")
    requested_set = set(requested)
    return tuple(value for value in allowed_values if value in requested_set)


def dev_jobs(
    condition: str | None,
    domain: str | None,
    *,
    conditions: str | None = None,
    domains: str | None = None,
) -> list[StageGateJob]:
    """Return the mixed 10-task development replication matrix."""
    singular_selected = condition is not None or domain is not None
    plural_selected = conditions is not None or domains is not None
    if singular_selected and plural_selected:
        raise ValueError("dev mode cannot mix singular and plural selectors")
    if singular_selected:
        if condition is None or domain is None:
            raise ValueError("dev mode subsets require both condition and domain")
        return [
            StageGateJob(
                condition=validate_condition(condition),
                domain=validate_domain(domain),
                mode="dev",
            )
        ]

    selected_conditions = (
        _parse_selector_csv(
            conditions,
            allowed_values=CONDITIONS,
            label="conditions",
        )
        if conditions is not None
        else CONDITIONS
    )
    selected_domains = (
        _parse_selector_csv(
            domains,
            allowed_values=DOMAINS,
            label="domains",
        )
        if domains is not None
        else DOMAINS
    )
    return [
        StageGateJob(condition=condition_name, domain=domain_name, mode="dev")
        for condition_name in selected_conditions
        for domain_name in selected_domains
    ]


def resolve_modal_job_concurrency(
    *,
    mode: RunMode,
    jobs: list[StageGateJob],
    override: int = 0,
) -> int:
    """Return Modal job-level concurrency for a launcher invocation."""
    if override < 0:
        raise ValueError("modal_job_concurrency must be non-negative")
    if override > 0:
        return override
    if mode == "dev":
        return max(1, min(DEFAULT_DEV_MODAL_JOB_CONCURRENCY, len(jobs)))
    return max(1, len(jobs))


def _has_any_selector(
    condition: str | None,
    domain: str | None,
    conditions: str | None,
    domains: str | None,
) -> bool:
    return any(
        selector is not None for selector in (condition, domain, conditions, domains)
    )


def planned_jobs(
    *,
    mode: RunMode,
    condition: str | None = None,
    domain: str | None = None,
    conditions: str | None = None,
    domains: str | None = None,
    allow_dev_smoke_domain: bool = False,
) -> list[StageGateJob]:
    """Return jobs for the requested mode."""
    if mode == "final":
        if _has_any_selector(condition, domain, conditions, domains):
            raise ValueError("final mode does not accept condition/domain subsets")
        return final_matrix()
    if mode == "dev":
        return dev_jobs(
            condition,
            domain,
            conditions=conditions,
            domains=domains,
        )
    if conditions is not None or domains is not None:
        raise ValueError(
            f"{mode} mode does not accept plural condition/domain selectors"
        )
    return smoke_jobs(
        condition,
        domain,
        allow_dev_smoke_domain=allow_dev_smoke_domain,
    )


def save_name(batch_id: str, job: StageGateJob) -> str:
    """Return the tau2 save name for a job."""
    return f"{job.mode}_{batch_id}_{job.condition}_{job.domain}"


def run_constants(job: StageGateJob) -> dict[str, RUN_CONSTANT_VALUE]:
    """Return mode-specific constants for one StageGate job."""
    if job.mode == "smoke":
        return dict(SMOKE_CONSTANTS)
    if job.mode == "dev":
        constants = dict(DEV_CONSTANTS)
        constants["speech_complexity"] = DEV_SPEECH_COMPLEXITY_BY_DOMAIN[job.domain]
        return constants
    return dict(FINAL_CONSTANTS)


def trace_run_id(batch_id: str, job: StageGateJob) -> str:
    """Return the stable trace run identifier for one job."""
    return f"{batch_id}:{job.condition}:{job.domain}"


def trace_jsonl_path(batch_id: str, job: StageGateJob) -> str:
    """Return the StageGate trace JSONL path inside the Modal Volume."""
    return f"/runs/{batch_id}/{job.condition}/{job.domain}/trace_events.jsonl"


def artifact_dir(batch_id: str, job: StageGateJob) -> str:
    """Return the per-job artifact directory inside the Modal Volume."""
    return f"/runs/{batch_id}/{job.condition}/{job.domain}"


def simulation_output_dir(batch_id: str, job: StageGateJob) -> str:
    """Return the copied simulation output path inside the artifact directory."""
    return f"{artifact_dir(batch_id, job)}/simulation_output"


def tau2_save_to(batch_id: str, job: StageGateJob) -> str:
    """Return the tau2 --save-to target for one Modal job."""
    if job.mode == "smoke":
        return save_name(batch_id, job)
    return simulation_output_dir(batch_id, job)


def build_tau2_command(batch_id: str, job: StageGateJob) -> list[str]:
    """Build the fixed, sanitized tau2 command for a job."""
    constants = run_constants(job)
    argv = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        job.domain,
        "--audio-native",
        "--audio-native-provider",
        str(constants["provider"]),
        "--audio-native-model",
        str(constants["model"]),
        "--reasoning-effort",
        str(constants["reasoning_effort"]),
        "--speech-complexity",
        str(constants["speech_complexity"]),
        "--tick-duration",
        str(constants["tick_duration"]),
        "--max-steps-seconds",
        str(constants["max_steps_seconds"]),
        "--max-concurrency",
        str(constants["max_concurrency"]),
        "--seed",
        str(constants["seed"]),
    ]
    if job.mode in {"smoke", "dev"}:
        argv.extend(
            [
                "--num-tasks",
                str(constants["num_tasks"]),
                "--verbose-logs",
            ]
        )
        if constants.get("audio_taps"):
            argv.append("--audio-taps")
    else:
        argv.append("--verbose-logs")
    if job.mode == "final" or constants.get("auto_resume"):
        argv.append("--auto-resume")
    argv.extend(["--save-to", tau2_save_to(batch_id, job)])
    return argv


def command_to_log(argv: list[str]) -> str:
    """Return shell-quoted command text for logs without secrets."""
    return shlex.join(argv)


def wait_for_stagegate_calls(
    calls: list[SpawnedStageGateCall],
    *,
    log: logging.Logger | None = None,
    poll_timeout_seconds: float = 60.0,
    transient_retry_sleep_seconds: float = 15.0,
    max_transient_wait_errors: int = 20,
) -> list[Any]:
    """Wait for every spawned StageGate Modal call before returning."""
    active_logger = log or logger
    results: list[Any] = []
    failures: list[tuple[SpawnedStageGateCall, BaseException]] = []

    active_logger.info(
        "Waiting for %s StageGate Modal job(s) to complete before exiting",
        len(calls),
    )
    for spawned in calls:
        job = spawned.job
        call_id = spawned.function_call_id or "unknown"
        active_logger.info(
            "Waiting for StageGate Modal job condition=%s domain=%s "
            "function_call_id=%s",
            job.condition,
            job.domain,
            call_id,
        )
        transient_wait_errors = 0
        while True:
            try:
                result = spawned.call.get(timeout=poll_timeout_seconds)
            except TimeoutError:
                active_logger.info(
                    "StageGate Modal job still running condition=%s domain=%s "
                    "function_call_id=%s",
                    job.condition,
                    job.domain,
                    call_id,
                )
                continue
            except Exception as exc:
                if (
                    is_transient_modal_wait_error(exc)
                    and transient_wait_errors < max_transient_wait_errors
                ):
                    transient_wait_errors += 1
                    active_logger.warning(
                        "Transient Modal wait error for condition=%s domain=%s "
                        "function_call_id=%s; retrying wait (%s/%s): %s",
                        job.condition,
                        job.domain,
                        call_id,
                        transient_wait_errors,
                        max_transient_wait_errors,
                        exc,
                    )
                    if transient_retry_sleep_seconds > 0:
                        time.sleep(transient_retry_sleep_seconds)
                    continue
                failures.append((spawned, exc))
                active_logger.exception(
                    "StageGate Modal job failed condition=%s domain=%s "
                    "function_call_id=%s",
                    job.condition,
                    job.domain,
                    call_id,
                )
                break
            else:
                results.append(result)
                active_logger.info(
                    "StageGate Modal job completed condition=%s domain=%s "
                    "function_call_id=%s",
                    job.condition,
                    job.domain,
                    call_id,
                )
                break

    if failures:
        failure_summaries = []
        for spawned, exc in failures:
            job = spawned.job
            call_id = spawned.function_call_id or "unknown"
            detail = str(exc) or type(exc).__name__
            failure_summaries.append(
                f"{job.condition}/{job.domain} function_call_id={call_id}: "
                f"{type(exc).__name__}: {detail}"
            )
        raise RuntimeError(
            f"{len(failures)} StageGate Modal job(s) failed: "
            + "; ".join(failure_summaries)
        ) from failures[0][1]

    return results


def is_transient_modal_wait_error(exc: BaseException) -> bool:
    """Return whether a local Modal wait error is worth retrying."""
    detail = str(exc).lower()
    transient_fragments = (
        "nodename nor servname provided",
        "name or service not known",
        "temporary failure in name resolution",
        "failed to connect to all addresses",
        "connection reset by peer",
        "connection aborted",
        "connection refused",
        "connection timed out",
        "transport is closing",
    )
    if not any(fragment in detail for fragment in transient_fragments):
        return False
    module_name = type(exc).__module__.lower()
    return "modal" in module_name or "grpclib" in module_name or "socket" in module_name


def validate_no_task_filters(argv: list[str]) -> None:
    """Reject task filtering flags."""
    for flag in FORBIDDEN_TASK_FILTER_FLAGS:
        if flag in argv or any(arg.startswith(f"{flag}=") for arg in argv):
            raise ValueError(f"task filter {flag} is not allowed")


def validate_final_command(argv: list[str]) -> None:
    """Validate constants and absence of task filters on a tau2 command."""
    validate_no_task_filters(argv)
    if _has_flag(argv, "--audio-taps"):
        raise ValueError("--audio-taps is not allowed in final mode")
    expected = {
        "--audio-native-provider": str(FINAL_CONSTANTS["provider"]),
        "--audio-native-model": str(FINAL_CONSTANTS["model"]),
        "--reasoning-effort": str(FINAL_CONSTANTS["reasoning_effort"]),
        "--speech-complexity": str(FINAL_CONSTANTS["speech_complexity"]),
        "--tick-duration": str(FINAL_CONSTANTS["tick_duration"]),
        "--max-steps-seconds": str(FINAL_CONSTANTS["max_steps_seconds"]),
        "--max-concurrency": str(FINAL_CONSTANTS["max_concurrency"]),
        "--seed": str(FINAL_CONSTANTS["seed"]),
    }
    for flag, value in expected.items():
        if _flag_value(argv, flag) != value:
            raise ValueError(f"{flag} must be {value!r}")
    if not _has_flag(argv, "--auto-resume"):
        raise ValueError("--auto-resume is required in final mode")


def validate_smoke_command(argv: list[str]) -> None:
    """Validate constants and task filtering for a paid smoke command."""
    if _has_flag(argv, "--task-ids"):
        raise ValueError("--task-ids is not allowed in smoke mode")
    expected = {
        "--audio-native-provider": str(SMOKE_CONSTANTS["provider"]),
        "--audio-native-model": str(SMOKE_CONSTANTS["model"]),
        "--reasoning-effort": str(SMOKE_CONSTANTS["reasoning_effort"]),
        "--speech-complexity": str(SMOKE_CONSTANTS["speech_complexity"]),
        "--tick-duration": str(SMOKE_CONSTANTS["tick_duration"]),
        "--max-steps-seconds": str(SMOKE_CONSTANTS["max_steps_seconds"]),
        "--max-concurrency": str(SMOKE_CONSTANTS["max_concurrency"]),
        "--seed": str(SMOKE_CONSTANTS["seed"]),
        "--num-tasks": str(SMOKE_CONSTANTS["num_tasks"]),
    }
    for flag, value in expected.items():
        if _flag_value(argv, flag) != value:
            raise ValueError(f"{flag} must be {value!r}")
    if not _has_flag(argv, "--audio-taps"):
        raise ValueError("--audio-taps is required in smoke mode")
    if _has_flag(argv, "--auto-resume"):
        raise ValueError("--auto-resume is not allowed in smoke mode")


def validate_dev_command(argv: list[str], *, job: StageGateJob) -> None:
    """Validate constants and task filtering for a paid development command."""
    if _has_flag(argv, "--task-ids"):
        raise ValueError("--task-ids is not allowed in dev mode")
    constants = run_constants(job)
    expected = {
        "--audio-native-provider": str(constants["provider"]),
        "--audio-native-model": str(constants["model"]),
        "--reasoning-effort": str(constants["reasoning_effort"]),
        "--speech-complexity": str(constants["speech_complexity"]),
        "--tick-duration": str(constants["tick_duration"]),
        "--max-steps-seconds": str(constants["max_steps_seconds"]),
        "--max-concurrency": str(constants["max_concurrency"]),
        "--seed": str(constants["seed"]),
        "--num-tasks": str(constants["num_tasks"]),
    }
    for flag, value in expected.items():
        if _flag_value(argv, flag) != value:
            raise ValueError(f"{flag} must be {value!r}")
    if _has_flag(argv, "--audio-taps"):
        raise ValueError("--audio-taps is not allowed in dev mode by default")
    if not _has_flag(argv, "--auto-resume"):
        raise ValueError("--auto-resume is required in dev mode")


def planned_manifest(
    *,
    batch_id: str,
    repo_url: str,
    repo_ref: str,
    mode: RunMode,
    jobs: list[StageGateJob],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the local pre-spawn batch manifest."""
    manifest_run_constants = run_constants(jobs[0]) if jobs else {}
    return {
        "schema_version": "stagegate.modal.batch_manifest.planned.v1",
        "batch_id": batch_id,
        "repo_url": repo_url,
        "requested_repo_ref": repo_ref,
        "mode": mode,
        "created_at": created_at or utc_now_iso(),
        "final_constants": dict(FINAL_CONSTANTS),
        "run_constants": manifest_run_constants,
        "runs": [
            job_manifest_base(
                batch_id=batch_id,
                repo_url=repo_url,
                repo_ref=repo_ref,
                resolved_commit_sha=None,
                job=job,
                start_timestamp=None,
                end_timestamp=None,
                status="planned",
            )
            for job in jobs
        ],
    }


def write_planned_manifest(
    *,
    batch_id: str,
    repo_url: str,
    repo_ref: str,
    mode: RunMode,
    condition: str | None = None,
    domain: str | None = None,
    conditions: str | None = None,
    domains: str | None = None,
    allow_dev_smoke_domain: bool = False,
    output_path: Path = Path("batch_manifest_planned.json"),
) -> dict[str, Any]:
    """Create a plan-only manifest without importing or contacting Modal."""
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
    write_json(output_path, manifest)
    return manifest


def run_final_hygiene_guard(manifest_path: Path) -> CompletedProcess[str]:
    """Run the final hygiene guard against a generated manifest."""
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            "scripts/stagegate_final_run_hygiene.py",
            "--manifest",
            str(manifest_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=repo_root,
    )


def job_manifest_base(
    *,
    batch_id: str,
    repo_url: str,
    repo_ref: str,
    resolved_commit_sha: str | None,
    job: StageGateJob,
    start_timestamp: str | None,
    end_timestamp: str | None,
    status: str,
    modal_function_call_id: str | None = None,
    modal_container_id: str | None = None,
) -> dict[str, Any]:
    """Build the common job manifest shape."""
    argv = build_tau2_command(batch_id, job)
    constants = run_constants(job)
    if job.mode == "final":
        validate_final_command(argv)
    elif job.mode == "smoke":
        validate_smoke_command(argv)
    else:
        validate_dev_command(argv, job=job)
    return {
        "schema_version": "stagegate.modal.job_manifest.v1",
        "batch_id": batch_id,
        "condition": job.condition,
        "domain": job.domain,
        "repo_url": repo_url,
        "requested_repo_ref": repo_ref,
        "resolved_commit_sha": resolved_commit_sha,
        "final_constants": dict(FINAL_CONSTANTS),
        "run_constants": constants,
        "args": argv,
        "command": argv,
        "sanitized_command_argv": argv,
        "save_name": save_name(batch_id, job),
        "save_to": tau2_save_to(batch_id, job),
        "artifact_dir": artifact_dir(batch_id, job),
        "trace_jsonl": trace_jsonl_path(batch_id, job),
        "trace_run_id": trace_run_id(batch_id, job),
        "simulation_output_dir": simulation_output_dir(batch_id, job),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "status": status,
        "modal_function_call_id": modal_function_call_id,
        "modal_container_id": modal_container_id,
        "mode": job.mode,
        "model": constants["model"],
        "provider": constants["provider"],
        "reasoning_effort": constants["reasoning_effort"],
        "speech_complexity": constants["speech_complexity"],
        "tick_duration": constants["tick_duration"],
        "max_steps_seconds": constants["max_steps_seconds"],
        "timeout": constants["max_steps_seconds"],
        "seed": constants["seed"],
        "concurrency": constants["max_concurrency"],
        "max_concurrency": constants["max_concurrency"],
        "num_tasks": constants.get("num_tasks"),
        "task_ids": None,
        "audio_taps": constants.get("audio_taps", False),
        "auto_resume": constants.get("auto_resume", True),
    }


def command_metadata(
    *,
    batch_id: str,
    repo_url: str,
    repo_ref: str,
    resolved_commit_sha: str | None,
    job: StageGateJob,
) -> dict[str, Any]:
    """Build command metadata for a job."""
    argv = build_tau2_command(batch_id, job)
    constants = run_constants(job)
    return {
        "schema_version": "stagegate.modal.command_metadata.v1",
        "batch_id": batch_id,
        "condition": job.condition,
        "domain": job.domain,
        "repo_url": repo_url,
        "requested_repo_ref": repo_ref,
        "resolved_commit_sha": resolved_commit_sha,
        "mode": job.mode,
        "sanitized_command_argv": argv,
        "sanitized_command": command_to_log(argv),
        "save_to": tau2_save_to(batch_id, job),
        "final_constants": dict(FINAL_CONSTANTS),
        "run_constants": constants,
        "trace_jsonl": trace_jsonl_path(batch_id, job),
        "trace_run_id": trace_run_id(batch_id, job),
    }


def collect_completed_manifest(batch_dir: Path) -> dict[str, Any]:
    """Build a completed batch manifest from per-job manifests under batch_dir."""
    job_manifests = sorted(batch_dir.glob("*/*/job_manifest.json"))
    runs: list[dict[str, Any]] = []
    for manifest_path in job_manifests:
        with manifest_path.open("r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return {
        "schema_version": "stagegate.modal.batch_manifest.completed.v1",
        "batch_id": batch_dir.name,
        "created_at": utc_now_iso(),
        "runs": runs,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def jobs_as_dicts(jobs: list[StageGateJob]) -> list[dict[str, str]]:
    """Serialize jobs for assertions and logging."""
    return [asdict(job) for job in jobs]


def check_modal_preflight(
    *,
    secret_name: str = DEFAULT_SECRET_NAME,
    modal_cmd: str = "modal",
    command_runner=subprocess.run,
) -> dict[str, Any]:
    """Check Modal CLI availability, auth, and Secret existence without secrets."""
    if shutil.which(modal_cmd) is None:
        return {
            "ok": False,
            "modal_version": None,
            "secret_name": secret_name,
            "required_keys": list(REQUIRED_PROVIDER_SECRET_KEYS),
            "secret_exists": False,
            "error": f"Modal CLI {modal_cmd!r} was not found on PATH",
        }

    version_result = command_runner(
        [modal_cmd, "--version"],
        check=False,
        text=True,
        capture_output=True,
    )
    if version_result.returncode != 0:
        return {
            "ok": False,
            "modal_version": None,
            "secret_name": secret_name,
            "required_keys": list(REQUIRED_PROVIDER_SECRET_KEYS),
            "secret_exists": False,
            "error": "Modal CLI version check failed",
        }

    secret_result = command_runner(
        [modal_cmd, "secret", "list", "--json"],
        check=False,
        text=True,
        capture_output=True,
    )
    if secret_result.returncode != 0:
        return {
            "ok": False,
            "modal_version": version_result.stdout.strip(),
            "secret_name": secret_name,
            "required_keys": list(REQUIRED_PROVIDER_SECRET_KEYS),
            "secret_exists": False,
            "error": "Modal secret list failed; check Modal auth/config",
        }

    secrets = json.loads(secret_result.stdout or "[]")
    secret_exists = any(secret.get("Name") == secret_name for secret in secrets)
    return {
        "ok": secret_exists,
        "modal_version": version_result.stdout.strip(),
        "secret_name": secret_name,
        "required_keys": list(REQUIRED_PROVIDER_SECRET_KEYS),
        "secret_exists": secret_exists,
        "error": None
        if secret_exists
        else f"Modal Secret {secret_name!r} was not found",
    }


def build_arg_parser() -> ArgumentParser:
    """Build the plan/preflight CLI parser."""
    parser = ArgumentParser(
        description="Plan and preflight StageGate Modal final/smoke runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan",
        help="Generate a plan-only manifest without importing Modal.",
    )
    plan.add_argument("--batch-id", required=True)
    plan.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    plan.add_argument("--repo-ref", required=True)
    plan.add_argument("--mode", choices=("final", "smoke", "dev"), default="final")
    plan.add_argument("--condition")
    plan.add_argument("--domain")
    plan.add_argument(
        "--conditions",
        help="Comma-separated dev-mode condition selectors, e.g. baseline,stage_only.",
    )
    plan.add_argument(
        "--domains",
        help="Comma-separated dev-mode domain selectors, e.g. retail,airline,telecom.",
    )
    plan.add_argument(
        "--allow-dev-smoke-domain",
        action="store_true",
        help="Allow non-retail smoke jobs for development only.",
    )
    plan.add_argument(
        "--output",
        type=Path,
        default=Path("batch_manifest_planned.json"),
    )
    plan.add_argument("--print-matrix", action="store_true")
    plan.add_argument(
        "--skip-hygiene",
        action="store_true",
        help="Do not run final hygiene after writing a final-mode plan.",
    )

    preflight = subparsers.add_parser(
        "preflight",
        help="Check Modal CLI/auth and required Secret existence.",
    )
    preflight.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    preflight.add_argument("--modal-cmd", default="modal")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI for plan-only dry-run and Modal preflight."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.command == "plan":
        return _run_plan_command(args)
    if args.command == "preflight":
        return _run_preflight_command(args)
    raise AssertionError(f"Unhandled command {args.command!r}")


def _run_plan_command(args: Namespace) -> int:
    manifest = write_planned_manifest(
        batch_id=args.batch_id,
        repo_url=args.repo_url,
        repo_ref=args.repo_ref,
        mode=args.mode,
        condition=args.condition,
        domain=args.domain,
        conditions=args.conditions,
        domains=args.domains,
        allow_dev_smoke_domain=args.allow_dev_smoke_domain,
        output_path=args.output,
    )
    logger.info("Wrote planned manifest to %s", args.output)
    if args.print_matrix:
        for run in manifest["runs"]:
            logger.info(
                "Planned %s/%s mode=%s command=%s",
                run["condition"],
                run["domain"],
                run["mode"],
                command_to_log(run["sanitized_command_argv"]),
            )
    if args.mode == "final" and not args.skip_hygiene:
        result = run_final_hygiene_guard(args.output)
        if result.stdout:
            logger.info(result.stdout.strip())
        if result.stderr:
            logger.info(result.stderr.strip())
    return 0


def _run_preflight_command(args: Namespace) -> int:
    result = check_modal_preflight(
        secret_name=args.secret_name,
        modal_cmd=args.modal_cmd,
    )
    logger.info("Modal version: %s", result["modal_version"] or "unavailable")
    logger.info(
        "Required Secret %s exists: %s", args.secret_name, result["secret_exists"]
    )
    logger.info("Required Secret keys: %s", ", ".join(result["required_keys"]))
    if not result["ok"]:
        logger.error(result["error"])
        return 1
    return 0


def _flag_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


if __name__ == "__main__":
    sys.exit(main())
