import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.stagegate_final_run_hygiene import validate_final_run_manifest
from scripts.stagegate_modal_runner_config import (
    CONDITIONS,
    DEFAULT_DEV_MODAL_JOB_CONCURRENCY,
    DEFAULT_REPO_URL,
    DEV_CONSTANTS,
    DEV_SPEECH_COMPLEXITY_BY_DOMAIN,
    DOMAINS,
    FINAL_CONSTANTS,
    OPTIONAL_CONTROL_VOICE_ID_KEYS,
    REQUIRED_API_SECRET_KEYS,
    REQUIRED_PROVIDER_SECRET_KEYS,
    REQUIRED_REGULAR_VOICE_ID_KEYS,
    SMOKE_CONSTANTS,
    SpawnedStageGateCall,
    StageGateJob,
    build_tau2_command,
    check_modal_preflight,
    collect_completed_manifest,
    command_metadata,
    final_matrix,
    is_full_commit_sha,
    job_manifest_base,
    planned_jobs,
    planned_manifest,
    require_full_commit_sha,
    resolve_modal_job_concurrency,
    run_constants,
    save_name,
    trace_run_id,
    validate_dev_command,
    validate_final_command,
    validate_smoke_command,
    wait_for_stagegate_calls,
    write_json,
    write_planned_manifest,
)

COMMIT_SHA = "1910fe2998f230bda6f6ad1b69edef4124275d63"


class FakeBlockingCall:
    def __init__(self, result=None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


def test_final_matrix_is_exact_condition_domain_product():
    jobs = final_matrix()

    assert len(jobs) == 9
    assert [(job.condition, job.domain) for job in jobs] == [
        (condition, domain) for condition in CONDITIONS for domain in DOMAINS
    ]
    assert {job.mode for job in jobs} == {"final"}


def test_final_mode_rejects_subsets_and_non_sha_refs():
    with pytest.raises(ValueError, match="condition/domain"):
        planned_jobs(mode="final", condition="baseline")

    with pytest.raises(ValueError, match="40-character"):
        require_full_commit_sha("stagegate", mode="final")

    with pytest.raises(ValueError, match="40-character"):
        require_full_commit_sha("stagegate", mode="dev")

    require_full_commit_sha("stagegate", mode="smoke")


def test_smoke_mode_defaults_to_all_conditions_on_retail():
    assert planned_jobs(mode="smoke") == [
        StageGateJob(condition="baseline", domain="retail", mode="smoke"),
        StageGateJob(condition="stage_only", domain="retail", mode="smoke"),
        StageGateJob(condition="stagegate", domain="retail", mode="smoke"),
    ]


def test_smoke_mode_rejects_condition_subsets_and_non_retail_without_override():
    with pytest.raises(ValueError, match="all StageGate conditions"):
        planned_jobs(mode="smoke", condition="stagegate")

    with pytest.raises(ValueError, match="retail only"):
        planned_jobs(mode="smoke", domain="telecom")

    assert planned_jobs(
        mode="smoke",
        domain="telecom",
        allow_dev_smoke_domain=True,
    ) == [
        StageGateJob(condition="baseline", domain="telecom", mode="smoke"),
        StageGateJob(condition="stage_only", domain="telecom", mode="smoke"),
        StageGateJob(condition="stagegate", domain="telecom", mode="smoke"),
    ]


def test_dev_mode_defaults_to_mixed_ten_task_replication_matrix():
    jobs = planned_jobs(mode="dev")

    assert len(jobs) == 9
    assert [(job.condition, job.domain) for job in jobs] == [
        (condition, domain) for condition in CONDITIONS for domain in DOMAINS
    ]
    assert {job.mode for job in jobs} == {"dev"}


def test_dev_mode_allows_full_condition_domain_job_subset():
    with pytest.raises(ValueError, match="both condition and domain"):
        planned_jobs(mode="dev", condition="stagegate")

    with pytest.raises(ValueError, match="both condition and domain"):
        planned_jobs(mode="dev", domain="telecom")

    assert planned_jobs(mode="dev", condition="stagegate", domain="telecom") == [
        StageGateJob(condition="stagegate", domain="telecom", mode="dev")
    ]


def test_dev_mode_allows_plural_candidate_matrix_selectors():
    jobs = planned_jobs(
        mode="dev",
        conditions="baseline,stage_only",
        domains="retail,airline,telecom",
    )

    assert len(jobs) == 6
    assert [(job.condition, job.domain) for job in jobs] == [
        ("baseline", "retail"),
        ("baseline", "airline"),
        ("baseline", "telecom"),
        ("stage_only", "retail"),
        ("stage_only", "airline"),
        ("stage_only", "telecom"),
    ]
    assert {job.mode for job in jobs} == {"dev"}


def test_dev_plural_selectors_are_canonical_and_reject_bad_values():
    assert planned_jobs(
        mode="dev",
        conditions="stage_only,baseline",
        domains="telecom,retail",
    ) == [
        StageGateJob(condition="baseline", domain="retail", mode="dev"),
        StageGateJob(condition="baseline", domain="telecom", mode="dev"),
        StageGateJob(condition="stage_only", domain="retail", mode="dev"),
        StageGateJob(condition="stage_only", domain="telecom", mode="dev"),
    ]

    with pytest.raises(ValueError, match="cannot mix singular and plural"):
        planned_jobs(
            mode="dev",
            condition="baseline",
            domains="retail,telecom",
        )

    with pytest.raises(ValueError, match="unsupported conditions"):
        planned_jobs(mode="dev", conditions="baseline,other")

    with pytest.raises(ValueError, match="duplicates"):
        planned_jobs(mode="dev", domains="retail,retail")


def test_non_dev_modes_reject_plural_selectors():
    with pytest.raises(ValueError, match="final mode does not accept"):
        planned_jobs(mode="final", conditions="baseline,stage_only")

    with pytest.raises(ValueError, match="smoke mode does not accept plural"):
        planned_jobs(mode="smoke", domains="retail,airline")


def test_modal_job_concurrency_defaults_and_overrides():
    dev_jobs_matrix = planned_jobs(
        mode="dev",
        conditions="baseline,stage_only",
        domains="retail,airline,telecom",
    )
    smoke_jobs_matrix = planned_jobs(mode="smoke")

    assert (
        resolve_modal_job_concurrency(
            mode="dev",
            jobs=dev_jobs_matrix,
        )
        == DEFAULT_DEV_MODAL_JOB_CONCURRENCY
    )
    assert (
        resolve_modal_job_concurrency(
            mode="dev",
            jobs=[dev_jobs_matrix[0]],
        )
        == 1
    )
    assert resolve_modal_job_concurrency(
        mode="smoke",
        jobs=smoke_jobs_matrix,
    ) == len(smoke_jobs_matrix)
    assert (
        resolve_modal_job_concurrency(
            mode="dev",
            jobs=dev_jobs_matrix,
            override=2,
        )
        == 2
    )
    with pytest.raises(ValueError, match="non-negative"):
        resolve_modal_job_concurrency(
            mode="dev",
            jobs=dev_jobs_matrix,
            override=-1,
        )


def test_build_tau2_command_enforces_final_constants_and_no_task_filters():
    job = StageGateJob(condition="stagegate", domain="airline", mode="final")
    argv = build_tau2_command("batch", job)

    validate_final_command(argv)
    assert "--num-tasks" not in argv
    assert "--task-ids" not in argv
    assert "--audio-taps" not in argv
    assert argv[argv.index("--audio-native-model") + 1] == FINAL_CONSTANTS["model"]
    assert argv[argv.index("--speech-complexity") + 1] == "regular"
    assert argv[argv.index("--max-steps-seconds") + 1] == "1200"
    assert argv[argv.index("--save-to") + 1] == "final_batch_stagegate_airline"


def test_build_tau2_command_enforces_smoke_constants_and_debug_artifacts():
    job = StageGateJob(condition="stage_only", domain="retail", mode="smoke")
    argv = build_tau2_command("batch", job)

    validate_smoke_command(argv)
    assert argv[argv.index("--audio-native-model") + 1] == SMOKE_CONSTANTS["model"]
    assert argv[argv.index("--speech-complexity") + 1] == "control"
    assert argv[argv.index("--max-steps-seconds") + 1] == "300"
    assert argv[argv.index("--num-tasks") + 1] == "1"
    assert "--audio-taps" in argv
    assert "--auto-resume" not in argv
    assert "--task-ids" not in argv
    assert argv[argv.index("--save-to") + 1] == "smoke_batch_stage_only_retail"


def test_build_tau2_command_enforces_dev_replication_constants():
    retail_job = StageGateJob(condition="stagegate", domain="retail", mode="dev")
    retail_argv = build_tau2_command("batch", retail_job)
    airline_job = StageGateJob(condition="baseline", domain="airline", mode="dev")
    airline_argv = build_tau2_command("batch", airline_job)

    validate_dev_command(retail_argv, job=retail_job)
    validate_dev_command(airline_argv, job=airline_job)
    assert (
        retail_argv[retail_argv.index("--audio-native-model") + 1]
        == (DEV_CONSTANTS["model"])
    )
    assert retail_argv[retail_argv.index("--speech-complexity") + 1] == "regular"
    assert airline_argv[airline_argv.index("--speech-complexity") + 1] == "control"
    assert retail_argv[retail_argv.index("--max-steps-seconds") + 1] == "600"
    assert retail_argv[retail_argv.index("--num-tasks") + 1] == "10"
    assert "--audio-taps" not in retail_argv
    assert "--auto-resume" not in retail_argv
    assert "--task-ids" not in retail_argv
    assert retail_argv[retail_argv.index("--save-to") + 1] == (
        "dev_batch_stagegate_retail"
    )


def test_validate_final_command_rejects_task_filters_and_mutated_constants():
    job = StageGateJob(condition="baseline", domain="retail", mode="final")
    argv = build_tau2_command("batch", job)

    with pytest.raises(ValueError, match="task filter"):
        validate_final_command([*argv, "--num-tasks", "1"])

    mutated = list(argv)
    mutated[mutated.index("--speech-complexity") + 1] = "control"
    with pytest.raises(ValueError, match="--speech-complexity"):
        validate_final_command(mutated)

    with pytest.raises(ValueError, match="audio-taps"):
        validate_final_command([*argv, "--audio-taps"])


def test_validate_smoke_command_rejects_final_shape_and_task_ids():
    job = StageGateJob(condition="baseline", domain="retail", mode="smoke")
    argv = build_tau2_command("batch", job)

    with pytest.raises(ValueError, match="--task-ids"):
        validate_smoke_command([*argv, "--task-ids", "retail_1"])

    with pytest.raises(ValueError, match="auto-resume"):
        validate_smoke_command([*argv, "--auto-resume"])

    without_audio_taps = [arg for arg in argv if arg != "--audio-taps"]
    with pytest.raises(ValueError, match="audio-taps"):
        validate_smoke_command(without_audio_taps)


def test_validate_dev_command_rejects_task_ids_audio_taps_and_auto_resume():
    job = StageGateJob(condition="baseline", domain="retail", mode="dev")
    argv = build_tau2_command("batch", job)

    with pytest.raises(ValueError, match="--task-ids"):
        validate_dev_command([*argv, "--task-ids", "retail_1"], job=job)

    with pytest.raises(ValueError, match="audio-taps"):
        validate_dev_command([*argv, "--audio-taps"], job=job)

    with pytest.raises(ValueError, match="auto-resume"):
        validate_dev_command([*argv, "--auto-resume"], job=job)


def test_manifest_shape_passes_final_hygiene_guard():
    jobs = final_matrix()
    manifest = planned_manifest(
        batch_id="batch",
        repo_url=DEFAULT_REPO_URL,
        repo_ref=COMMIT_SHA,
        mode="final",
        jobs=jobs,
        created_at="2026-05-10T00:00:00+00:00",
    )

    assert validate_final_run_manifest(manifest["runs"]) == []
    assert all(run["mode"] == "final" for run in manifest["runs"])
    assert all(
        is_full_commit_sha(run["requested_repo_ref"]) for run in manifest["runs"]
    )


def test_smoke_manifest_is_rejected_by_final_hygiene_guard():
    job = StageGateJob(condition="baseline", domain="retail", mode="smoke")
    manifest = job_manifest_base(
        batch_id="dryrun",
        repo_url=DEFAULT_REPO_URL,
        repo_ref="stagegate",
        resolved_commit_sha=None,
        job=job,
        start_timestamp=None,
        end_timestamp=None,
        status="planned",
    )

    errors = validate_final_run_manifest([manifest])

    assert any("mode='final'" in error for error in errors)
    assert any("40-character repo SHA" in error for error in errors)


def test_dev_manifest_is_rejected_by_final_hygiene_guard():
    job = StageGateJob(condition="baseline", domain="retail", mode="dev")
    manifest = job_manifest_base(
        batch_id="dryrun",
        repo_url=DEFAULT_REPO_URL,
        repo_ref=COMMIT_SHA,
        resolved_commit_sha=None,
        job=job,
        start_timestamp=None,
        end_timestamp=None,
        status="planned",
    )

    errors = validate_final_run_manifest([manifest])

    assert any("mode='final'" in error for error in errors)
    assert any("must not use task filters" in error for error in errors)


def test_smoke_manifest_has_trace_run_id_and_smoke_metadata():
    job = StageGateJob(condition="stagegate", domain="retail", mode="smoke")
    manifest = job_manifest_base(
        batch_id="batch",
        repo_url=DEFAULT_REPO_URL,
        repo_ref="stagegate",
        resolved_commit_sha=None,
        job=job,
        start_timestamp=None,
        end_timestamp=None,
        status="planned",
    )

    assert manifest["run_constants"] == run_constants(job)
    assert manifest["trace_jsonl"] == "/runs/batch/stagegate/retail/trace_events.jsonl"
    assert manifest["trace_run_id"] == trace_run_id("batch", job)
    assert manifest["num_tasks"] == "1"
    assert manifest["audio_taps"] is True
    assert manifest["auto_resume"] is False
    assert manifest["save_name"] == save_name("batch", job)


def test_dev_manifest_has_mixed_speech_and_ten_task_metadata():
    jobs = planned_jobs(mode="dev")
    manifests = [
        job_manifest_base(
            batch_id="batch",
            repo_url=DEFAULT_REPO_URL,
            repo_ref=COMMIT_SHA,
            resolved_commit_sha=COMMIT_SHA,
            job=job,
            start_timestamp=None,
            end_timestamp=None,
            status="planned",
        )
        for job in jobs
    ]

    assert {manifest["mode"] for manifest in manifests} == {"dev"}
    assert {manifest["num_tasks"] for manifest in manifests} == {"10"}
    assert {manifest["max_steps_seconds"] for manifest in manifests} == {"600"}
    assert all(manifest["audio_taps"] is False for manifest in manifests)
    assert all(manifest["auto_resume"] is False for manifest in manifests)
    assert {
        manifest["domain"]: manifest["speech_complexity"] for manifest in manifests
    } == DEV_SPEECH_COMPLEXITY_BY_DOMAIN


def test_collect_completed_manifest_reads_per_job_manifests(tmp_path: Path):
    batch_dir = tmp_path / "batch"
    job = StageGateJob(condition="baseline", domain="retail", mode="final")
    manifest = job_manifest_base(
        batch_id="batch",
        repo_url=DEFAULT_REPO_URL,
        repo_ref=COMMIT_SHA,
        resolved_commit_sha=COMMIT_SHA,
        job=job,
        start_timestamp="2026-05-10T00:00:00+00:00",
        end_timestamp="2026-05-10T01:00:00+00:00",
        status="succeeded",
    )
    write_json(batch_dir / "baseline" / "retail" / "job_manifest.json", manifest)

    completed = collect_completed_manifest(batch_dir)

    assert completed["batch_id"] == "batch"
    assert completed["runs"] == [manifest]


def test_command_metadata_contains_sanitized_argv_only():
    job = StageGateJob(condition="baseline", domain="retail", mode="final")
    metadata = command_metadata(
        batch_id="batch",
        repo_url=DEFAULT_REPO_URL,
        repo_ref=COMMIT_SHA,
        resolved_commit_sha=COMMIT_SHA,
        job=job,
    )

    assert metadata["sanitized_command_argv"] == build_tau2_command("batch", job)
    assert metadata["trace_jsonl"] == "/runs/batch/baseline/retail/trace_events.jsonl"
    assert metadata["trace_run_id"] == "batch:baseline:retail"
    assert metadata["run_constants"] == run_constants(job)
    assert "OPENAI_API_KEY" not in metadata["sanitized_command"]
    assert "ELEVENLABS_API_KEY" not in metadata["sanitized_command"]
    assert "DEEPGRAM_API_KEY" not in metadata["sanitized_command"]


def test_wait_for_stagegate_calls_blocks_on_every_scheduled_job():
    jobs = planned_jobs(mode="smoke")
    fake_calls = [
        FakeBlockingCall(result={"status": "succeeded", "condition": job.condition})
        for job in jobs
    ]
    spawned = [
        SpawnedStageGateCall(
            job=job,
            function_call_id=f"fc-{index}",
            call=fake_call,
        )
        for index, (job, fake_call) in enumerate(zip(jobs, fake_calls))
    ]

    results = wait_for_stagegate_calls(spawned)

    assert len(results) == 3
    assert [result["condition"] for result in results] == [
        "baseline",
        "stage_only",
        "stagegate",
    ]
    assert [fake_call.get_calls for fake_call in fake_calls] == [1, 1, 1]


def test_wait_for_stagegate_calls_waits_all_jobs_before_raising():
    jobs = planned_jobs(mode="smoke")
    fake_calls = [
        FakeBlockingCall(exc=RuntimeError("first failure")),
        FakeBlockingCall(result={"status": "succeeded"}),
        FakeBlockingCall(exc=ValueError("third failure")),
    ]
    spawned = [
        SpawnedStageGateCall(
            job=job,
            function_call_id=f"fc-{index}",
            call=fake_call,
        )
        for index, (job, fake_call) in enumerate(zip(jobs, fake_calls))
    ]

    with pytest.raises(RuntimeError) as exc_info:
        wait_for_stagegate_calls(spawned)

    assert [fake_call.get_calls for fake_call in fake_calls] == [1, 1, 1]
    message = str(exc_info.value)
    assert "2 StageGate Modal job(s) failed" in message
    assert "baseline/retail function_call_id=fc-0" in message
    assert "stagegate/retail function_call_id=fc-2" in message


def test_write_planned_manifest_is_plan_only_and_final_hygiene_clean(tmp_path: Path):
    output = tmp_path / "batch_manifest_planned.json"

    manifest = write_planned_manifest(
        batch_id="batch",
        repo_url=DEFAULT_REPO_URL,
        repo_ref=COMMIT_SHA,
        mode="final",
        output_path=output,
    )

    assert output.exists()
    assert len(manifest["runs"]) == 9
    assert validate_final_run_manifest(manifest["runs"]) == []
    assert "modal" not in sys.modules


def test_write_planned_smoke_manifest_is_three_retail_jobs(tmp_path: Path):
    output = tmp_path / "batch_manifest_planned.json"

    manifest = write_planned_manifest(
        batch_id="batch",
        repo_url=DEFAULT_REPO_URL,
        repo_ref="stagegate",
        mode="smoke",
        output_path=output,
    )

    assert output.exists()
    assert [(run["condition"], run["domain"]) for run in manifest["runs"]] == [
        ("baseline", "retail"),
        ("stage_only", "retail"),
        ("stagegate", "retail"),
    ]
    assert all(run["mode"] == "smoke" for run in manifest["runs"])
    assert all(run["speech_complexity"] == "control" for run in manifest["runs"])
    assert any(validate_final_run_manifest(manifest["runs"]))


def test_write_planned_dev_manifest_is_mixed_replication_matrix(tmp_path: Path):
    output = tmp_path / "batch_manifest_planned.json"

    manifest = write_planned_manifest(
        batch_id="batch",
        repo_url=DEFAULT_REPO_URL,
        repo_ref=COMMIT_SHA,
        mode="dev",
        output_path=output,
    )

    assert output.exists()
    assert len(manifest["runs"]) == 9
    assert all(run["mode"] == "dev" for run in manifest["runs"])
    assert all(run["num_tasks"] == "10" for run in manifest["runs"])
    assert {
        run["domain"]: run["speech_complexity"] for run in manifest["runs"]
    } == DEV_SPEECH_COMPLEXITY_BY_DOMAIN
    assert any(validate_final_run_manifest(manifest["runs"]))


def test_write_planned_dev_manifest_accepts_candidate_selectors(tmp_path: Path):
    output = tmp_path / "batch_manifest_planned.json"

    manifest = write_planned_manifest(
        batch_id="batch",
        repo_url=DEFAULT_REPO_URL,
        repo_ref=COMMIT_SHA,
        mode="dev",
        conditions="baseline,stage_only",
        domains="retail,airline,telecom",
        output_path=output,
    )

    assert output.exists()
    assert [(run["condition"], run["domain"]) for run in manifest["runs"]] == [
        ("baseline", "retail"),
        ("baseline", "airline"),
        ("baseline", "telecom"),
        ("stage_only", "retail"),
        ("stage_only", "airline"),
        ("stage_only", "telecom"),
    ]
    assert all(run["num_tasks"] == "10" for run in manifest["runs"])
    assert all(run["max_concurrency"] == "1" for run in manifest["runs"])
    assert all(run["audio_taps"] is False for run in manifest["runs"])


def test_plan_only_cli_does_not_require_modal_or_secret(tmp_path: Path):
    output = tmp_path / "batch_manifest_planned.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/stagegate_modal_runner_config.py",
            "plan",
            "--batch-id",
            "batch",
            "--repo-ref",
            COMMIT_SHA,
            "--output",
            str(output),
            "--print-matrix",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["runs"]) == 9
    assert "tau3-voice-secrets" not in result.stdout
    assert "OPENAI_API_KEY" not in result.stdout


def test_modal_preflight_reports_missing_secret_without_secret_values():
    def fake_runner(argv, **kwargs):
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="modal client version: 1.4.2\n"
            )
        if argv[1:] == ["secret", "list", "--json"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps([{"Name": "other-secret"}]),
            )
        raise AssertionError(argv)

    result = check_modal_preflight(
        secret_name="tau3-voice-secrets",
        modal_cmd=sys.executable,
        command_runner=fake_runner,
    )

    assert result["ok"] is False
    assert result["secret_exists"] is False
    assert result["required_keys"] == list(REQUIRED_PROVIDER_SECRET_KEYS)
    assert "super-secret-value" not in json.dumps(result)


def test_modal_preflight_accepts_existing_secret():
    def fake_runner(argv, **kwargs):
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="modal client version: 1.4.2\n"
            )
        if argv[1:] == ["secret", "list", "--json"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps([{"Name": "tau3-voice-secrets"}]),
            )
        raise AssertionError(argv)

    result = check_modal_preflight(
        secret_name="tau3-voice-secrets",
        modal_cmd=sys.executable,
        command_runner=fake_runner,
    )

    assert result["ok"] is True
    assert result["secret_exists"] is True


def test_modal_secret_required_keys_include_control_and_regular_voice_personas():
    assert REQUIRED_API_SECRET_KEYS == (
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "DEEPGRAM_API_KEY",
    )
    assert REQUIRED_REGULAR_VOICE_ID_KEYS == (
        "TAU2_VOICE_ID_MILDRED_KAPLAN",
        "TAU2_VOICE_ID_ARJUN_ROY",
        "TAU2_VOICE_ID_WEI_LIN",
        "TAU2_VOICE_ID_MAMADOU_DIALLO",
        "TAU2_VOICE_ID_PRIYA_PATIL",
    )
    assert set(REQUIRED_PROVIDER_SECRET_KEYS) == {
        *REQUIRED_API_SECRET_KEYS,
        *OPTIONAL_CONTROL_VOICE_ID_KEYS,
        *REQUIRED_REGULAR_VOICE_ID_KEYS,
    }


def test_modal_runner_has_remote_secret_key_preflight_without_env_reads():
    source = Path("modal_tau3_voice_stagegate.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    verify_secret_keys = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_secret_keys"
    )
    constants = {
        node.value
        for node in ast.walk(verify_secret_keys)
        if isinstance(node, ast.Constant)
    }

    assert "required_keys" in constants
    assert "secret_name" in constants
    assert "OPENAI_API_KEY" not in constants
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr == "environ"
        for node in ast.walk(verify_secret_keys)
    )
    assert "verify_secret_keys_only" in source
    assert "REQUIRED_PROVIDER_SECRET_KEYS" in source


def test_modal_runner_waits_for_spawned_calls_before_entrypoint_exits():
    source = Path("modal_tau3_voice_stagegate.py").read_text(encoding="utf-8")

    assert "run_domain.spawn(" in source
    assert "wait_for_stagegate_calls(calls, log=logger)" in source
    assert source.index("run_domain.spawn(") < source.index(
        "wait_for_stagegate_calls(calls, log=logger)"
    )
    assert "FunctionCall.get()" in source
    assert "blocking/waiting mode" in source


def test_modal_runner_throttles_dev_job_batches():
    source = Path("modal_tau3_voice_stagegate.py").read_text(encoding="utf-8")

    assert "modal_job_concurrency" in source
    assert "resolve_modal_job_concurrency(" in source
    assert "range(0, len(jobs), job_concurrency)" in source
    assert "blocking until this Modal batch completes" in source


def test_modal_runner_packages_config_helpers_for_remote_import():
    source = Path("modal_tau3_voice_stagegate.py").read_text(encoding="utf-8")

    assert '.add_local_python_source("scripts")' in source
    assert "@app.function(\n    image=image,\n    secrets=[" in source


def test_modal_job_function_does_not_write_shared_batch_manifest():
    source = Path("modal_tau3_voice_stagegate.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    run_domain = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_domain"
    )
    constants = {
        node.value for node in ast.walk(run_domain) if isinstance(node, ast.Constant)
    }

    assert "batch_manifest_planned.json" not in constants
    assert "batch_manifest_completed.json" not in constants
