import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.stagegate_final_run_hygiene import validate_final_run_manifest
from scripts.stagegate_modal_runner_config import (
    CONDITIONS,
    DEFAULT_REPO_URL,
    DOMAINS,
    FINAL_CONSTANTS,
    OPTIONAL_CONTROL_VOICE_ID_KEYS,
    REQUIRED_API_SECRET_KEYS,
    REQUIRED_PROVIDER_SECRET_KEYS,
    REQUIRED_REGULAR_VOICE_ID_KEYS,
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
    validate_final_command,
    write_json,
    write_planned_manifest,
)

COMMIT_SHA = "1910fe2998f230bda6f6ad1b69edef4124275d63"


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

    require_full_commit_sha("stagegate", mode="smoke")


def test_smoke_mode_defaults_and_accepts_subset():
    assert planned_jobs(mode="smoke") == [
        StageGateJob(condition="baseline", domain="retail", mode="smoke")
    ]
    assert planned_jobs(mode="smoke", condition="stagegate", domain="telecom") == [
        StageGateJob(condition="stagegate", domain="telecom", mode="smoke")
    ]


def test_build_tau2_command_enforces_final_constants_and_no_task_filters():
    job = StageGateJob(condition="stagegate", domain="airline", mode="final")
    argv = build_tau2_command("batch", job)

    validate_final_command(argv)
    assert "--num-tasks" not in argv
    assert "--task-ids" not in argv
    assert argv[argv.index("--audio-native-model") + 1] == FINAL_CONSTANTS["model"]
    assert argv[argv.index("--speech-complexity") + 1] == "regular"
    assert argv[argv.index("--save-to") + 1] == "batch_stagegate_airline"


def test_validate_final_command_rejects_task_filters_and_mutated_constants():
    job = StageGateJob(condition="baseline", domain="retail", mode="final")
    argv = build_tau2_command("batch", job)

    with pytest.raises(ValueError, match="task filter"):
        validate_final_command([*argv, "--num-tasks", "1"])

    mutated = list(argv)
    mutated[mutated.index("--speech-complexity") + 1] = "control"
    with pytest.raises(ValueError, match="--speech-complexity"):
        validate_final_command(mutated)


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
    assert "OPENAI_API_KEY" not in metadata["sanitized_command"]


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


def test_modal_secret_required_keys_include_regular_voice_personas_only():
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
        *REQUIRED_REGULAR_VOICE_ID_KEYS,
    }
    assert not set(OPTIONAL_CONTROL_VOICE_ID_KEYS) & set(REQUIRED_PROVIDER_SECRET_KEYS)


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
