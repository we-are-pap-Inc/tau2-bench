from scripts.stagegate_final_run_hygiene import validate_final_run_manifest


def _run(domain: str, condition: str, **overrides):
    repo_ref = "1910fe2998f230bda6f6ad1b69edef4124275d63"
    run = {
        "domain": domain,
        "condition": condition,
        "mode": "final",
        "requested_repo_ref": repo_ref,
        "speech_complexity": "regular",
        "model": "gpt-realtime-2",
        "provider": "openai",
        "reasoning_effort": "high",
        "tick_duration": "0.2",
        "timeout": "1200",
        "max_steps_seconds": "1200",
        "seed": "300",
        "concurrency": "1",
        "max_concurrency": "1",
    }
    run.update(overrides)
    return run


def _valid_manifest():
    return [
        _run(domain, condition)
        for domain in ("retail", "airline", "telecom")
        for condition in ("baseline", "stage_only", "stagegate")
    ]


def test_stagegate_final_run_hygiene_accepts_required_matrix():
    assert validate_final_run_manifest(_valid_manifest()) == []


def test_stagegate_final_run_hygiene_rejects_task_filters():
    manifest = _valid_manifest()
    manifest[0]["args"] = ["tau2", "run", "--num-tasks", "5"]
    manifest[1]["task_ids"] = ["retail_1"]

    errors = validate_final_run_manifest(manifest)

    assert any("task filters" in error for error in errors)


def test_stagegate_final_run_hygiene_rejects_non_regular_speech():
    manifest = _valid_manifest()
    manifest[0]["speech_complexity"] = "control"

    errors = validate_final_run_manifest(manifest)

    assert any("speech_complexity" in error for error in errors)


def test_stagegate_final_run_hygiene_rejects_inconsistent_settings():
    manifest = _valid_manifest()
    manifest[1]["model"] = "different-model"
    manifest[4]["timeout"] = "900"
    manifest[4]["max_steps_seconds"] = "900"
    manifest[8]["concurrency"] = 8
    manifest[8]["max_concurrency"] = 8

    errors = validate_final_run_manifest(manifest)

    assert any("retail: inconsistent model" in error for error in errors)
    assert any("airline: inconsistent timeout" in error for error in errors)
    assert any("telecom: inconsistent concurrency" in error for error in errors)


def test_stagegate_final_run_hygiene_requires_domains_and_conditions():
    manifest = [
        run
        for run in _valid_manifest()
        if not (run["domain"] == "telecom" and run["condition"] == "stagegate")
    ]

    errors = validate_final_run_manifest(manifest)

    assert any("telecom: missing required conditions" in error for error in errors)


def test_stagegate_final_run_hygiene_rejects_smoke_and_non_sha_refs():
    manifest = _valid_manifest()
    manifest[0]["mode"] = "smoke"
    manifest[1]["requested_repo_ref"] = "stagegate"

    errors = validate_final_run_manifest(manifest)

    assert any("mode='final'" in error for error in errors)
    assert any("40-character repo SHA" in error for error in errors)
