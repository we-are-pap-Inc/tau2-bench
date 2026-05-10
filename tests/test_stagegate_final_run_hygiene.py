from scripts.stagegate_final_run_hygiene import validate_final_run_manifest


def _run(domain: str, condition: str, **overrides):
    run = {
        "domain": domain,
        "condition": condition,
        "speech_complexity": "regular",
        "model": "gpt-realtime-2",
        "provider": "openai",
        "reasoning_effort": "medium",
        "timeout": 900,
        "seed": 123,
        "concurrency": 4,
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
    manifest[4]["timeout"] = 1200
    manifest[8]["concurrency"] = 8

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
