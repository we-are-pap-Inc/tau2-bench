from scripts.stagegate_prohibited_diff_guard import (
    is_prohibited_stagegate_path,
    prohibited_paths,
)


def test_stagegate_guard_allows_runtime_tests_docs_and_scripts():
    allowed = [
        "src/tau2/voice/audio_native/openai/stagegate/controller.py",
        "src/tau2/orchestrator/full_duplex_orchestrator.py",
        "src/tau2/runner/simulation.py",
        "tests/test_streaming/test_stagegate.py",
        "docs/stagegate/05-tracing-and-visualization.md",
        "scripts/stagegate_posthoc_outcomes.py",
    ]

    assert prohibited_paths(allowed) == []


def test_stagegate_guard_blocks_benchmark_validity_paths():
    blocked = [
        "data/tau2/domains/retail/tasks.json",
        "data/tau2/domains/airline/policy.md",
        "data/tau2/domains/telecom/db.json",
        "src/tau2/domains/retail/tools.py",
        "src/tau2/domains/telecom/user_tools.py",
        "src/tau2/evaluator/evaluator.py",
        "src/tau2/metrics/agent_metrics.py",
        "src/tau2/user/user_simulator.py",
        "src/tau2/user_simulation_voice_presets.py",
        "src/tau2/scripts/recombine_rewards.py",
    ]

    assert prohibited_paths(blocked) == blocked


def test_stagegate_guard_normalizes_relative_paths():
    assert is_prohibited_stagegate_path("./data/tau2/domains/mock/tasks.json")
