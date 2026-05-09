import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.data_model.message import ToolCall
from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.orchestrator import BaseOrchestrator
from tau2.runner import batch as runner_batch
from tau2.runner.simulation import _trace_final_outcome
from tau2.voice.audio_native.openai.stagegate import StageGateController, TraceEvent
from tau2.voice.audio_native.openai.stagegate.trace import JsonlTraceWriter


def _test_tool(arg: str) -> str:
    """A test tool.

    Args:
        arg: A test argument.

    Returns:
        A test result.
    """
    return f"result:{arg}"


class StageGateToolkit(ToolKitBase):
    def __init__(self):
        self.write_count = 0

    @is_tool(ToolType.READ)
    def get_account(self, account_id: str) -> dict:
        """Get account details.

        Args:
            account_id: The account ID.

        Returns:
            Account details.
        """
        return {"account_id": account_id, "status": "active"}

    @is_tool(ToolType.WRITE)
    def update_account(self, account_id: str, plan_name: str) -> dict:
        """Update an account plan.

        Args:
            account_id: The account ID.
            plan_name: The new plan name.

        Returns:
            Updated account details.
        """
        self.write_count += 1
        return {"account_id": account_id, "plan_name": plan_name}


def _environment() -> Environment:
    return Environment(
        domain_name="mock",
        policy="Public policy.",
        tools=StageGateToolkit(),
    )


def _controller(environment: Environment) -> StageGateController:
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    controller.set_domain_name(environment.get_domain_name())
    return controller


def _orchestrator_shell(environment: Environment) -> FullDuplexOrchestrator:
    orchestrator = FullDuplexOrchestrator.__new__(FullDuplexOrchestrator)
    orchestrator.environment = environment
    orchestrator.num_errors = 0
    return orchestrator


def test_stagegate_agent_adds_advance_stage_only_when_enabled(monkeypatch):
    monkeypatch.setenv("TAU2_STAGEGATE_CONDITION", "stage_only")
    adapter = MagicMock()
    adapter.is_connected = False
    adapter.connect.side_effect = lambda *args, **kwargs: setattr(
        adapter, "is_connected", True
    )

    agent = DiscreteTimeAudioNativeAgent(
        tools=[Tool(_test_tool)],
        domain_policy="Policy.",
        adapter=adapter,
        provider="openai",
    )
    agent.get_init_state()

    tool_names = [tool.name for tool in adapter.connect.call_args.kwargs["tools"]]
    assert tool_names == ["_test_tool", "advance_stage"]
    assert (
        "StageGate operating rules" in adapter.connect.call_args.kwargs["system_prompt"]
    )


def test_stagegate_agent_leaves_baseline_tools_unchanged(monkeypatch):
    monkeypatch.delenv("TAU2_STAGEGATE_CONDITION", raising=False)
    adapter = MagicMock()
    adapter.is_connected = False
    adapter.connect.side_effect = lambda *args, **kwargs: setattr(
        adapter, "is_connected", True
    )

    agent = DiscreteTimeAudioNativeAgent(
        tools=[Tool(_test_tool)],
        domain_policy="Policy.",
        adapter=adapter,
        provider="openai",
    )
    agent.get_init_state()

    tool_names = [tool.name for tool in adapter.connect.call_args.kwargs["tools"]]
    assert tool_names == ["_test_tool"]
    assert (
        "StageGate operating rules"
        not in adapter.connect.call_args.kwargs["system_prompt"]
    )


def test_stagegate_condition_currently_enables_stage_only(monkeypatch):
    monkeypatch.setenv("TAU2_STAGEGATE_CONDITION", "stagegate")
    adapter = MagicMock()
    adapter.is_connected = False
    adapter.connect.side_effect = lambda *args, **kwargs: setattr(
        adapter, "is_connected", True
    )

    agent = DiscreteTimeAudioNativeAgent(
        tools=[Tool(_test_tool)],
        domain_policy="Policy.",
        adapter=adapter,
        provider="openai",
    )
    agent.get_init_state()

    tool_names = [tool.name for tool in adapter.connect.call_args.kwargs["tools"]]
    assert tool_names == ["_test_tool", "advance_stage"]
    assert (
        "StageGate operating rules" in adapter.connect.call_args.kwargs["system_prompt"]
    )
    assert not hasattr(agent.stagegate_controller, "ledger")
    assert not hasattr(agent.stagegate_controller, "validator")


def test_advance_stage_returns_packet_without_domain_tool_execution():
    environment = _environment()
    controller = _controller(environment)
    orchestrator = _orchestrator_shell(environment)
    environment.get_response = MagicMock(side_effect=AssertionError("unexpected call"))

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_stage",
            name="advance_stage",
            arguments={
                "current_stage": "understand_intent",
                "observed_facts": ["customer confirmed the requested plan change"],
                "last_action": "asked for confirmation",
            },
        ),
        tick_id=3,
    )

    packet = json.loads(result.content)
    assert packet["schema_version"] == "stagegate.stage_packet.v1"
    assert packet["stage"] == "identify_or_authenticate"
    assert environment.tools.write_count == 0
    environment.get_response.assert_not_called()


def test_stage_only_does_not_block_normal_domain_tools():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    orchestrator = _orchestrator_shell(environment)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=2,
    )

    content = json.loads(result.content)
    assert result.error is False
    assert content["plan_name"] == "premium"
    assert environment.tools.write_count == 1
    assert orchestrator.num_errors == 0


def test_trace_event_uses_canonical_schema():
    event = TraceEvent(
        event_type="model_function_call",
        condition="stage_only",
        run_id="run_123",
        domain="mock",
        task_id="task_123",
        sim_id="sim_123",
        tick_index=4,
        tool_name="get_account",
        tool_args={"account_id": "acct_123"},
    )

    data = json.loads(event.model_dump_json())
    assert data["schema_version"] == "stagegate.trace.v1"
    assert data["ts"].endswith("Z")
    assert data["run_id"] == "run_123"
    assert data["tick_index"] == 4
    assert data["leakage_risk"] == "none"


def test_trace_writer_is_noop_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TAU2_TRACE_JSONL", raising=False)
    trace_path = tmp_path / "trace_events.jsonl"
    writer = JsonlTraceWriter.from_env()

    writer.write(TraceEvent(event_type="run_start", condition="baseline"))

    assert not trace_path.exists()


def test_trace_writer_records_stagegate_jsonl(monkeypatch, tmp_path):
    trace_path = tmp_path / "stagegate.jsonl"
    monkeypatch.setenv("TAU2_STAGEGATE_CONDITION", "stage_only")
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))
    monkeypatch.setenv("TAU2_TRACE_RUN_ID", "run_stagegate")

    environment = _environment()
    controller = StageGateController.from_env(
        provider="openai",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    controller.set_trace_context(
        domain_name=environment.get_domain_name(),
        task_id="task_1",
        sim_id="sim_1",
    )

    controller.handle_advance_stage(
        ToolCall(
            id="call_stage",
            name="advance_stage",
            arguments={
                "current_stage": "understand_intent",
                "observed_facts": [],
                "last_action": "started",
            },
        ),
        tick_id=1,
    )

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "advance_stage_call",
        "stage_packet_returned",
    ]
    assert all(event["schema_version"] == "stagegate.trace.v1" for event in events)
    assert all(event["run_id"] == "run_stagegate" for event in events)
    assert all(event["domain"] == "mock" for event in events)
    assert events[0]["tool_name"] == "advance_stage"
    assert events[0]["tick_index"] == 1
    assert events[1]["stage"] == "identify_or_authenticate"
    assert events[1]["latency_ms"] >= 0


def test_domain_tool_trace_events_are_emitted(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))

    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    controller.set_trace_context(
        domain_name=environment.get_domain_name(),
        task_id="task_2",
        sim_id="sim_2",
    )
    orchestrator = _orchestrator_shell(environment)
    tool_call = ToolCall(
        id="call_read",
        name="get_account",
        arguments={"account_id": "acct_123"},
    )

    controller.trace_model_function_call(tool_call, tick_id=7)
    result = orchestrator._execute_stagegate_tool_call(
        controller,
        tool_call,
        tick_id=7,
    )

    assert json.loads(result.content)["status"] == "active"
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "model_function_call",
        "domain_tool_call",
        "domain_tool_result",
    ]
    assert all(event["task_id"] == "task_2" for event in events)
    assert all(event["sim_id"] == "sim_2" for event in events)
    assert events[0]["tool_args"] == {"account_id": "acct_123"}
    assert events[2]["latency_ms"] >= 0
    assert events[2]["payload"]["tool_error"] is False


def test_final_outcome_trace_is_posthoc(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))

    environment = _environment()
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    controller.set_trace_context(
        domain_name=environment.get_domain_name(),
        task_id="task_3",
        sim_id="sim_3",
    )
    simulation = SimulationRun(
        id="sim_3",
        task_id="task_3",
        start_time="2026-05-08T00:00:00",
        end_time="2026-05-08T00:00:01",
        duration=1.0,
        termination_reason="agent_stop",
        reward_info=RewardInfo(reward=1.0),
    )

    controller.trace_final_outcome(simulation)

    event = json.loads(trace_path.read_text().strip())
    assert event["event_type"] == "final_outcome"
    assert event["visible_to_agent"] is False
    assert event["leakage_risk"] == "posthoc_evaluator"
    assert event["reward"] == 1.0
    assert event["passed"] is True


def test_final_outcome_trace_uses_orchestrator_trial(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))

    environment = _environment()
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    orchestrator = SimpleNamespace(
        agent=SimpleNamespace(stagegate_controller=controller),
        environment=environment,
        task=SimpleNamespace(id="task_4"),
        trial=2,
    )
    simulation = SimulationRun(
        id="sim_4",
        task_id="task_4",
        start_time="2026-05-08T00:00:00",
        end_time="2026-05-08T00:00:01",
        duration=1.0,
        termination_reason="agent_stop",
        reward_info=RewardInfo(reward=0.0),
    )

    _trace_final_outcome(orchestrator, simulation)

    event = json.loads(trace_path.read_text().strip())
    assert event["event_type"] == "final_outcome"
    assert event["trial"] == 2
    assert event["sim_id"] == "sim_4"


def test_run_single_task_attaches_trial_for_trace_context(monkeypatch):
    orchestrator = SimpleNamespace()

    monkeypatch.setattr(
        runner_batch,
        "build_orchestrator",
        lambda *args, **kwargs: orchestrator,
    )
    monkeypatch.setattr(runner_batch, "_build_env_kwargs", lambda *args, **kwargs: None)

    def run_simulation(orchestrator_arg, **kwargs):
        assert orchestrator_arg is orchestrator
        assert orchestrator_arg.trial == 4
        return SimulationRun(
            id="sim_6",
            task_id="task_6",
            start_time="2026-05-08T00:00:00",
            end_time="2026-05-08T00:00:01",
            duration=1.0,
            termination_reason="agent_stop",
            reward_info=RewardInfo(reward=1.0),
        )

    monkeypatch.setattr(runner_batch, "run_simulation", run_simulation)

    result = runner_batch.run_single_task(
        SimpleNamespace(
            domain="mock",
            effective_agent="agent",
            effective_user="user",
        ),
        SimpleNamespace(id="task_6"),
        trial=4,
    )

    assert result.id == "sim_6"


def test_full_duplex_run_traces_run_end_on_exception(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))

    environment = _environment()
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    orchestrator = _orchestrator_shell(environment)
    orchestrator.agent = SimpleNamespace(stagegate_controller=controller)
    orchestrator.task = SimpleNamespace(id="task_5")
    orchestrator.simulation_id = "sim_5"
    orchestrator.trial = 3
    orchestrator._run_start_perf = None

    def raise_from_base_run(self):
        self._run_start_perf = time.perf_counter()
        raise RuntimeError("sim failed")

    monkeypatch.setattr(BaseOrchestrator, "run", raise_from_base_run)

    with pytest.raises(RuntimeError, match="sim failed"):
        FullDuplexOrchestrator.run(orchestrator)

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["run_start", "run_end"]
    assert events[0]["trial"] == 3
    assert events[1]["trial"] == 3
    assert events[1]["payload"]["termination_reason"] == "exception"
    assert events[1]["payload"]["duration_seconds"] >= 0
