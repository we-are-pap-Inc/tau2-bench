import json
from unittest.mock import MagicMock

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.data_model.message import ToolCall
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.voice.audio_native.openai.stagegate import StageGateController


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


def test_trace_writer_records_stagegate_jsonl(monkeypatch, tmp_path):
    trace_path = tmp_path / "stagegate.jsonl"
    monkeypatch.setenv("TAU2_STAGEGATE_CONDITION", "stage_only")
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))

    environment = _environment()
    controller = StageGateController.from_env(
        provider="openai",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    controller.set_domain_name(environment.get_domain_name())

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
    assert all(
        event["schema_version"] == "stagegate.trace_event.v1" for event in events
    )
