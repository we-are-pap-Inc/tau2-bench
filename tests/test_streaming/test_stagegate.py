import asyncio
import json
import re
import time
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.orchestrator import BaseOrchestrator
from tau2.runner import batch as runner_batch
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import InputAudioTranscriptionCompletedEvent
from tau2.voice.audio_native.openai.stagegate import (
    EntityLedger,
    EvidenceSource,
    LedgerStatus,
    PreWriteValidator,
    StageGateController,
    TraceEvent,
)
from tau2.voice.audio_native.openai.stagegate.trace import JsonlTraceWriter
from tau2.voice.audio_native.tick_result import TickResult


def _test_tool(arg: str) -> str:
    """A test tool.

    Args:
        arg: A test argument.

    Returns:
        A test result.
    """
    return f"result:{arg}"


def get_customer_by_id(customer_id: str) -> dict:
    """Get customer details.

    Args:
        customer_id: The customer ID.

    Returns:
        Customer details.
    """
    return {"customer_id": customer_id}


def get_details_by_id(id: str) -> dict:
    """Get details for a telecom object.

    Args:
        id: The object ID.

    Returns:
        Object details.
    """
    return {"id": id}


def send_payment_request(customer_id: str, bill_id: str) -> str:
    """Send a payment request.

    Args:
        customer_id: The customer ID.
        bill_id: The bill ID.

    Returns:
        Request status.
    """
    return "sent"


class StageGateToolkit(ToolKitBase):
    def __init__(self):
        self.write_count = 0
        self.service_tasks = {
            "service_task_1": {
                "task_id": "service_task_1",
                "status": "pending",
            }
        }

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

    @is_tool(ToolType.READ)
    def get_users(self) -> list[dict]:
        """Get users.

        Returns:
            User records.
        """
        return [{"user_id": "user_1", "name": "Test User"}]

    @is_tool(ToolType.READ)
    def get_tasks(self) -> list[dict]:
        """Get service tasks.

        Returns:
            Service task records.
        """
        return list(self.service_tasks.values())

    @is_tool(ToolType.WRITE)
    def update_task_status(self, task_id: str, status: str) -> dict:
        """Update service task status.

        Args:
            task_id: The domain service task reference.
            status: The new service task status.

        Returns:
            Updated service task details.
        """
        self.write_count += 1
        self.service_tasks[task_id]["status"] = status
        return dict(self.service_tasks[task_id])


def _environment(domain_name: str = "mock") -> Environment:
    return Environment(
        domain_name=domain_name,
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


class ScriptedStageGateAgent:
    def __init__(self, controller: StageGateController, tool_call: ToolCall):
        self.stagegate_controller = controller
        self.tool_call = tool_call
        self.received_chunks = []

    @classmethod
    def is_stop(cls, message):
        return False

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
    ):
        self.received_chunks.append(participant_chunk)
        return (
            AssistantMessage(
                role="assistant",
                content=None,
                contains_speech=False,
                tool_calls=[self.tool_call],
            ),
            state,
        )


class ScriptedUtteranceAgent:
    def __init__(self, controller: StageGateController, content: str):
        self.stagegate_controller = controller
        self.content = content

    @classmethod
    def is_stop(cls, message):
        return False

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
    ):
        return AssistantMessage.text(self.content), state


class ScriptedUser:
    def __init__(self, chunks: list[UserMessage]):
        self.chunks = list(chunks)

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
    ):
        if self.chunks:
            return self.chunks.pop(0), state
        return UserMessage(role="user", content=None, contains_speech=False), state


def _visibility_orchestrator(
    environment: Environment,
    controller: StageGateController,
    *,
    user_chunks: list[UserMessage],
) -> FullDuplexOrchestrator:
    orchestrator = _orchestrator_shell(environment)
    write_call = ToolCall(
        id="call_write",
        name="update_account",
        arguments={"account_id": "acct_123", "plan_name": "premium"},
    )
    orchestrator.agent = ScriptedStageGateAgent(controller, write_call)
    orchestrator.user = ScriptedUser(user_chunks)
    orchestrator.agent_state = SimpleNamespace()
    orchestrator.user_state = SimpleNamespace()
    orchestrator.current_agent_chunk = AssistantMessage.text(
        "Please confirm the account change."
    )
    orchestrator.current_user_chunk = UserMessage.text("I want the premium plan.")
    orchestrator.pending_agent_tool_results = None
    orchestrator.pending_user_tool_results = None
    orchestrator.ticks = []
    orchestrator.tick_duration_seconds = None
    orchestrator.step_count = 0
    orchestrator.done = False
    orchestrator.termination_reason = None
    orchestrator.task = SimpleNamespace(id="task_visibility")
    orchestrator.simulation_id = "sim_visibility"
    return orchestrator


def _prepare_validated_account_change(
    orchestrator: FullDuplexOrchestrator,
    controller: StageGateController,
) -> None:
    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tick_id=1,
    )
    controller.record_assistant_utterance(
        AssistantMessage.text(
            "I will update account acct_123 to premium. "
            "This will change the account plan status."
        ),
        tick_id=0,
    )


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


def test_stage_only_condition_has_no_ledger(monkeypatch):
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

    assert not hasattr(agent.stagegate_controller, "ledger")
    assert not hasattr(agent.stagegate_controller, "validator")


def test_stagegate_condition_enables_entity_ledger(monkeypatch):
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
    assert hasattr(agent.stagegate_controller, "ledger")
    assert hasattr(agent.stagegate_controller, "validator")


def test_entity_ledger_initializes_domain_slots_and_serializes():
    ledger = EntityLedger.for_domain("retail")

    assert list(ledger.slots) == [
        "customer_name",
        "email",
        "phone",
        "order_id",
        "item_id",
        "return_reason",
        "refund_or_exchange_intent",
        "address",
        "payment_method",
        "confirmation",
    ]
    data = json.loads(ledger.model_dump_json())
    assert data["domain_name"] == "retail"
    assert data["slots"]["order_id"]["status"] == "missing"
    assert data["slots"]["order_id"]["evidence"] == []


def test_ledger_updates_from_visible_model_tool_arguments():
    environment = _environment(domain_name="retail")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    controller.trace_model_function_call(
        ToolCall(
            id="call_cancel",
            name="cancel_pending_order",
            arguments={"order_id": "O-12345", "reason": "ordered by mistake"},
        ),
        tick_id=9,
    )

    order_slot = controller.ledger.slots["order_id"]
    assert order_slot.status is LedgerStatus.HEARD_NOT_CONFIRMED
    assert order_slot.value == "O-12345"
    assert order_slot.evidence[-1].source == EvidenceSource.MODEL_TOOL_ARGUMENT.value
    assert order_slot.evidence[-1].event_id == "call_cancel"
    assert order_slot.evidence[-1].tick_index == 9
    assert controller.ledger.slots["return_reason"].value == "ordered by mistake"
    assert controller.ledger.slots["refund_or_exchange_intent"].value == "cancel"


def test_ledger_updates_from_successful_official_tool_results():
    environment = _environment(domain_name="telecom")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    tool_call = ToolCall(
        id="call_customer",
        name="get_customer_by_id",
        arguments={"customer_id": "cust_123"},
    )
    tool_result = ToolMessage(
        id="call_customer",
        role="tool",
        content=json.dumps(
            {
                "customer_id": "cust_123",
                "full_name": "Ada Lovelace",
                "phone_number": "+1-555-0100",
                "address": {
                    "street": "1 Algorithm Way",
                    "city": "London",
                    "state": "CA",
                    "zip_code": "90001",
                },
            }
        ),
        error=False,
    )

    controller.trace_domain_tool_result(tool_call, tool_result, tick_id=4)

    assert controller.ledger.slots["account_id"].status is LedgerStatus.TOOL_VERIFIED
    assert controller.ledger.slots["account_id"].value == "cust_123"
    assert controller.ledger.slots["customer_name"].value == "Ada Lovelace"
    assert controller.ledger.slots["phone_line"].value == "+1-555-0100"
    assert controller.ledger.slots["service_address"].value["street"] == (
        "1 Algorithm Way"
    )


def test_ledger_does_not_treat_product_or_plan_names_as_customer_names():
    retail_ledger = EntityLedger.for_domain("retail")
    retail_ledger.update_from_tool_result(
        tool_name="get_product_details",
        content=json.dumps(
            {
                "product_id": "prod_123",
                "name": "Everyday Backpack",
                "variants": {"v_1": {"item_id": "item_1"}},
            }
        ),
        event_id="call_product",
        tick_index=1,
    )

    assert retail_ledger.slots["customer_name"].status is LedgerStatus.MISSING
    assert retail_ledger.slots["item_id"].value == "item_1"

    telecom_ledger = EntityLedger.for_domain("telecom")
    telecom_ledger.update_from_tool_result(
        tool_name="get_details_by_id",
        content=json.dumps(
            {
                "plan_id": "plan_unlimited",
                "name": "Unlimited Plus",
                "data_limit_gb": 100,
            }
        ),
        event_id="call_plan",
        tick_index=1,
    )

    assert telecom_ledger.slots["customer_name"].status is LedgerStatus.MISSING
    assert telecom_ledger.slots["plan_name"].value == "Unlimited Plus"


def test_errored_tool_results_do_not_update_ledger():
    environment = _environment(domain_name="telecom")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    controller.trace_domain_tool_result(
        ToolCall(
            id="call_customer",
            name="get_customer_by_id",
            arguments={"customer_id": "cust_123"},
        ),
        ToolMessage(
            id="call_customer",
            role="tool",
            content=json.dumps({"customer_id": "cust_123"}),
            error=True,
        ),
        tick_id=4,
    )

    assert controller.ledger.slots["account_id"].status is LedgerStatus.MISSING


def test_ledger_contradiction_surfaces_as_ambiguous_stage_fact():
    environment = _environment(domain_name="telecom")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    controller.trace_model_function_call(
        ToolCall(
            id="call_first",
            name="get_customer_by_id",
            arguments={"customer_id": "cust_123"},
        ),
        tick_id=1,
    )
    controller.trace_model_function_call(
        ToolCall(
            id="call_second",
            name="get_customer_by_id",
            arguments={"customer_id": "cust_999"},
        ),
        tick_id=2,
    )

    slot = controller.ledger.slots["account_id"]
    assert slot.status is LedgerStatus.CONTRADICTED
    assert slot.alternatives == ["cust_999"]

    packet_result = controller.handle_advance_stage(
        ToolCall(
            id="call_stage",
            name="advance_stage",
            arguments={
                "current_stage": "collect_required_exact_entities",
                "observed_facts": [],
                "last_action": "customer lookup argument changed",
            },
        ),
        tick_id=3,
    )
    packet = json.loads(packet_result.content)
    assert "account_id: cust_123, cust_999" in packet["ambiguous_facts"]
    assert (
        packet["ask_next"]
        == "Clarify the exact value for: account_id: cust_123, cust_999."
    )


def test_ledger_update_trace_events_are_emitted(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))
    environment = _environment(domain_name="telecom")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    controller.set_trace_context(
        benchmark_task_id="task_ledger",
        sim_id="sim_ledger",
    )

    controller.trace_model_function_call(
        ToolCall(
            id="call_customer",
            name="get_customer_by_id",
            arguments={"customer_id": "cust_123"},
        ),
        tick_id=7,
    )

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "model_function_call",
        "ledger_update",
    ]
    ledger_event = events[1]
    assert ledger_event["source"] == EvidenceSource.MODEL_TOOL_ARGUMENT.value
    assert ledger_event["visible_to_agent"] is True
    assert ledger_event["tick_index"] == 7
    assert ledger_event["ledger_delta"]["account_id"]["status"] == (
        "heard_not_confirmed"
    )
    assert ledger_event["ledger_delta"]["account_id"]["evidence"]["event_id"] == (
        "call_customer"
    )


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
        condition="stage_only",
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


def test_baseline_trace_jsonl_does_not_route_tools_through_stagegate(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(tmp_path / "trace_events.jsonl"))
    environment = _environment()
    controller = StageGateController(
        condition="baseline",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    tool_call = ToolCall(
        id="call_write",
        name="update_account",
        arguments={"account_id": "acct_123", "plan_name": "premium"},
    )
    agent = ScriptedStageGateAgent(controller, tool_call)
    orchestrator.agent = agent

    def fail_stagegate_execution(*args, **kwargs):
        raise AssertionError("baseline routed through StageGate wrapper")

    monkeypatch.setattr(
        orchestrator,
        "_execute_stagegate_tool_call",
        fail_stagegate_execution,
    )

    _, _, _, tool_results = orchestrator._process_participant_turn(
        participant=agent,
        state=SimpleNamespace(),
        incoming_chunk=None,
        is_agent=True,
        pending_tool_results=None,
        tick_id=1,
    )

    assert tool_results[0].error is False
    assert json.loads(tool_results[0].content)["plan_name"] == "premium"
    assert environment.tools.write_count == 1

    events = [
        json.loads(line)
        for line in (tmp_path / "trace_events.jsonl").read_text().splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "model_function_call",
        "domain_tool_call",
        "domain_tool_result",
    ]
    assert all(event["condition"] == "baseline" for event in events)
    assert all(not event["event_type"].startswith("validator_") for event in events)


def test_baseline_trace_jsonl_records_assistant_utterance_without_stagegate(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(tmp_path / "trace_events.jsonl"))
    environment = _environment()
    controller = StageGateController(
        condition="baseline",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    agent = ScriptedUtteranceAgent(controller, "I can help with that account.")
    orchestrator.agent = agent

    orchestrator._process_participant_turn(
        participant=agent,
        state=SimpleNamespace(),
        incoming_chunk=None,
        is_agent=True,
        pending_tool_results=None,
        tick_id=1,
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "trace_events.jsonl").read_text().splitlines()
    ]
    assert [event["event_type"] for event in events] == ["assistant_audio_event"]
    assert events[0]["source"] == EvidenceSource.ASSISTANT_UTTERANCE.value
    assert events[0]["payload"]["content"] == "I can help with that account."


def test_validator_never_mutates_domain_state_when_blocking():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
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

    packet = json.loads(result.content)
    assert result.error is True
    assert packet["schema_version"] == "stagegate.stage_packet.v1"
    assert environment.tools.write_count == 0
    assert orchestrator.num_errors == 1


def test_missing_confirmation_blocks_write(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tick_id=1,
    )
    controller.record_assistant_utterance(
        AssistantMessage.text(
            "I will update account acct_123 to premium. "
            "This will change the account plan status."
        ),
        tick_id=2,
    )

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=3,
    )

    packet = json.loads(result.content)
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert result.error is True
    assert packet["stage"] == "propose_action_and_confirm"
    assert packet["missing_facts"] == ["missing_confirmation"]
    assert environment.tools.write_count == 0
    assert events[-2]["event_type"] == "validator_check"
    assert events[-1]["event_type"] == "validator_block"
    assert events[-1]["validator_reason"] == "missing_confirmation"


def test_confirmed_exact_identifier_allows_lookup_or_write_when_policy_allows():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    read_result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tick_id=1,
    )
    assert read_result.error is False
    assert json.loads(read_result.content)["account_id"] == "acct_123"

    controller.record_assistant_utterance(
        AssistantMessage.text(
            "I will update account acct_123 to premium. "
            "This will change the account plan status."
        ),
        tick_id=2,
    )
    controller.record_agent_visible_user_transcript("Yes, I confirm.", tick_id=3)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=4,
    )

    content = json.loads(result.content)
    assert result.error is False
    assert content["plan_name"] == "premium"
    assert environment.tools.write_count == 1
    assert orchestrator.num_errors == 0


def test_service_task_ref_preserves_mock_task_write_validation():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    user_result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(id="call_users", name="get_users", arguments={}),
        tick_id=1,
    )
    task_result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(id="call_tasks", name="get_tasks", arguments={}),
        tick_id=2,
    )
    assert user_result.error is False
    assert task_result.error is False

    controller.record_assistant_utterance(
        AssistantMessage.text(
            "I will update service task service_task_1 to completed. "
            "This will change the service task status."
        ),
        tick_id=3,
    )
    controller.record_agent_visible_user_transcript("Yes, I confirm.", tick_id=4)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_update_task",
            name="update_task_status",
            arguments={"task_id": "service_task_1", "status": "completed"},
        ),
        tick_id=5,
    )

    content = json.loads(result.content)
    assert result.error is False
    assert content["status"] == "completed"
    assert environment.tools.write_count == 1
    assert controller.validator.state.verified_identifiers["service_task_ref"] == {
        "service_task_1"
    }


def test_clean_user_message_content_is_rejected_as_runtime_evidence():
    environment = _environment(domain_name="retail")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    with pytest.raises(ValueError, match="simulator gold text"):
        controller.record_visible_message(
            UserMessage.text("Yes, I confirm."),
            is_agent=False,
            tick_id=4,
        )

    assert all(
        slot.status is not LedgerStatus.USER_CONFIRMED
        for slot in controller.ledger.slots.values()
    )
    assert controller.validator.state.last_user_confirmation is None


def test_same_tick_user_confirmation_cannot_satisfy_validator():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _visibility_orchestrator(
        environment,
        controller,
        user_chunks=[UserMessage.text("Yes, I confirm.")],
    )
    _prepare_validated_account_change(orchestrator, controller)

    orchestrator.step()

    tick = orchestrator.ticks[-1]
    packet = json.loads(tick.agent_tool_results[0].content)
    assert tick.user_chunk.content == "Yes, I confirm."
    assert orchestrator.agent.received_chunks[-1].content == "I want the premium plan."
    assert tick.agent_tool_results[0].error is True
    assert packet["missing_facts"] == ["missing_confirmation"]
    assert environment.tools.write_count == 0


def test_next_tick_clean_user_content_does_not_satisfy_validator():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _visibility_orchestrator(
        environment,
        controller,
        user_chunks=[UserMessage.text("Yes, I confirm.")],
    )
    _prepare_validated_account_change(orchestrator, controller)

    orchestrator.step()
    first_tick_result = orchestrator.ticks[-1].agent_tool_results[0]
    assert first_tick_result.error is True
    assert environment.tools.write_count == 0

    orchestrator.step()

    second_tick_result = orchestrator.ticks[-1].agent_tool_results[0]
    assert orchestrator.agent.received_chunks[-1].content == "Yes, I confirm."
    assert second_tick_result.error is True
    assert json.loads(second_tick_result.content)["missing_facts"] == [
        "missing_confirmation"
    ]
    assert environment.tools.write_count == 0


def test_agent_visible_user_transcript_satisfies_confirmation():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tick_id=1,
    )
    controller.record_assistant_utterance(
        AssistantMessage.text(
            "I will update account acct_123 to premium. "
            "This will change the account plan status."
        ),
        tick_id=2,
    )
    controller.record_agent_visible_user_transcript("Yes, I confirm.", tick_id=3)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=4,
    )

    assert result.error is False
    assert json.loads(result.content)["plan_name"] == "premium"
    assert environment.tools.write_count == 1


def test_openai_adapter_records_input_transcription_event():
    adapter = DiscreteTimeOpenAIAdapter(
        tick_duration_ms=200,
        provider=MagicMock(),
    )
    tick_result = TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0,
        bytes_per_tick=1600,
        bytes_per_second=8000,
    )

    asyncio.run(
        adapter._process_event(
            tick_result,
            InputAudioTranscriptionCompletedEvent(
                event_id="evt_transcript",
                type="conversation.item.input_audio_transcription.completed",
                item_id="item_user",
                transcript="Yes, I confirm.",
            ),
        )
    )

    assert tick_result.user_transcripts == ["Yes, I confirm."]


def test_agent_wires_provider_user_transcript_to_stagegate_confirmation(monkeypatch):
    monkeypatch.setenv("TAU2_STAGEGATE_CONDITION", "stagegate")
    environment = _environment()
    adapter = MagicMock()
    adapter.is_connected = True
    adapter.run_tick.return_value = TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0,
        bytes_per_tick=8000,
        bytes_per_second=8000,
        user_transcripts=["Yes, I confirm."],
    )
    agent = DiscreteTimeAudioNativeAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        adapter=adapter,
        provider="openai",
    )
    controller = agent.stagegate_controller
    controller.set_domain_name(environment.get_domain_name())
    controller.validator.record_tool_result(
        tool_call=ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tool_result=ToolMessage(
            id="call_read",
            role="tool",
            content=json.dumps({"account_id": "acct_123", "status": "active"}),
            error=False,
        ),
        tick_index=0,
    )
    controller.record_assistant_utterance(
        AssistantMessage.text(
            "I will update account acct_123 to premium. "
            "This will change the account plan status."
        ),
        tick_id=0,
    )
    state = agent.get_init_state()

    agent.get_next_chunk(
        state,
        participant_chunk=UserMessage.text("clean simulator text is ignored"),
    )
    decision = controller.validate_tool_call(
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        )
    )

    assert controller.validator.state.last_user_confirmation == "Yes, I confirm."
    assert decision.decision == "allow"


def test_validator_uses_confirmation_only_after_visible_recording():
    environment = _environment()
    validator = PreWriteValidator(
        domain_name="mock",
        domain_policy="Public policy.",
        tools=environment.get_tools(),
    )
    write_call = ToolCall(
        id="call_write",
        name="update_account",
        arguments={"account_id": "acct_123", "plan_name": "premium"},
    )
    validator.record_tool_result(
        tool_call=ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tool_result=ToolMessage(
            id="call_read",
            role="tool",
            content=json.dumps({"account_id": "acct_123", "status": "active"}),
            error=False,
        ),
        tick_index=1,
    )
    validator.record_assistant_utterance(
        content=(
            "I will update account acct_123 to premium. "
            "This will change the account plan status."
        ),
        tick_index=2,
    )

    before_delivery = validator.validate(write_call)
    validator.record_user_confirmation_evidence(
        content="Yes, I confirm.",
        tick_index=3,
        source=EvidenceSource.AGENT_VISIBLE_TRANSCRIPT,
    )
    after_delivery = validator.validate(write_call)

    assert before_delivery.decision == "block"
    assert before_delivery.reason == "missing_confirmation"
    assert after_delivery.decision == "allow"


def test_model_tool_argument_text_cannot_satisfy_user_confirmation():
    environment = _environment()
    validator = PreWriteValidator(
        domain_name="mock",
        domain_policy="Public policy.",
        tools=environment.get_tools(),
    )
    write_call = ToolCall(
        id="call_write",
        name="update_account",
        arguments={"account_id": "acct_123", "plan_name": "premium"},
    )
    validator.record_tool_result(
        tool_call=ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tool_result=ToolMessage(
            id="call_read",
            role="tool",
            content=json.dumps({"account_id": "acct_123", "status": "active"}),
            error=False,
        ),
        tick_index=1,
    )
    validator.record_assistant_utterance(
        content=(
            "I will update account acct_123 to premium. "
            "This will change the account plan status."
        ),
        tick_index=2,
    )
    validator.record_user_confirmation_evidence(
        content="Yes, I confirm.",
        tick_index=3,
        source=EvidenceSource.MODEL_TOOL_ARGUMENT,
    )

    decision = validator.validate(write_call)

    assert decision.decision == "block"
    assert decision.reason == "missing_confirmation"


def test_forbidden_oracle_evidence_sources_are_rejected_at_runtime():
    validator = PreWriteValidator(
        domain_name="mock",
        domain_policy="Public policy.",
        tools=_environment().get_tools(),
    )

    with pytest.raises(ValueError, match="simulator_gold_text"):
        validator.record_user_confirmation_evidence(
            content="Yes, I confirm.",
            tick_index=1,
            source=EvidenceSource.SIMULATOR_GOLD_TEXT,
        )
    with pytest.raises(ValueError, match="posthoc_oracle"):
        validator.record_user_confirmation_evidence(
            content="Yes, I confirm.",
            tick_index=1,
            source=EvidenceSource.POSTHOC_ORACLE,
        )


def test_confirmation_requires_exact_identifier_mention():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tick_id=1,
    )
    controller.record_assistant_utterance(
        AssistantMessage.text(
            "I will update account acct_1234 to premium. "
            "This will change the account plan status."
        ),
        tick_id=2,
    )
    controller.record_agent_visible_user_transcript("Yes, I confirm.", tick_id=3)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=4,
    )

    packet = json.loads(result.content)
    assert result.error is True
    assert packet["missing_facts"] == ["missing_confirmation"]
    assert environment.tools.write_count == 0


def test_action_statement_requires_consequence():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tick_id=1,
    )
    controller.record_assistant_utterance(
        AssistantMessage.text("I will update account acct_123 to premium."),
        tick_id=2,
    )
    controller.record_agent_visible_user_transcript("Yes, I confirm.", tick_id=3)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=4,
    )

    packet = json.loads(result.content)
    assert result.error is True
    assert packet["missing_facts"] == ["missing_action_summary"]
    assert environment.tools.write_count == 0


def test_read_only_tools_not_overblocked():
    environment = _environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_read",
            name="get_account",
            arguments={"account_id": "acct_123"},
        ),
        tick_id=1,
    )

    assert result.error is False
    assert json.loads(result.content)["status"] == "active"
    assert environment.tools.write_count == 0
    assert orchestrator.num_errors == 0


def test_stagegate_does_not_read_task_objective():
    assert "task_objective" not in signature(PreWriteValidator.validate).parameters


def test_stagegate_does_not_read_expected_final_db():
    assert "expected_final_db" not in signature(PreWriteValidator.validate).parameters


def test_stagegate_does_not_read_user_simulator_private_state():
    assert "user_simulator_private_state" not in (
        signature(PreWriteValidator.validate).parameters
    )


def test_stagegate_does_not_read_evaluator_result():
    assert "evaluator_result" not in signature(PreWriteValidator.validate).parameters


def test_stagegate_does_not_route_by_task_id():
    environment = _environment()
    tool_call = ToolCall(
        id="call_write",
        name="update_account",
        arguments={"account_id": "acct_123", "plan_name": "premium"},
    )

    decisions = []
    for benchmark_task_id in ("task_a", "task_b"):
        controller = StageGateController(
            condition="stagegate",
            domain_policy=environment.get_policy(),
            tools=environment.get_tools(),
            domain_name=environment.get_domain_name(),
        )
        controller.set_trace_context(
            benchmark_task_id=benchmark_task_id,
            sim_id=f"sim_{benchmark_task_id}",
        )
        decisions.append(controller.validate_tool_call(tool_call).model_dump())

    assert decisions[0]["decision"] == decisions[1]["decision"]
    assert decisions[0]["reason"] == decisions[1]["reason"]
    assert decisions[0]["checks"] == decisions[1]["checks"]


def test_stagegate_control_code_does_not_use_task_id_identifier_name():
    repo_root = Path(__file__).resolve().parents[2]
    runtime_root = repo_root / "src/tau2/voice/audio_native/openai/stagegate"
    control_files = [
        runtime_root / "validator.py",
        runtime_root / "ledger.py",
        runtime_root / "orchestrator.py",
    ]
    forbidden = re.compile(r"\btask_id\b")

    matches = []
    for path in control_files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if forbidden.search(line):
                matches.append(f"{path.name}:{line_number}:{line.strip()}")

    assert matches == []


def test_stagegate_runtime_does_not_read_oracle_outcome_fields():
    repo_root = Path(__file__).resolve().parents[2]
    runtime_root = repo_root / "src/tau2/voice/audio_native/openai/stagegate"
    forbidden = {
        "build_outcome_rows",
        "reward_info",
        "reward_breakdown",
        "reward",
        "evaluator_result",
        "stagegate_posthoc_outcomes",
        "trace_final_outcome",
        "write_jsonl",
    }

    matches = []
    for path in runtime_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                matches.append(f"{path.name}:{token}")

    runner_text = (repo_root / "src/tau2/runner/simulation.py").read_text(
        encoding="utf-8"
    )
    assert matches == []
    assert "trace_final_outcome" not in runner_text
    assert "_trace_final_outcome" not in runner_text


def test_policy_preconditions_require_all_required_visible_fields():
    validator = PreWriteValidator(
        domain_name="telecom",
        domain_policy="Public policy.",
        tools=[
            Tool(get_customer_by_id),
            Tool(get_details_by_id),
            Tool(send_payment_request),
        ],
    )
    validator.record_tool_result(
        tool_call=ToolCall(
            id="call_customer",
            name="get_customer_by_id",
            arguments={"customer_id": "C1"},
        ),
        tool_result=ToolMessage(
            id="call_customer",
            role="tool",
            content=json.dumps(
                {"customer_id": "C1", "full_name": "Ada", "bill_ids": ["B1"]}
            ),
            error=False,
        ),
        tick_index=1,
    )
    validator.record_assistant_utterance(
        content=(
            "I will send payment request for customer C1 and bill B1. "
            "This will change the bill status to awaiting payment."
        ),
        tick_index=2,
    )
    validator.record_user_confirmation_evidence(
        content="Yes, I confirm.",
        tick_index=3,
        source=EvidenceSource.AGENT_VISIBLE_TRANSCRIPT,
    )
    write_call = ToolCall(
        id="call_payment",
        name="send_payment_request",
        arguments={"customer_id": "C1", "bill_id": "B1"},
    )

    missing_bill_state = validator.validate(write_call)

    assert missing_bill_state.decision == "block"
    assert missing_bill_state.reason == "missing_policy_precondition_state"

    validator.record_tool_result(
        tool_call=ToolCall(
            id="call_bill",
            name="get_details_by_id",
            arguments={"id": "B1"},
        ),
        tool_result=ToolMessage(
            id="call_bill",
            role="tool",
            content=json.dumps(
                {
                    "bill_id": "B1",
                    "customer_id": "C1",
                    "status": "Overdue",
                    "total_due": 42.5,
                }
            ),
            error=False,
        ),
        tick_index=4,
    )

    allowed = validator.validate(write_call)

    assert allowed.decision == "allow"
    assert allowed.reason == "validated"


def test_trace_event_uses_canonical_schema():
    event = TraceEvent(
        event_type="model_function_call",
        condition="stage_only",
        run_id="run_123",
        domain="mock",
        benchmark_task_id="task_123",
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
        benchmark_task_id="task_1",
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
        benchmark_task_id="task_2",
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
        "validator_check",
        "validator_allow",
        "domain_tool_call",
        "domain_tool_result",
    ]
    assert all(event["benchmark_task_id"] == "task_2" for event in events)
    assert all(event["sim_id"] == "sim_2" for event in events)
    assert events[0]["tool_args"] == {"account_id": "acct_123"}
    assert events[2]["validator_reason"] == "read_only_tool"
    assert events[4]["latency_ms"] >= 0
    assert events[4]["payload"]["tool_error"] is False


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
