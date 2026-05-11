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

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer the customer to a human agent.

        Args:
            summary: A summary for the human agent.

        Returns:
            Transfer status.
        """
        return "Transfer successful"


class RetailExchangeToolkit(ToolKitBase):
    def __init__(self):
        self.write_count = 0

    @is_tool(ToolType.READ)
    def find_user_id_by_name_zip(
        self,
        first_name: str,
        last_name: str,
        zip: str,
    ) -> str:
        """Find a retail user by name and ZIP.

        Args:
            first_name: User first name.
            last_name: User last name.
            zip: User ZIP code.

        Returns:
            User ID.
        """
        return "yusuf_rossi_9620"

    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> dict:
        """Get retail user details.

        Args:
            user_id: User ID.

        Returns:
            User profile details.
        """
        return {
            "user_id": user_id,
            "name": {"first_name": "Yusuf", "last_name": "Rossi"},
            "email": "yusuf.rossi7301@example.com",
            "address": {"zip": "19122"},
            "payment_methods": [
                {
                    "payment_method_id": "credit_card_9513926",
                    "source": "credit card ending 2478",
                }
            ],
        }

    @is_tool(ToolType.READ)
    def get_order_details(self, order_id: str) -> dict:
        """Get retail order details.

        Args:
            order_id: Order ID.

        Returns:
            Order details.
        """
        return {
            "order_id": order_id,
            "user_id": "yusuf_rossi_9620",
            "status": "delivered",
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "1656367028",
                    "item_id": "1151293680",
                    "options": {
                        "switch type": "linear",
                        "backlight": "RGB",
                        "size": "full size",
                    },
                },
                {
                    "name": "Smart Thermostat",
                    "product_id": "4896585277",
                    "item_id": "4983901480",
                    "options": {
                        "compatibility": "Apple HomeKit",
                        "color": "black",
                    },
                },
            ],
            "payment_history": [
                {
                    "transaction_type": "payment",
                    "payment_method_id": "credit_card_9513926",
                }
            ],
        }

    @is_tool(ToolType.READ)
    def get_product_details(self, product_id: str) -> dict:
        """Get retail product details.

        Args:
            product_id: Product ID.

        Returns:
            Product details.
        """
        if product_id == "1656367028":
            return {
                "name": "Mechanical Keyboard",
                "product_id": product_id,
                "variants": {
                    "7706410293": {
                        "item_id": "7706410293",
                        "options": {
                            "switch type": "clicky",
                            "backlight": "none",
                            "size": "full size",
                        },
                        "available": True,
                    },
                    "9025753381": {
                        "item_id": "9025753381",
                        "options": {
                            "switch type": "clicky",
                            "backlight": "RGB",
                            "size": "full size",
                        },
                        "available": False,
                    },
                },
            }
        return {
            "name": "Smart Thermostat",
            "product_id": product_id,
            "variants": {
                "7747408585": {
                    "item_id": "7747408585",
                    "options": {
                        "compatibility": "Google Assistant",
                        "color": "black",
                    },
                    "available": True,
                },
                "4983901480": {
                    "item_id": "4983901480",
                    "options": {
                        "compatibility": "Apple HomeKit",
                        "color": "black",
                    },
                    "available": True,
                },
            },
        }

    @is_tool(ToolType.WRITE)
    def exchange_delivered_order_items(
        self,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str],
        payment_method_id: str,
    ) -> dict:
        """Exchange delivered order items.

        Args:
            order_id: Order ID.
            item_ids: Existing delivered item IDs.
            new_item_ids: Replacement item IDs.
            payment_method_id: Payment method for price difference.

        Returns:
            Exchange request result.
        """
        self.write_count += 1
        return {
            "order_id": order_id,
            "status": "exchange requested",
            "exchange_items": item_ids,
            "exchange_new_items": new_item_ids,
            "exchange_payment_method_id": payment_method_id,
        }

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer the customer to a human agent.

        Args:
            summary: A summary for the human agent.

        Returns:
            Transfer status.
        """
        return "Transfer successful"


def _environment(domain_name: str = "mock") -> Environment:
    return Environment(
        domain_name=domain_name,
        policy="Public policy.",
        tools=StageGateToolkit(),
    )


def _retail_exchange_environment() -> Environment:
    return Environment(
        domain_name="retail",
        policy="Retail public policy.",
        tools=RetailExchangeToolkit(),
    )


def _controller(environment: Environment) -> StageGateController:
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
    )
    controller.set_domain_name(environment.get_domain_name())
    return controller


def _advance_stage_packet(
    controller: StageGateController,
    *,
    current_stage: str = "understand_intent",
    observed_facts: list[str] | None = None,
    last_action: str = "started",
    blocker: str | None = None,
    tick_id: int = 1,
) -> dict:
    arguments = {
        "current_stage": current_stage,
        "observed_facts": observed_facts or [],
        "last_action": last_action,
    }
    if blocker is not None:
        arguments["blocker"] = blocker
    result = controller.handle_advance_stage(
        ToolCall(
            id=f"call_stage_{tick_id}",
            name="advance_stage",
            arguments=arguments,
        ),
        tick_id=tick_id,
    )
    assert result.error is False
    return json.loads(result.content)


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
            "This will change the account plan status. Please confirm."
        ),
        tick_id=0,
    )


def _exchange_tool_call(call_id: str = "call_exchange") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="exchange_delivered_order_items",
        arguments={
            "order_id": "#W2378156",
            "item_ids": ["1151293680", "4983901480"],
            "new_item_ids": ["7706410293", "7747408585"],
            "payment_method_id": "credit_card_9513926",
        },
    )


def _pending_write_id(controller: StageGateController) -> str:
    snapshot = controller.validator.pending_write_snapshot()
    assert snapshot is not None
    return str(snapshot["pending_write_id"])


def _step_by_name(packet: dict, step_name: str) -> dict:
    for step in packet["next_required_steps"]:
        if step["step"] == step_name:
            return step
    raise AssertionError(f"missing next_required_steps entry {step_name!r}")


def _assert_summary_step(packet: dict, tool_name: str) -> None:
    summary_step = _step_by_name(packet, "call_tool")
    assert summary_step["tool_name"] == "record_pending_write_summary"
    assert summary_step["arguments"] == {
        "summary_presented": True,
        "action_type": tool_name,
        "consequence_presented": True,
        "confirmation_requested": True,
    }
    assert "pending_write_id" not in summary_step["arguments"]


def _assert_confirmation_step(packet: dict) -> None:
    confirmation_step = _step_by_name(packet, "call_tool_if_user_confirms")
    assert confirmation_step["tool_name"] == "record_pending_write_confirmation"
    assert confirmation_step["arguments"] == {
        "decision": "confirmed",
        "basis": "latest_user_turn",
    }
    assert "pending_write_id" not in confirmation_step["arguments"]


def _assert_retry_step(packet: dict, tool_name: str) -> None:
    retry_step = _step_by_name(packet, "retry_original_write")
    assert retry_step["tool_name"] == tool_name


def _record_pending_summary(
    orchestrator: FullDuplexOrchestrator,
    controller: StageGateController,
    *,
    pending_write_id: str | None = None,
    tick_id: int = 11,
    action_type: str = "exchange_delivered_order_items",
) -> ToolMessage:
    return orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id=f"call_record_summary_{tick_id}",
            name="record_pending_write_summary",
            arguments={
                "summary_presented": True,
                "action_type": action_type,
                "consequence_presented": True,
                "confirmation_requested": True,
            }
            | (
                {"pending_write_id": pending_write_id}
                if pending_write_id is not None
                else {}
            ),
        ),
        tick_id=tick_id,
    )


def _record_pending_confirmation(
    orchestrator: FullDuplexOrchestrator,
    controller: StageGateController,
    *,
    pending_write_id: str | None = None,
    decision: str = "confirmed",
    basis: str = "latest_user_turn",
    user_tick_id: int = 12,
    tool_tick_id: int = 13,
) -> ToolMessage:
    controller.record_agent_visible_user_transcript(
        "user response event",
        tick_id=user_tick_id,
    )
    return orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id=f"call_record_confirmation_{tool_tick_id}",
            name="record_pending_write_confirmation",
            arguments={
                "decision": decision,
                "basis": basis,
            }
            | (
                {"pending_write_id": pending_write_id}
                if pending_write_id is not None
                else {}
            ),
        ),
        tick_id=tool_tick_id,
    )


def _prepare_validated_retail_exchange(
    orchestrator: FullDuplexOrchestrator,
    controller: StageGateController,
) -> None:
    for tick_id, tool_call in enumerate(
        [
            ToolCall(
                id="call_find_user",
                name="find_user_id_by_name_zip",
                arguments={
                    "first_name": "Yusuf",
                    "last_name": "Rossi",
                    "zip": "19122",
                },
            ),
            ToolCall(
                id="call_user",
                name="get_user_details",
                arguments={"user_id": "yusuf_rossi_9620"},
            ),
            ToolCall(
                id="call_order",
                name="get_order_details",
                arguments={"order_id": "#W2378156"},
            ),
            ToolCall(
                id="call_keyboard",
                name="get_product_details",
                arguments={"product_id": "1656367028"},
            ),
            ToolCall(
                id="call_thermostat",
                name="get_product_details",
                arguments={"product_id": "4896585277"},
            ),
        ],
        start=1,
    ):
        result = orchestrator._execute_stagegate_tool_call(
            controller,
            tool_call,
            tick_id=tick_id,
        )
        assert result.error is False


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
    system_prompt = adapter.connect.call_args.kwargs["system_prompt"]
    assert tool_names == ["_test_tool", "advance_stage"]
    assert "StageGate operating rules" in system_prompt
    assert (
        "call advance_stage before the first customer-specific domain tool call"
        in system_prompt
    )
    assert "include only facts visible in the conversation" in system_prompt


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
    assert not hasattr(agent.stagegate_controller, "ledger")
    assert not hasattr(agent.stagegate_controller, "validator")


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

    tools = adapter.connect.call_args.kwargs["tools"]
    tool_names = [tool.name for tool in tools]
    system_prompt = adapter.connect.call_args.kwargs["system_prompt"]
    assert tool_names == [
        "_test_tool",
        "advance_stage",
        "record_pending_write_summary",
        "record_pending_write_confirmation",
    ]
    tool_schema_by_name = {tool.name: tool.params.model_json_schema() for tool in tools}
    assert "pending_write_id" not in tool_schema_by_name[
        "record_pending_write_summary"
    ].get("required", [])
    assert "pending_write_id" not in tool_schema_by_name[
        "record_pending_write_confirmation"
    ].get("required", [])
    assert "StageGate operating rules" in system_prompt
    assert (
        "call advance_stage before the first customer-specific domain tool call"
        in system_prompt
    )
    assert "record_pending_write_summary" in system_prompt
    assert "no pending_write_id is needed" in system_prompt
    assert hasattr(agent.stagegate_controller, "ledger")
    assert hasattr(agent.stagegate_controller, "validator")


def test_pending_write_tool_descriptions_explain_active_handle():
    controller = StageGateController(
        condition="stagegate",
        domain_policy="Policy.",
        tools=[Tool(_test_tool)],
        domain_name="mock",
    )

    summary_description = controller.pending_write_summary_tool.openai_schema[
        "function"
    ]["description"]
    confirmation_description = controller.pending_write_confirmation_tool.openai_schema[
        "function"
    ]["description"]

    normalized_descriptions = [
        " ".join(description.split())
        for description in (summary_description, confirmation_description)
    ]
    for description in normalized_descriptions:
        assert "active pending write" in description
        assert "pending_write_id is not required" in description
        assert "retry the same original write tool directly" in description
    assert (
        "after you have told the user the pending action and consequence"
        in normalized_descriptions[0]
    )
    assert (
        "after the user responds to that confirmation request"
        in normalized_descriptions[1]
    )


def test_entity_ledger_initializes_domain_slots_and_serializes():
    ledger = EntityLedger.for_domain("retail")

    assert list(ledger.slots) == [
        "customer_name",
        "email",
        "phone",
        "order_id",
        "item_id",
        "order_item_ids",
        "candidate_replacement_item_ids",
        "selected_old_item_ids",
        "selected_new_item_ids",
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
    assert retail_ledger.slots["item_id"].status is LedgerStatus.MISSING
    assert retail_ledger.slots["candidate_replacement_item_ids"].value == ["item_1"]

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
        == "Ask one concise clarification question for account_id: cust_123, cust_999."
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


def test_max_advance_stage_calls_triggers_guard(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))
    environment = _environment()
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
        max_advance_stage_calls_per_sim=1,
        max_repeated_same_stage=10,
        max_repeated_same_blocker=10,
    )

    first_packet = _advance_stage_packet(controller, tick_id=1)
    guarded_packet = _advance_stage_packet(controller, tick_id=2)

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert first_packet["schema_version"] == "stagegate.stage_packet.v1"
    assert guarded_packet["schema_version"] == "stagegate.stage_packet.v1"
    assert guarded_packet["allowed_write_tools"] == []
    assert guarded_packet["ask_next"].startswith("Ask the customer for")
    assert controller.loop_guard_triggered is True
    assert any(event["event_type"] == "stage_loop_guard_triggered" for event in events)


def test_repeated_same_stage_triggers_guard():
    environment = _environment()
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
        max_advance_stage_calls_per_sim=10,
        max_repeated_same_stage=1,
        max_repeated_same_blocker=10,
    )

    _advance_stage_packet(controller, current_stage="understand_intent", tick_id=1)
    packet = _advance_stage_packet(
        controller,
        current_stage="understand_intent",
        tick_id=2,
    )

    assert controller.loop_guard_triggered is True
    assert controller.repeated_stage_count == 2
    assert packet["do_not"][0] == "Do not call advance_stage again immediately."


def test_repeated_blocker_triggers_guard():
    environment = _environment()
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
        max_advance_stage_calls_per_sim=10,
        max_repeated_same_stage=10,
        max_repeated_same_blocker=1,
    )

    _advance_stage_packet(
        controller,
        current_stage="propose_action_and_confirm",
        blocker="missing_confirmation",
        tick_id=1,
    )
    packet = _advance_stage_packet(
        controller,
        current_stage="propose_action_and_confirm",
        blocker="missing_confirmation",
        tick_id=2,
    )

    assert controller.loop_guard_triggered is True
    assert controller.repeated_blocker_count == 2
    assert packet["missing_facts"][0] == "missing_confirmation"


def test_guard_fallback_does_not_execute_domain_tool():
    environment = _environment()
    environment.get_response = MagicMock(side_effect=AssertionError("unexpected call"))
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
        max_advance_stage_calls_per_sim=1,
        max_repeated_same_stage=10,
        max_repeated_same_blocker=10,
    )
    orchestrator = _orchestrator_shell(environment)

    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_stage_1",
            name="advance_stage",
            arguments={
                "current_stage": "understand_intent",
                "observed_facts": [],
                "last_action": "started",
            },
        ),
        tick_id=1,
    )
    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_stage_2",
            name="advance_stage",
            arguments={
                "current_stage": "understand_intent",
                "observed_facts": [],
                "last_action": "still unclear",
            },
        ),
        tick_id=2,
    )

    assert result.error is False
    assert json.loads(result.content)["allowed_write_tools"] == []
    assert environment.tools.write_count == 0
    environment.get_response.assert_not_called()


def test_stage_packets_include_only_stage_scoped_missing_facts():
    environment = _environment(domain_name="retail")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    packet = _advance_stage_packet(
        controller,
        current_stage="understand_intent",
        tick_id=1,
    )

    missing = " ".join(packet["missing_facts"]).lower()
    assert "payment" not in missing
    assert "address" not in missing
    assert len(packet["missing_facts"]) < len(controller.ledger.slots)


def test_stage_only_missing_fact_matching_ignores_short_substrings():
    environment = _environment(domain_name="retail")
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    packet = _advance_stage_packet(
        controller,
        current_stage="understand_intent",
        observed_facts=["The order is late."],
    )

    assert "customer email, phone, or name and ZIP" in packet["missing_facts"]
    assert "order ID if known" in packet["missing_facts"]


def test_stage_only_missing_fact_matching_treats_slash_as_alternative():
    environment = _environment(domain_name="retail")
    controller = StageGateController(
        condition="stage_only",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    packet = _advance_stage_packet(
        controller,
        current_stage="inspect_state_with_read_tools",
        observed_facts=["Official order status and refund facts were verified."],
    )

    assert (
        "policy-relevant order status and payment/refund facts"
        not in packet["missing_facts"]
    )


def test_retail_identity_packet_excludes_later_stage_slots():
    environment = _environment(domain_name="retail")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    packet = _advance_stage_packet(controller, current_stage="understand_intent")
    text = " ".join([*packet["missing_facts"], *packet["do_not"]]).lower()

    assert "payment method" not in text
    assert "address" not in text


def test_airline_identity_packet_excludes_fee_and_change_slots():
    environment = _environment(domain_name="airline")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    packet = _advance_stage_packet(controller, current_stage="understand_intent")
    text = " ".join(packet["missing_facts"]).lower()

    assert "fee" not in text
    assert "change" not in text


def test_telecom_identity_packet_excludes_plan_and_device_slots():
    environment = _environment(domain_name="telecom")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    packet = _advance_stage_packet(controller, current_stage="understand_intent")
    text = " ".join(packet["missing_facts"]).lower()

    assert "plan" not in text
    assert "device" not in text


def test_stagegate_packet_uses_ledger_enrichment():
    environment = _environment(domain_name="retail")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    controller.trace_model_function_call(
        ToolCall(
            id="call_email",
            name="find_user_id_by_email",
            arguments={"email": "ada@example.com"},
        ),
        tick_id=1,
    )

    packet = _advance_stage_packet(controller, current_stage="understand_intent")

    assert packet["known_facts"]["email"]["value"] == "ada@example.com"
    assert "customer email" not in packet["missing_facts"]


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


def test_pending_write_creation_blocks_first_write(monkeypatch, tmp_path):
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
    assert packet["missing_facts"] == ["missing_action_summary"]
    assert "record_pending_write_summary" in packet["ask_next"]
    assert "record_pending_write_confirmation" in packet["ask_next"]
    assert "pending_write_id" not in packet["ask_next"]
    assert packet["allowed_internal_tools"] == ["record_pending_write_summary"]
    assert packet["disallowed_tools"] == [
        "advance_stage",
        "transfer_to_human_agents",
    ]
    _assert_summary_step(packet, "update_account")
    _assert_confirmation_step(packet)
    _assert_retry_step(packet, "update_account")
    assert "retry update_account directly" in packet["when_done"]
    assert environment.tools.write_count == 0
    assert "pending_write_created" in [event["event_type"] for event in events]
    assert events[-1]["event_type"] == "validator_block"
    assert events[-1]["validator_reason"] == "missing_action_summary"


def test_summary_tool_records_summary_without_transcript_parsing():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    blocked = orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    assert blocked.error is True

    summary_result = _record_pending_summary(orchestrator, controller, tick_id=11)

    assert summary_result.error is False
    assert json.loads(summary_result.content)["reason"] == "summary_recorded"
    assert controller.validator.pending_write_snapshot()["status"] == "summarized"


def test_confirmation_tool_records_confirmed_after_user_turn():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)

    confirmation_result = _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=12,
        tool_tick_id=13,
    )

    assert confirmation_result.error is False
    assert json.loads(confirmation_result.content)["reason"] == "confirmation_confirmed"
    assert controller.validator.pending_write_snapshot()["status"] == "confirmed"


def test_pending_write_tools_accept_optional_debug_id():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    pending_id = _pending_write_id(controller)
    summary_result = _record_pending_summary(
        orchestrator,
        controller,
        pending_write_id=pending_id,
        tick_id=11,
    )
    confirmation_result = _record_pending_confirmation(
        orchestrator,
        controller,
        pending_write_id=pending_id,
        user_tick_id=12,
        tool_tick_id=13,
    )

    assert summary_result.error is False
    assert confirmation_result.error is False
    assert controller.validator.pending_write_snapshot()["status"] == "confirmed"


def test_retry_same_write_allowed_after_structured_confirmation():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=12,
        tool_tick_id=13,
    )

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_retry_exchange"),
        tick_id=14,
    )

    assert result.error is False
    assert json.loads(result.content)["status"] == "exchange requested"
    assert environment.tools.write_count == 1
    assert controller.validator.pending_write_snapshot()["status"] == "consumed"


def test_confirmation_tool_before_summary_does_not_confirm():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    result = _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=11,
        tool_tick_id=12,
    )

    assert result.error is True
    assert json.loads(result.content)["reason"] == "pending_write_not_summarized"
    assert controller.validator.pending_write_snapshot()["status"] == "needs_summary"


def test_denied_or_unclear_pending_write_does_not_allow_write():
    for decision, reason in (
        ("denied", "pending_write_denied"),
        ("unclear", "pending_write_unclear"),
    ):
        environment = _retail_exchange_environment()
        controller = StageGateController(
            condition="stagegate",
            domain_policy=environment.get_policy(),
            tools=environment.get_tools(),
            domain_name=environment.get_domain_name(),
        )
        orchestrator = _orchestrator_shell(environment)
        _prepare_validated_retail_exchange(orchestrator, controller)

        orchestrator._execute_stagegate_tool_call(
            controller,
            _exchange_tool_call("call_initial_exchange"),
            tick_id=10,
        )
        _record_pending_summary(orchestrator, controller, tick_id=11)
        _record_pending_confirmation(
            orchestrator,
            controller,
            decision=decision,
            basis="user_declined" if decision == "denied" else "unclear_response",
            user_tick_id=12,
            tool_tick_id=13,
        )

        result = orchestrator._execute_stagegate_tool_call(
            controller,
            _exchange_tool_call("call_retry_exchange"),
            tick_id=14,
        )

        packet = json.loads(result.content)
        assert result.error is True
        assert controller.last_validator_decision["reason"] == reason
        assert packet["missing_facts"] == [reason]
        assert environment.tools.write_count == 0


def test_unclear_pending_write_keeps_protocol_active_for_clarification():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        decision="unclear",
        basis="unclear_response",
        user_tick_id=12,
        tool_tick_id=13,
    )

    packet = _advance_stage_packet(
        controller,
        current_stage="verify_result_and_close",
        observed_facts=["Model claims the write can close."],
        last_action="No confirmed pending write yet.",
        tick_id=14,
    )

    assert packet["stage"] == "propose_action_and_confirm"
    assert packet["missing_facts"] == ["pending_write_unclear"]
    assert packet["allowed_internal_tools"] == ["record_pending_write_confirmation"]
    assert packet["disallowed_tools"] == [
        "advance_stage",
        "transfer_to_human_agents",
    ]
    clarification_step = _step_by_name(packet, "ask_one_clarification")
    assert "clarification" in clarification_step["instruction"]

    transfer = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_transfer_unclear_path",
            name="transfer_to_human_agents",
            arguments={"summary": "User response was unclear."},
        ),
        tick_id=15,
    )
    transfer_packet = json.loads(transfer.content)
    assert transfer.error is True
    assert transfer_packet["missing_facts"] == ["transfer_blocked_pending_write"]

    repeated_without_user = _record_pending_confirmation(
        orchestrator,
        controller,
        decision="confirmed",
        basis="latest_user_turn",
        user_tick_id=12,
        tool_tick_id=16,
    )
    assert repeated_without_user.error is True
    assert json.loads(repeated_without_user.content)["reason"] == (
        "missing_user_turn_after_unclear_confirmation"
    )

    clarified = _record_pending_confirmation(
        orchestrator,
        controller,
        decision="confirmed",
        basis="latest_user_turn",
        user_tick_id=17,
        tool_tick_id=18,
    )
    assert clarified.error is False
    assert controller.validator.pending_write_snapshot()["status"] == "confirmed"


def test_denied_pending_write_allows_non_write_resolution_path():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        decision="denied",
        basis="user_declined",
        user_tick_id=12,
        tool_tick_id=13,
    )

    packet = _advance_stage_packet(
        controller,
        current_stage="verify_result_and_close",
        observed_facts=["Model claims the write can close."],
        last_action="User declined the pending write.",
        tick_id=14,
    )
    transfer = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_transfer_denied_path",
            name="transfer_to_human_agents",
            arguments={"summary": "User declined the pending write."},
        ),
        tick_id=15,
    )

    assert packet["stage"] == "propose_action_and_confirm"
    assert packet["missing_facts"] == ["pending_write_denied"]
    assert packet["next_required_steps"][0]["step"] == "do_not_retry_denied_write"
    assert packet["disallowed_tools"] == []
    assert transfer.error is False


def test_missing_exchange_summary_still_blocks_write():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)
    controller.record_agent_visible_user_transcript("user response event", tick_id=10)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call(),
        tick_id=11,
    )

    packet = json.loads(result.content)
    assert result.error is True
    assert packet["missing_facts"] == ["missing_action_summary"]
    assert "record_pending_write_summary" in packet["ask_next"]
    assert "record_pending_write_confirmation" in packet["ask_next"]
    assert "pending_write_id" not in packet["ask_next"]
    assert packet["allowed_internal_tools"] == ["record_pending_write_summary"]
    assert packet["disallowed_tools"] == [
        "advance_stage",
        "transfer_to_human_agents",
    ]
    _assert_summary_step(packet, "exchange_delivered_order_items")
    _assert_confirmation_step(packet)
    _assert_retry_step(packet, "exchange_delivered_order_items")
    assert "retry exchange_delivered_order_items directly" in packet["when_done"]
    assert "advance_stage" not in packet["when_done"]
    assert environment.tools.write_count == 0


def test_transfer_to_human_blocked_during_resolvable_pending_write(
    monkeypatch,
    tmp_path,
):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_transfer",
            name="transfer_to_human_agents",
            arguments={"summary": "Cannot complete the pending write protocol."},
        ),
        tick_id=11,
    )

    packet = json.loads(result.content)
    event_types = [
        json.loads(line)["event_type"] for line in trace_path.read_text().splitlines()
    ]
    assert result.error is True
    assert packet["missing_facts"] == ["transfer_blocked_pending_write"]
    assert "record_pending_write_summary" in packet["ask_next"]
    assert "pending_write_id" not in packet["ask_next"]
    assert packet["allowed_internal_tools"] == ["record_pending_write_summary"]
    assert packet["disallowed_tools"] == [
        "advance_stage",
        "transfer_to_human_agents",
    ]
    _assert_summary_step(packet, "exchange_delivered_order_items")
    _assert_confirmation_step(packet)
    _assert_retry_step(packet, "exchange_delivered_order_items")
    assert "transfer_blocked_pending_write" in event_types
    assert environment.tools.write_count == 0


def test_transfer_to_human_allowed_without_resolvable_pending_write():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    no_pending = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_transfer_no_pending",
            name="transfer_to_human_agents",
            arguments={"summary": "Out of scope."},
        ),
        tick_id=1,
    )
    assert no_pending.error is False
    assert no_pending.content == "Transfer successful"

    _prepare_validated_retail_exchange(orchestrator, controller)
    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        decision="denied",
        basis="user_declined",
        user_tick_id=12,
        tool_tick_id=13,
    )
    denied_pending = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_transfer_denied",
            name="transfer_to_human_agents",
            arguments={"summary": "User declined the pending write."},
        ),
        tick_id=14,
    )

    assert denied_pending.error is False
    assert denied_pending.content == "Transfer successful"


def test_pending_exchange_summary_and_confirmation_allows_retry():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    blocked = orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    assert blocked.error is True
    assert controller.validator.pending_write_snapshot()["status"] == "needs_summary"
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=12,
        tool_tick_id=13,
    )

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_retry_exchange"),
        tick_id=14,
    )

    assert result.error is False
    assert json.loads(result.content)["status"] == "exchange requested"
    assert environment.tools.write_count == 1
    assert controller.validator.pending_write_snapshot()["status"] == "consumed"


def test_pending_exchange_retry_with_changed_args_blocks_as_mismatch(
    monkeypatch,
    tmp_path,
):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=12,
        tool_tick_id=13,
    )

    changed_call = _exchange_tool_call("call_changed_exchange")
    changed_call.arguments["new_item_ids"] = ["9025753381", "7747408585"]
    result = orchestrator._execute_stagegate_tool_call(
        controller,
        changed_call,
        tick_id=14,
    )

    packet = json.loads(result.content)
    assert result.error is True
    assert packet["missing_facts"] == ["pending_write_mismatch"]
    assert controller.last_validator_decision["reason"] == "pending_write_mismatch"
    assert environment.tools.write_count == 0
    event_types = [
        json.loads(line)["event_type"] for line in trace_path.read_text().splitlines()
    ]
    assert "pending_write_mismatch" in event_types
    assert "pending_write_expired" in event_types


def test_advance_stage_needs_summary_repeats_required_protocol_steps():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    packet = _advance_stage_packet(
        controller,
        current_stage="verify_result_and_close",
        observed_facts=["Model claims confirmation happened."],
        last_action="No structured pending write summary yet.",
        tick_id=11,
    )

    assert packet["stage"] == "propose_action_and_confirm"
    assert packet["allowed_internal_tools"] == ["record_pending_write_summary"]
    assert packet["allowed_write_tools"] == []
    assert packet["disallowed_tools"] == [
        "advance_stage",
        "transfer_to_human_agents",
    ]
    _assert_summary_step(packet, "exchange_delivered_order_items")
    _assert_confirmation_step(packet)
    _assert_retry_step(packet, "exchange_delivered_order_items")


def test_advance_stage_summarized_repeats_confirmation_recording_steps():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    packet = _advance_stage_packet(
        controller,
        current_stage="verify_result_and_close",
        observed_facts=["Model claims confirmation happened."],
        last_action="No structured pending write confirmation yet.",
        tick_id=12,
    )

    assert packet["stage"] == "propose_action_and_confirm"
    assert packet["allowed_internal_tools"] == ["record_pending_write_confirmation"]
    assert packet["allowed_write_tools"] == []
    assert packet["disallowed_tools"] == [
        "advance_stage",
        "transfer_to_human_agents",
    ]
    _assert_confirmation_step(packet)
    _assert_retry_step(packet, "exchange_delivered_order_items")


def test_confirmed_unconsumed_pending_write_keeps_stage_at_execute():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=12,
        tool_tick_id=13,
    )

    packet = _advance_stage_packet(
        controller,
        current_stage="verify_result_and_close",
        observed_facts=["Model claims the write was validated."],
        last_action="No domain write result yet.",
        tick_id=14,
    )

    assert packet["stage"] == "execute_write_action"
    assert packet["allowed_write_tools"] == ["exchange_delivered_order_items"]
    assert packet["allowed_internal_tools"] == []
    assert packet["disallowed_tools"] == [
        "advance_stage",
        "transfer_to_human_agents",
    ]
    assert packet["next_required_steps"][0]["step"] == "retry_original_write"
    assert "Retry exchange_delivered_order_items" in packet["ask_next"]


def test_pending_write_trace_events_are_emitted(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=12,
        tool_tick_id=13,
    )
    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_retry_exchange"),
        tick_id=14,
    )

    event_types = [
        json.loads(line)["event_type"] for line in trace_path.read_text().splitlines()
    ]
    assert "pending_write_created" in event_types
    assert "pending_write_summary_recorded" in event_types
    assert "pending_write_confirmed" in event_types
    assert "pending_write_consumed" in event_types


def test_advance_stage_cannot_verify_after_blocked_write():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)

    blocked = orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call(),
        tick_id=10,
    )
    assert blocked.error is True

    packet = _advance_stage_packet(
        controller,
        current_stage="execute_write_action",
        observed_facts=[
            "Model claims action summary and confirmation are satisfied.",
        ],
        last_action="Validated prerequisite action summary and confirmation.",
        tick_id=11,
    )

    assert packet["stage"] in {"propose_action_and_confirm", "execute_write_action"}
    assert packet["stage"] != "verify_result_and_close"

    packet = _advance_stage_packet(
        controller,
        current_stage="verify_result_and_close",
        observed_facts=[
            "Model still claims the blocked write completed.",
        ],
        last_action="Attempting to close without a domain tool result.",
        tick_id=12,
    )

    assert packet["stage"] in {"propose_action_and_confirm", "execute_write_action"}


def test_verify_result_and_close_requires_successful_write_result():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )

    packet = _advance_stage_packet(
        controller,
        current_stage="execute_write_action",
        observed_facts=["User confirmed a write action."],
        last_action="Ready to verify.",
        tick_id=1,
    )

    assert packet["stage"] == "execute_write_action"


def test_retail_item_id_product_list_ambiguity_not_surfaced_at_close():
    environment = _retail_exchange_environment()
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)
    _prepare_validated_retail_exchange(orchestrator, controller)
    orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_initial_exchange"),
        tick_id=10,
    )
    _record_pending_summary(orchestrator, controller, tick_id=11)
    _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=12,
        tool_tick_id=13,
    )
    result = orchestrator._execute_stagegate_tool_call(
        controller,
        _exchange_tool_call("call_retry_exchange"),
        tick_id=14,
    )
    assert result.error is False

    packet = _advance_stage_packet(
        controller,
        current_stage="execute_write_action",
        observed_facts=["Exchange tool succeeded."],
        last_action="exchange_delivered_order_items returned exchange requested",
        tick_id=15,
    )

    assert packet["stage"] == "verify_result_and_close"
    assert not any(
        fact.startswith("item_id:")
        or fact.startswith("candidate_replacement_item_ids:")
        for fact in packet["ambiguous_facts"]
    )


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

    initial = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_initial_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=2,
    )
    assert initial.error is True
    _record_pending_summary(
        orchestrator,
        controller,
        tick_id=3,
        action_type="update_account",
    )
    _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=4,
        tool_tick_id=5,
    )

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_retry_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=6,
    )

    content = json.loads(result.content)
    assert result.error is False
    assert content["plan_name"] == "premium"
    assert environment.tools.write_count == 1
    assert orchestrator.num_errors == 1


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

    initial = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_initial_update_task",
            name="update_task_status",
            arguments={"task_id": "service_task_1", "status": "completed"},
        ),
        tick_id=3,
    )
    assert initial.error is True
    _record_pending_summary(
        orchestrator,
        controller,
        tick_id=4,
        action_type="update_task_status",
    )
    _record_pending_confirmation(
        orchestrator,
        controller,
        user_tick_id=5,
        tool_tick_id=6,
    )

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_retry_update_task",
            name="update_task_status",
            arguments={"task_id": "service_task_1", "status": "completed"},
        ),
        tick_id=7,
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
    assert controller.validator.state.latest_user_turn_tick is None


def test_same_tick_user_turn_cannot_satisfy_pending_confirmation():
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
    assert packet["missing_facts"] == ["missing_action_summary"]
    assert environment.tools.write_count == 0


def test_next_tick_clean_user_content_does_not_confirm_without_internal_tool():
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
        "missing_action_summary"
    ]
    assert environment.tools.write_count == 0


def test_user_turn_event_alone_does_not_satisfy_confirmation():
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
    controller.record_agent_visible_user_transcript("Yes, I confirm.", tick_id=2)

    result = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=3,
    )

    assert result.error is True
    assert json.loads(result.content)["missing_facts"] == ["missing_action_summary"]
    assert environment.tools.write_count == 0


def test_openai_adapter_records_input_transcription_event(monkeypatch):
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
    debug_messages: list[str] = []
    monkeypatch.setattr(
        "tau2.voice.audio_native.openai.discrete_time_adapter.logger.debug",
        debug_messages.append,
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
    assert debug_messages == [
        "Input transcription completed (item_id=item_user, chars=15)"
    ]
    assert "Yes, I confirm." not in "\n".join(debug_messages)


def test_agent_wires_provider_user_transcript_to_user_turn_order(monkeypatch):
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

    assert controller.validator.state.latest_user_turn_tick == 1
    assert decision.decision == "block"
    assert decision.reason == "missing_action_summary"


def test_validator_uses_structured_confirmation_only_after_user_turn():
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
    before_summary = validator.validate(write_call, tick_index=2)
    summary_result = validator.record_pending_write_summary(
        summary_presented=True,
        action_type="update_account",
        consequence_presented=True,
        confirmation_requested=True,
        tick_index=3,
    )
    before_user_turn = validator.record_pending_write_confirmation(
        decision="confirmed",
        basis="latest_user_turn",
        tick_index=4,
    )
    validator.record_user_turn(
        content="Yes, I confirm.",
        tick_index=5,
        source=EvidenceSource.AGENT_VISIBLE_TRANSCRIPT,
    )
    confirmation_result = validator.record_pending_write_confirmation(
        decision="confirmed",
        basis="latest_user_turn",
        tick_index=6,
    )
    after_confirmation = validator.validate(write_call, tick_index=7)

    assert before_summary.decision == "block"
    assert before_summary.reason == "missing_action_summary"
    assert summary_result.ok is True
    assert before_user_turn.ok is False
    assert before_user_turn.reason == "missing_user_turn_after_summary"
    assert confirmation_result.ok is True
    assert after_confirmation.decision == "allow"


def test_model_tool_argument_text_cannot_satisfy_user_turn_order():
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
    first_decision = validator.validate(write_call, tick_index=2)
    assert first_decision.pending_write_id is not None
    validator.record_pending_write_summary(
        summary_presented=True,
        action_type="update_account",
        consequence_presented=True,
        confirmation_requested=True,
        tick_index=3,
    )
    validator.record_user_confirmation_evidence(
        content="Yes, I confirm.",
        tick_index=4,
        source=EvidenceSource.MODEL_TOOL_ARGUMENT,
    )
    confirmation_result = validator.record_pending_write_confirmation(
        decision="confirmed",
        basis="latest_user_turn",
        tick_index=5,
    )

    decision = validator.validate(write_call)

    assert confirmation_result.ok is False
    assert confirmation_result.reason == "missing_user_turn_after_summary"
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


def test_validator_source_has_no_semantic_transcript_regex():
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root / "src/tau2/voice/audio_native/openai/stagegate/validator.py"
    ).read_text(encoding="utf-8")

    forbidden_tokens = [
        "CONFIRMATION_PATTERNS",
        "looks_like_user_confirmation",
        "looks_like_action_statement",
        "summary_requests_confirmation",
        "_exchange_summary_matches_tool_call",
        "_action_summary_matches_tool_call",
        "re.compile",
        "re.search",
        "re.findall",
    ]

    assert all(token not in source for token in forbidden_tokens)


def test_summary_tool_requires_structured_consequence_and_confirmation_request():
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
    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_initial_write",
            name="update_account",
            arguments={"account_id": "acct_123", "plan_name": "premium"},
        ),
        tick_id=2,
    )

    missing_consequence = orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_bad_summary",
            name="record_pending_write_summary",
            arguments={
                "summary_presented": True,
                "action_type": "update_account",
                "consequence_presented": False,
                "confirmation_requested": True,
            },
        ),
        tick_id=3,
    )

    assert missing_consequence.error is True
    assert json.loads(missing_consequence.content)["reason"] == (
        "consequence_not_presented"
    )
    assert controller.validator.pending_write_snapshot()["status"] == "needs_summary"


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

    pending_block = validator.validate(write_call, tick_index=5)
    assert pending_block.pending_write_id is not None
    validator.record_pending_write_summary(
        summary_presented=True,
        action_type="send_payment_request",
        consequence_presented=True,
        confirmation_requested=True,
        tick_index=6,
    )
    validator.record_user_turn(
        content="Yes, I confirm.",
        tick_index=7,
        source=EvidenceSource.AGENT_VISIBLE_TRANSCRIPT,
    )
    validator.record_pending_write_confirmation(
        decision="confirmed",
        basis="latest_user_turn",
        tick_index=8,
    )

    allowed = validator.validate(write_call, tick_index=9)

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


def test_trace_summary_includes_stage_validator_and_ledger_counts(
    monkeypatch,
    tmp_path,
):
    trace_path = tmp_path / "trace_events.jsonl"
    monkeypatch.setenv("TAU2_TRACE_JSONL", str(trace_path))
    environment = _environment(domain_name="telecom")
    controller = StageGateController(
        condition="stagegate",
        domain_policy=environment.get_policy(),
        tools=environment.get_tools(),
        domain_name=environment.get_domain_name(),
    )
    orchestrator = _orchestrator_shell(environment)

    _advance_stage_packet(controller, current_stage="understand_intent", tick_id=1)
    controller.trace_model_function_call(
        ToolCall(
            id="call_customer",
            name="get_customer_by_id",
            arguments={"customer_id": "cust_123"},
        ),
        tick_id=2,
    )
    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_read",
            name="get_tasks",
            arguments={},
        ),
        tick_id=3,
    )
    orchestrator._execute_stagegate_tool_call(
        controller,
        ToolCall(
            id="call_write",
            name="update_account",
            arguments={"account_id": "cust_123", "plan_name": "premium"},
        ),
        tick_id=4,
    )
    controller.trace_run_end(termination_reason="max_steps", duration_seconds=1.5)

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    summary_event = [
        event for event in events if event["event_type"] == "trace_summary"
    ][-1]
    run_end = events[-1]

    assert summary_event["payload"]["advance_stage_call_count"] == 1
    assert summary_event["payload"]["validator_allow_count"] == 1
    assert summary_event["payload"]["validator_block_count"] == 1
    assert summary_event["payload"]["ledger_update_count"] == 1
    assert summary_event["payload"]["stage_sequence"] == ["identify_or_authenticate"]
    assert summary_event["payload"]["final_stage"] == "identify_or_authenticate"
    assert summary_event["payload"]["last_stage_packet"]["schema_version"] == (
        "stagegate.stage_packet.v1"
    )
    assert summary_event["payload"]["last_validator_decision"]["decision"] == "block"
    assert run_end["payload"]["advance_stage_call_count"] == 1
    assert run_end["payload"]["termination_reason"] == "max_steps"


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
    assert [event["event_type"] for event in events] == [
        "run_start",
        "trace_summary",
        "run_end",
    ]
    assert events[0]["trial"] == 3
    assert events[1]["trial"] == 3
    assert events[2]["trial"] == 3
    assert events[2]["payload"]["termination_reason"] == "exception"
    assert events[2]["payload"]["duration_seconds"] >= 0
