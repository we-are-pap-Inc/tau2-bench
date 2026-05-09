"""Pre-write validation for StageGate domain tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.stagegate.ledger import (
    EntityLedger,
    LedgerStatus,
    normalize_value,
    parse_tool_result,
    values_by_key,
)
from tau2.voice.audio_native.openai.stagegate.stage_schema import StagePacket

ValidatorOutcome = Literal["allow", "block"]

READ_TOOL_PREFIXES = (
    "calculate",
    "can_",
    "check_",
    "find_",
    "get_",
    "list_",
    "run_",
    "search_",
)

SIDE_EFFECTING_PREFIXES = (
    "book_",
    "cancel_",
    "connect_",
    "create_",
    "delete_",
    "disable_",
    "dismiss_",
    "enable_",
    "exchange_",
    "grant_",
    "make_",
    "modify_",
    "reboot_",
    "refuel_",
    "remove_",
    "reset_",
    "resume_",
    "return_",
    "send_",
    "set_",
    "suspend_",
    "toggle_",
    "turn_",
    "update_",
)

SIDE_EFFECTING_TOOLS_BY_DOMAIN = {
    "mock": {
        "create_task",
        "dismiss_notification",
        "update_account",
        "update_task_status",
    },
    "retail": {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    },
    "airline": {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    },
    "telecom": {
        "disable_roaming",
        "enable_roaming",
        "refuel_data",
        "resume_line",
        "send_payment_request",
        "suspend_line",
    },
}

NON_SIDE_EFFECTING_TOOL_NAMES = {
    "advance_stage",
    "calculate",
    "transfer_to_human_agents",
}

IDENTITY_READ_TOOLS_BY_DOMAIN = {
    "mock": {"get_account", "get_users"},
    "retail": {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_user_details",
    },
    "airline": {
        "get_reservation_details",
        "get_user_details",
    },
    "telecom": {
        "get_customer_by_id",
        "get_customer_by_name",
        "get_customer_by_phone",
        "get_details_by_id",
    },
}

INSPECTION_TOOLS_BY_EXACT_ARG = {
    "account_id": {"get_account", "get_customer_by_id", "get_details_by_id"},
    "bill_id": {"get_bills_for_customer", "get_details_by_id"},
    "customer_id": {"get_customer_by_id", "get_details_by_id"},
    "line_id": {"get_data_usage", "get_details_by_id"},
    "order_id": {"get_order_details"},
    "reservation_id": {"get_reservation_details"},
    "task_id": {"get_tasks"},
    "user_id": {"get_user_details"},
}

ACTION_WORDS = {
    "book",
    "cancel",
    "change",
    "create",
    "disable",
    "enable",
    "exchange",
    "modify",
    "refund",
    "resume",
    "return",
    "send",
    "suspend",
    "update",
}

CONSEQUENCE_WORDS = {
    "charge",
    "cost",
    "fee",
    "paid",
    "payment",
    "refund",
    "request",
    "status",
    "will",
}

CONFIRMATION_PATTERNS = (
    re.compile(r"\b(confirm|confirmed|yes|yep|yeah|correct|proceed)\b", re.I),
    re.compile(r"\b(go ahead|sounds good|that's right|that is right)\b", re.I),
)

NEGATIVE_CONFIRMATION_PATTERN = re.compile(r"\b(no|don't|do not|stop|wait)\b", re.I)


class ValidatorDecision(BaseModel):
    """A StageGate validator decision for one model tool call."""

    decision: ValidatorOutcome
    reason: str
    checks: dict[str, bool] = Field(default_factory=dict)
    corrective_packet: Optional[StagePacket] = None

    @property
    def allowed(self) -> bool:
        """Whether the original tool call may execute."""
        return self.decision == "allow"


@dataclass(frozen=True)
class ActionRequirement:
    """Visible requirements that must be true before a side-effecting tool."""

    exact_args: tuple[str, ...] = ()
    identity_tools: tuple[str, ...] = ()
    inspection_tools: tuple[str, ...] = ()
    precondition_fields: tuple[str, ...] = ()
    requires_action_statement: bool = True
    requires_user_confirmation: bool = True


@dataclass
class InspectionRecord:
    """A successful visible read-tool result."""

    tool_name: str
    args: dict[str, Any]
    payload: Any
    tick_index: Optional[int]
    keys: set[str] = field(default_factory=set)
    normalized_values: set[str] = field(default_factory=set)


@dataclass
class VisibleConversationState:
    """Validator state derived only from agent-visible conversation events."""

    read_inspections: list[InspectionRecord] = field(default_factory=list)
    verified_identifiers: dict[str, set[str]] = field(default_factory=dict)
    last_action_statement: Optional[str] = None
    last_action_statement_tick: Optional[int] = None
    last_user_confirmation: Optional[str] = None
    last_user_confirmation_tick: Optional[int] = None


class PreWriteValidator:
    """Blocks unsafe side-effecting tool calls before domain state changes."""

    def __init__(
        self,
        *,
        domain_name: Optional[str],
        tools: list[Tool],
        domain_policy: str,
    ) -> None:
        self.domain_name = normalize_domain(domain_name)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.domain_policy = domain_policy
        self.state = VisibleConversationState()

    def set_domain_name(self, domain_name: Optional[str]) -> None:
        """Update the public domain name used for domain-specific rules."""
        self.domain_name = normalize_domain(domain_name)

    def validate(
        self,
        tool_call: ToolCall,
        *,
        ledger: Optional[EntityLedger] = None,
    ) -> ValidatorDecision:
        """Validate a domain tool call without calling the tool."""
        if not self.is_side_effecting_tool(tool_call.name):
            return ValidatorDecision(
                decision="allow",
                reason="read_only_tool",
                checks={"side_effecting_tool": False},
            )

        requirement = self.requirement_for_tool(tool_call.name)
        checks: dict[str, bool] = {
            "side_effecting_tool": True,
            "tool_arguments_complete": self._arguments_complete(tool_call),
            "tool_arguments_non_ambiguous": self._arguments_non_ambiguous(tool_call),
            "identity_verified": self._identity_verified(requirement),
            "exact_identifiers_verified": self._exact_identifiers_verified(
                tool_call,
                requirement,
                ledger=ledger,
            ),
            "policy_state_inspected": self._policy_state_inspected(
                tool_call,
                requirement,
            ),
            "policy_preconditions_represented": self._policy_preconditions_represented(
                tool_call,
                requirement,
            ),
            "assistant_stated_action": self._assistant_stated_action(requirement),
            "user_confirmed": self._user_confirmed(tool_call, requirement),
        }
        reason_by_check = {
            "tool_arguments_complete": "incomplete_tool_arguments",
            "tool_arguments_non_ambiguous": "ambiguous_tool_arguments",
            "identity_verified": "missing_verified_identity",
            "exact_identifiers_verified": "missing_verified_identifier",
            "policy_state_inspected": "missing_policy_state_inspection",
            "policy_preconditions_represented": "missing_policy_precondition_state",
            "assistant_stated_action": "missing_action_summary",
            "user_confirmed": "missing_confirmation",
        }
        for check_name, reason in reason_by_check.items():
            if not checks[check_name]:
                return self._block(tool_call, reason=reason, checks=checks)
        return ValidatorDecision(
            decision="allow",
            reason="validated",
            checks=checks,
        )

    def record_visible_message(
        self,
        *,
        role: Literal["assistant", "user"],
        content: Optional[str],
        tick_index: Optional[int] = None,
    ) -> None:
        """Record agent-visible text from a participant chunk."""
        if not content:
            return
        if role == "assistant":
            if looks_like_action_statement(content):
                self.state.last_action_statement = content
                self.state.last_action_statement_tick = tick_index
        elif looks_like_user_confirmation(content):
            self.state.last_user_confirmation = content
            self.state.last_user_confirmation_tick = tick_index

    def record_tool_result(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolMessage,
        tick_index: Optional[int] = None,
    ) -> None:
        """Record successful read-tool output for later validation."""
        if tool_result.error or self.is_side_effecting_tool(tool_call.name):
            return

        payload = parse_tool_result(tool_result.content)
        record = InspectionRecord(
            tool_name=tool_call.name,
            args=dict(tool_call.arguments),
            payload=payload,
            tick_index=tick_index,
            keys=flatten_keys(payload),
            normalized_values=payload_values(payload),
        )
        for value in iter_argument_values(tool_call.arguments):
            normalized = normalize_value(value)
            if normalized is not None:
                record.normalized_values.add(normalized)
        self.state.read_inspections.append(record)
        self._record_verified_arguments(tool_call.arguments)
        self._record_verified_payload(payload)

    def requirement_for_tool(self, tool_name: str) -> ActionRequirement:
        """Return policy-facing requirements for a side-effecting tool."""
        exact_args = exact_identifier_args_for_tool(tool_name)
        inspection_tools = inspection_tools_for_args(exact_args)
        identity_tools = tuple(
            sorted(IDENTITY_READ_TOOLS_BY_DOMAIN.get(self.domain_name, ()))
        )
        precondition_fields = precondition_fields_for_tool(tool_name)
        if self.domain_name == "mock" and tool_name == "update_account":
            return ActionRequirement(
                exact_args=("account_id",),
                identity_tools=("get_account",),
                inspection_tools=("get_account",),
                precondition_fields=("status",),
            )
        return ActionRequirement(
            exact_args=exact_args,
            identity_tools=identity_tools,
            inspection_tools=inspection_tools,
            precondition_fields=precondition_fields,
        )

    def is_side_effecting_tool(self, tool_name: str) -> bool:
        """Classify side-effecting tools from reviewed names and public schema text."""
        if tool_name in NON_SIDE_EFFECTING_TOOL_NAMES:
            return False
        if tool_name in SIDE_EFFECTING_TOOLS_BY_DOMAIN.get(self.domain_name, set()):
            return True
        if tool_name.startswith(READ_TOOL_PREFIXES):
            return False
        if tool_name.startswith(SIDE_EFFECTING_PREFIXES):
            return True

        tool = self.tools_by_name.get(tool_name)
        description = ""
        if tool is not None:
            description = f"{tool.short_desc}\n{tool.long_desc}".lower()
        side_effect_cues = (
            "update",
            "modify",
            "create",
            "delete",
            "cancel",
            "refund",
            "send",
            "suspend",
            "resume",
            "enable",
            "disable",
        )
        return any(cue in description for cue in side_effect_cues)

    def _block(
        self,
        tool_call: ToolCall,
        *,
        reason: str,
        checks: dict[str, bool],
    ) -> ValidatorDecision:
        return ValidatorDecision(
            decision="block",
            reason=reason,
            checks=checks,
            corrective_packet=build_corrective_packet(
                tool_call=tool_call,
                reason=reason,
                read_tools=self._read_tool_names(),
            ),
        )

    def _arguments_complete(self, tool_call: ToolCall) -> bool:
        tool = self.tools_by_name.get(tool_call.name)
        if tool is None:
            return bool(tool_call.arguments)
        required = set(tool.params.model_json_schema().get("required", []))
        for arg_name in required:
            if arg_name not in tool_call.arguments:
                return False
            if is_empty_value(tool_call.arguments[arg_name]):
                return False
        return True

    def _arguments_non_ambiguous(self, tool_call: ToolCall) -> bool:
        return not any(
            is_ambiguous_value(value) for value in tool_call.arguments.values()
        )

    def _identity_verified(self, requirement: ActionRequirement) -> bool:
        if not requirement.identity_tools:
            return True
        return any(
            inspection.tool_name in requirement.identity_tools
            for inspection in self.state.read_inspections
        )

    def _exact_identifiers_verified(
        self,
        tool_call: ToolCall,
        requirement: ActionRequirement,
        *,
        ledger: Optional[EntityLedger],
    ) -> bool:
        for arg_name in requirement.exact_args:
            if arg_name not in tool_call.arguments:
                return False
            values = list(iter_values(tool_call.arguments[arg_name]))
            if not values:
                return False
            for value in values:
                if not self._is_value_verified(arg_name, value, ledger=ledger):
                    return False
        return True

    def _policy_state_inspected(
        self,
        tool_call: ToolCall,
        requirement: ActionRequirement,
    ) -> bool:
        if not requirement.inspection_tools:
            return True
        return any(
            self._inspection_matches_tool_call(inspection, tool_call, requirement)
            for inspection in self.state.read_inspections
        )

    def _policy_preconditions_represented(
        self,
        tool_call: ToolCall,
        requirement: ActionRequirement,
    ) -> bool:
        if not requirement.precondition_fields:
            return True
        return any(
            self._inspection_matches_tool_call(inspection, tool_call, requirement)
            and any(
                field in inspection.keys for field in requirement.precondition_fields
            )
            for inspection in self.state.read_inspections
        )

    def _assistant_stated_action(self, requirement: ActionRequirement) -> bool:
        if not requirement.requires_action_statement:
            return True
        return self.state.last_action_statement is not None

    def _user_confirmed(
        self,
        tool_call: ToolCall,
        requirement: ActionRequirement,
    ) -> bool:
        if not requirement.requires_user_confirmation:
            return True
        if (
            self.state.last_action_statement_tick is None
            or self.state.last_user_confirmation_tick is None
        ):
            return False
        if (
            self.state.last_user_confirmation_tick
            < self.state.last_action_statement_tick
        ):
            return False
        return self._action_statement_mentions_exact_args(tool_call, requirement)

    def _is_value_verified(
        self,
        arg_name: str,
        value: Any,
        *,
        ledger: Optional[EntityLedger],
    ) -> bool:
        normalized = normalize_value(value)
        if normalized is None:
            return False
        if normalized in self.state.verified_identifiers.get(arg_name, set()):
            return True
        if self._ledger_verifies_value(arg_name, normalized, ledger):
            return True
        return self._last_confirmed_action_mentions(normalized)

    def _ledger_verifies_value(
        self,
        arg_name: str,
        normalized: str,
        ledger: Optional[EntityLedger],
    ) -> bool:
        if ledger is None:
            return False
        slot_names = ledger_slots_for_arg(arg_name)
        for slot_name in slot_names:
            slot = ledger.slots.get(slot_name)
            if slot is None:
                continue
            if slot.status not in {
                LedgerStatus.TOOL_VERIFIED,
                LedgerStatus.USER_CONFIRMED,
            }:
                continue
            slot_values = list(iter_values(slot.value))
            if any(normalize_value(value) == normalized for value in slot_values):
                return True
        return False

    def _last_confirmed_action_mentions(self, normalized_value: str) -> bool:
        if self.state.last_action_statement is None:
            return False
        statement = normalize_value(self.state.last_action_statement)
        if statement is None:
            return False
        return normalized_value in statement

    def _action_statement_mentions_exact_args(
        self,
        tool_call: ToolCall,
        requirement: ActionRequirement,
    ) -> bool:
        for arg_name in requirement.exact_args:
            if arg_name not in tool_call.arguments:
                return False
            for value in iter_values(tool_call.arguments[arg_name]):
                normalized = normalize_value(value)
                if normalized is not None and not self._last_confirmed_action_mentions(
                    normalized
                ):
                    return False
        return True

    def _inspection_matches_tool_call(
        self,
        inspection: InspectionRecord,
        tool_call: ToolCall,
        requirement: ActionRequirement,
    ) -> bool:
        if inspection.tool_name not in requirement.inspection_tools:
            return False
        exact_values = {
            normalize_value(value)
            for arg_name in requirement.exact_args
            if arg_name in tool_call.arguments
            for value in iter_values(tool_call.arguments[arg_name])
        }
        exact_values.discard(None)
        if not exact_values:
            return True
        return bool(exact_values & inspection.normalized_values)

    def _record_verified_arguments(self, arguments: dict[str, Any]) -> None:
        for arg_name, value in arguments.items():
            for item in iter_values(value):
                self._record_verified_identifier(arg_name, item)

    def _record_verified_payload(self, payload: Any) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (dict, list)):
                    self._record_verified_payload(value)
                else:
                    self._record_verified_identifier(key, value)
        elif isinstance(payload, list):
            for item in payload:
                self._record_verified_payload(item)

    def _record_verified_identifier(self, name: str, value: Any) -> None:
        normalized = normalize_value(value)
        if normalized is None:
            return
        self.state.verified_identifiers.setdefault(name, set()).add(normalized)
        for alias in equivalent_identifier_names(name):
            self.state.verified_identifiers.setdefault(alias, set()).add(normalized)

    def _read_tool_names(self) -> list[str]:
        return sorted(
            name for name in self.tools_by_name if not self.is_side_effecting_tool(name)
        )


def normalize_domain(domain_name: Optional[str]) -> str:
    """Normalize a public domain name."""
    return (domain_name or "").strip().lower()


def exact_identifier_args_for_tool(tool_name: str) -> tuple[str, ...]:
    """Return top-level tool args that identify exact mutable records."""
    arg_names: list[str] = []
    for candidate in (
        "account_id",
        "bill_id",
        "customer_id",
        "line_id",
        "order_id",
        "payment_method_id",
        "reservation_id",
        "task_id",
        "user_id",
    ):
        if candidate in tool_name:
            arg_names.append(candidate)
    if tool_name.startswith("book_reservation"):
        return ("user_id",)
    if tool_name.startswith("send_certificate"):
        return ("user_id",)
    if tool_name.startswith("modify_user_address"):
        return ("user_id",)
    if tool_name in {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
    }:
        return ("order_id",)
    if tool_name in {
        "cancel_reservation",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }:
        return ("reservation_id",)
    if tool_name in {
        "disable_roaming",
        "enable_roaming",
        "refuel_data",
        "resume_line",
        "suspend_line",
    }:
        return ("customer_id", "line_id")
    if tool_name == "send_payment_request":
        return ("customer_id", "bill_id")
    if tool_name == "update_account":
        return ("account_id",)
    if tool_name == "update_task_status":
        return ("task_id",)
    return tuple(arg_names)


def inspection_tools_for_args(exact_args: tuple[str, ...]) -> tuple[str, ...]:
    """Return read tools that can inspect state for exact mutable identifiers."""
    tools: set[str] = set()
    for arg_name in exact_args:
        tools.update(INSPECTION_TOOLS_BY_EXACT_ARG.get(arg_name, set()))
    return tuple(sorted(tools))


def precondition_fields_for_tool(tool_name: str) -> tuple[str, ...]:
    """Return policy-state fields that must be visible before calling a tool."""
    if tool_name in {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
    }:
        return ("status",)
    if tool_name in {
        "cancel_reservation",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }:
        return ("flights", "passengers", "cabin", "payment_history")
    if tool_name in {
        "disable_roaming",
        "enable_roaming",
        "refuel_data",
        "resume_line",
        "suspend_line",
    }:
        return ("status", "plan_id", "data_usage")
    if tool_name == "send_payment_request":
        return ("status", "total_amount_due")
    if tool_name == "update_account":
        return ("status",)
    return ()


def ledger_slots_for_arg(arg_name: str) -> tuple[str, ...]:
    """Map tool argument names to ledger slots when available."""
    return {
        "account_id": ("account_id",),
        "bill_id": ("bill_id",),
        "customer_id": ("account_id",),
        "line_id": ("phone_line",),
        "order_id": ("order_id",),
        "payment_method_id": ("payment_method", "payment_or_fee_acknowledgment"),
        "reservation_id": ("reservation_id",),
        "user_id": ("customer_name", "passenger_name"),
    }.get(arg_name, (arg_name,))


def equivalent_identifier_names(name: str) -> set[str]:
    """Return equivalent identifier keys seen across args and tool payloads."""
    equivalents = {name}
    if name == "id":
        equivalents.update({"account_id", "bill_id", "customer_id", "line_id"})
    if name == "customer_id":
        equivalents.add("account_id")
    if name == "account_id":
        equivalents.add("customer_id")
    return equivalents


def build_corrective_packet(
    *,
    tool_call: ToolCall,
    reason: str,
    read_tools: list[str],
) -> StagePacket:
    """Build a corrective StageGate packet for a blocked tool call."""
    return StagePacket(
        stage=stage_for_reason(reason),
        objective="Recover the missing prerequisite before executing a write/action tool.",
        known_facts={},
        missing_facts=[reason],
        ambiguous_facts=[],
        ask_next=corrective_instruction(tool_call=tool_call, reason=reason),
        allowed_read_tools=read_tools,
        allowed_write_tools=[],
        do_not=[
            f"Do not call {tool_call.name} again until the missing prerequisite is satisfied."
        ],
        exit_condition="The missing prerequisite is visible in conversation, ledger, or official read-tool state.",
        when_done="Call advance_stage or retry the original tool only after the corrective step is complete.",
    )


def stage_for_reason(reason: str) -> str:
    """Map a block reason to the most relevant StageGate stage."""
    if reason in {
        "missing_verified_identity",
        "missing_verified_identifier",
        "ambiguous_tool_arguments",
        "incomplete_tool_arguments",
    }:
        return "collect_required_exact_entities"
    if reason in {
        "missing_policy_state_inspection",
        "missing_policy_precondition_state",
    }:
        return "inspect_state_with_read_tools"
    return "propose_action_and_confirm"


def corrective_instruction(*, tool_call: ToolCall, reason: str) -> str:
    """Return concise corrective text for the model."""
    if reason == "missing_confirmation":
        return (
            f"Before making the change, summarize the intended {tool_call.name} "
            "action and consequence, then ask the user for explicit confirmation."
        )
    if reason == "missing_action_summary":
        return (
            "Tell the user exactly what will change and any consequence before "
            "asking for confirmation."
        )
    if reason == "missing_policy_state_inspection":
        return (
            "Use the narrowest read tool to inspect the current official state first."
        )
    if reason == "missing_policy_precondition_state":
        return "Inspect the policy-relevant state needed to establish the precondition first."
    if reason == "missing_verified_identifier":
        return "Verify the exact identifier with a read tool or explicit user confirmation."
    if reason == "missing_verified_identity":
        return "Authenticate or verify the account identity with an official read tool first."
    if reason == "ambiguous_tool_arguments":
        return "Ask one clarification question to resolve the ambiguous tool argument."
    return "Collect the missing required tool argument before retrying."


def looks_like_action_statement(content: str) -> bool:
    """Detect a visible assistant statement of intended action and consequence."""
    normalized = normalize_value(content)
    if normalized is None:
        return False
    has_action = any(word in normalized for word in ACTION_WORDS)
    has_consequence = any(word in normalized for word in CONSEQUENCE_WORDS)
    return has_action and has_consequence


def looks_like_user_confirmation(content: str) -> bool:
    """Detect an affirmative user confirmation from visible transcript text."""
    if NEGATIVE_CONFIRMATION_PATTERN.search(content):
        return False
    return any(pattern.search(content) for pattern in CONFIRMATION_PATTERNS)


def is_empty_value(value: Any) -> bool:
    """Return whether a tool argument value is missing or empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def is_ambiguous_value(value: Any) -> bool:
    """Return whether a tool argument carries an obvious ambiguous placeholder."""
    if isinstance(value, str):
        normalized = normalize_value(value) or ""
        if normalized in {"unknown", "not sure", "n/a", "na", "maybe"}:
            return True
        return " or " in normalized
    if isinstance(value, list):
        return any(is_ambiguous_value(item) for item in value)
    if isinstance(value, dict):
        return any(is_ambiguous_value(item) for item in value.values())
    return False


def iter_values(value: Any) -> list[Any]:
    """Flatten list values but keep dictionaries as atomic values."""
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(iter_values(item))
        return flattened
    return [value]


def iter_argument_values(arguments: dict[str, Any]) -> list[Any]:
    """Return flattened values from top-level tool arguments."""
    values: list[Any] = []
    for value in arguments.values():
        values.extend(iter_values(value))
    return values


def flatten_keys(payload: Any) -> set[str]:
    """Collect lowercase keys from a visible structured payload."""
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(key.lower())
            keys.update(flatten_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.update(flatten_keys(item))
    return keys


def payload_values(payload: Any) -> set[str]:
    """Collect normalized primitive values from a visible structured payload."""
    values: set[str] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            values.update(payload_values(value))
    elif isinstance(payload, list):
        for item in payload:
            values.update(payload_values(item))
    else:
        normalized = normalize_value(payload)
        if normalized is not None:
            values.add(normalized)
    return values


def tool_result_values_for_key(payload: Any, key: str) -> list[Any]:
    """Return values for a key from a visible tool-result payload."""
    return values_by_key(payload, {key})


def canonical_json(data: Any) -> str:
    """Return stable JSON for tests and trace payloads."""
    return json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
