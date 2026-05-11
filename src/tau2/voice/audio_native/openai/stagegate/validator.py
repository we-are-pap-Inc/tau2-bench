"""Pre-write validation for StageGate domain tool calls."""

from __future__ import annotations

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
)
from tau2.voice.audio_native.openai.stagegate.stage_schema import (
    EvidenceSource,
    StagePacket,
    ensure_runtime_evidence_source,
)

ValidatorOutcome = Literal["allow", "block"]
SERVICE_TASK_REF = "service_task_ref"
SERVICE_TASK_READ_TOOL = "get_tasks"
SERVICE_TASK_WRITE_TOOL = "update_task_status"

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
    SERVICE_TASK_REF: {SERVICE_TASK_READ_TOOL},
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
    "exchanging",
    "modify",
    "refund",
    "resume",
    "return",
    "send",
    "submit",
    "suspend",
    "swap",
    "swapped",
    "update",
}

CONSEQUENCE_WORDS = {
    "card",
    "charge",
    "cost",
    "credit",
    "difference",
    "email",
    "fee",
    "instruction",
    "instructions",
    "paid",
    "payment",
    "price",
    "refund",
    "request",
    "status",
}

CONFIRMATION_PATTERNS = (
    re.compile(r"\b(confirm|confirmed|yes|yep|yeah|correct|proceed)\b", re.I),
    re.compile(
        r"\b(go ahead|please do|sounds good|that's right|that is right|okay,? do it)\b",
        re.I,
    ),
)

NEGATIVE_CONFIRMATION_PATTERN = re.compile(r"\b(no|don't|do not|stop|wait)\b", re.I)
ASSISTANT_UTTERANCE_BUFFER_CHARS = 6000
MAX_EVIDENCE_RECORDS = 30


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
class ActionSummaryEvidence:
    """Visible assistant evidence for one pending side-effecting action."""

    content: str
    tick_index: Optional[int]
    source: EvidenceSource


@dataclass
class UserConfirmationEvidence:
    """Visible user evidence for confirming a pending action summary."""

    content: str
    tick_index: Optional[int]
    source: EvidenceSource


@dataclass
class VisibleConversationState:
    """Validator state derived only from agent-visible conversation events."""

    read_inspections: list[InspectionRecord] = field(default_factory=list)
    verified_identifiers: dict[str, set[str]] = field(default_factory=dict)
    assistant_utterance_buffer: str = ""
    action_summaries: list[ActionSummaryEvidence] = field(default_factory=list)
    user_confirmations: list[UserConfirmationEvidence] = field(default_factory=list)
    last_action_statement: Optional[str] = None
    last_action_statement_tick: Optional[int] = None
    last_action_statement_source: Optional[EvidenceSource] = None
    last_user_confirmation: Optional[str] = None
    last_user_confirmation_tick: Optional[int] = None
    last_user_confirmation_source: Optional[EvidenceSource] = None


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
        matching_summary = self._matching_action_summary(tool_call, requirement)
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
            "assistant_stated_action": self._assistant_stated_action(
                requirement,
                matching_summary,
            ),
            "user_confirmed": self._user_confirmed(
                tool_call,
                requirement,
                matching_summary=matching_summary,
            ),
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
        """Record legacy assistant text; user text requires explicit evidence."""
        if role == "assistant":
            self.record_assistant_utterance(content=content, tick_index=tick_index)
            return
        raise ValueError(
            "UserMessage.content from audio-native chunks is simulator gold text; "
            "use record_user_confirmation_evidence with AGENT_VISIBLE_TRANSCRIPT."
        )

    def record_assistant_utterance(
        self,
        *,
        content: Optional[str],
        tick_index: Optional[int] = None,
        source: EvidenceSource | str = EvidenceSource.ASSISTANT_UTTERANCE,
    ) -> None:
        """Record assistant-visible action summaries from model output."""
        if not content:
            return
        source = ensure_runtime_evidence_source(source)
        if source is not EvidenceSource.ASSISTANT_UTTERANCE:
            raise ValueError(
                "assistant action evidence must use ASSISTANT_UTTERANCE source"
            )
        self.state.assistant_utterance_buffer = trim_text(
            f"{self.state.assistant_utterance_buffer}{content}",
            max_chars=ASSISTANT_UTTERANCE_BUFFER_CHARS,
        )
        if looks_like_action_statement(self.state.assistant_utterance_buffer):
            self._record_action_summary(
                content=self.state.assistant_utterance_buffer,
                tick_index=tick_index,
                source=source,
            )

    def record_user_confirmation_evidence(
        self,
        *,
        content: Optional[str],
        tick_index: Optional[int] = None,
        source: EvidenceSource | str,
    ) -> None:
        """Record user confirmation only from model-path transcript evidence."""
        if not content:
            return
        source = ensure_runtime_evidence_source(source)
        if source is not EvidenceSource.AGENT_VISIBLE_TRANSCRIPT:
            return
        if looks_like_user_confirmation(content):
            evidence = UserConfirmationEvidence(
                content=content,
                tick_index=tick_index,
                source=source,
            )
            self.state.user_confirmations.append(evidence)
            self.state.user_confirmations = self.state.user_confirmations[
                -MAX_EVIDENCE_RECORDS:
            ]
            self.state.last_user_confirmation = content
            self.state.last_user_confirmation_tick = tick_index
            self.state.last_user_confirmation_source = source
        self.state.assistant_utterance_buffer = ""

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
        self._record_verified_payload(payload, tool_name=tool_call.name)

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
            values = list(self._tool_argument_values(tool_call, arg_name))
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
            and all(
                precondition_field_present(inspection.keys, field)
                for field in requirement.precondition_fields
            )
            for inspection in self.state.read_inspections
        )

    def _record_action_summary(
        self,
        *,
        content: str,
        tick_index: Optional[int],
        source: EvidenceSource,
    ) -> None:
        summary = ActionSummaryEvidence(
            content=content,
            tick_index=tick_index,
            source=source,
        )
        self.state.action_summaries.append(summary)
        self.state.action_summaries = self.state.action_summaries[
            -MAX_EVIDENCE_RECORDS:
        ]
        self.state.last_action_statement = content
        self.state.last_action_statement_tick = tick_index
        self.state.last_action_statement_source = source

    def _assistant_stated_action(
        self,
        requirement: ActionRequirement,
        matching_summary: Optional[ActionSummaryEvidence],
    ) -> bool:
        if not requirement.requires_action_statement:
            return True
        return matching_summary is not None

    def _user_confirmed(
        self,
        tool_call: ToolCall,
        requirement: ActionRequirement,
        *,
        matching_summary: Optional[ActionSummaryEvidence],
    ) -> bool:
        if not requirement.requires_user_confirmation:
            return True
        if matching_summary is None or matching_summary.tick_index is None:
            return False
        return any(
            confirmation.source is EvidenceSource.AGENT_VISIBLE_TRANSCRIPT
            and confirmation.tick_index is not None
            and confirmation.tick_index > matching_summary.tick_index
            for confirmation in self.state.user_confirmations
        )

    def _matching_action_summary(
        self,
        tool_call: ToolCall,
        requirement: ActionRequirement,
    ) -> Optional[ActionSummaryEvidence]:
        if not requirement.requires_action_statement:
            return None
        for summary in reversed(self.state.action_summaries):
            if self._action_summary_matches_tool_call(summary, tool_call, requirement):
                return summary
        return None

    def _action_summary_matches_tool_call(
        self,
        summary: ActionSummaryEvidence,
        tool_call: ToolCall,
        requirement: ActionRequirement,
    ) -> bool:
        if summary.source is not EvidenceSource.ASSISTANT_UTTERANCE:
            return False
        if not looks_like_action_statement(summary.content):
            return False
        if tool_call.name == "exchange_delivered_order_items":
            return self._exchange_summary_matches_tool_call(summary.content, tool_call)
        return self._action_statement_mentions_exact_args(
            summary.content,
            tool_call,
            requirement,
        )

    def _exchange_summary_matches_tool_call(
        self,
        content: str,
        tool_call: ToolCall,
    ) -> bool:
        normalized = normalize_value(content) or ""
        if not re.search(
            r"\b(exchange|exchanging|exchanged|swap|swapped)\b", normalized
        ):
            return False
        if not summary_requests_confirmation(normalized):
            return False
        if not self._summary_mentions_order_reference(normalized, tool_call):
            return False
        return self._summary_mentions_exchange_items(normalized, tool_call)

    def _summary_mentions_order_reference(
        self,
        normalized_summary: str,
        tool_call: ToolCall,
    ) -> bool:
        order_id = normalize_value(tool_call.arguments.get("order_id"))
        if order_id and normalized_text_mentions_value(normalized_summary, order_id):
            return True
        return "order" in normalized_summary

    def _summary_mentions_exchange_items(
        self,
        normalized_summary: str,
        tool_call: ToolCall,
    ) -> bool:
        old_item_ids = {
            normalize_value(value)
            for value in iter_values(tool_call.arguments.get("item_ids"))
        }
        new_item_ids = {
            normalize_value(value)
            for value in iter_values(tool_call.arguments.get("new_item_ids"))
        }
        old_item_ids.discard(None)
        new_item_ids.discard(None)
        mentioned_old = {
            value
            for value in old_item_ids
            if normalized_text_mentions_value(normalized_summary, value)
        }
        mentioned_new = {
            value
            for value in new_item_ids
            if normalized_text_mentions_value(normalized_summary, value)
        }
        if mentioned_old and mentioned_new:
            return True

        target_values = old_item_ids | new_item_ids
        descriptor_tokens = self._descriptor_tokens_for_values(target_values)
        summary_tokens = meaningful_tokens(normalized_summary)
        return len(summary_tokens & descriptor_tokens) >= 3

    def _descriptor_tokens_for_values(self, normalized_values: set[str]) -> set[str]:
        if not normalized_values:
            return set()
        tokens: set[str] = set()
        for inspection in self.state.read_inspections:
            for text in descriptor_strings_for_values(
                inspection.payload,
                normalized_values,
            ):
                tokens.update(meaningful_tokens(text))
        return tokens

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
        return normalized_text_mentions_value(statement, normalized_value)

    def _action_statement_mentions_exact_args(
        self,
        statement: str,
        tool_call: ToolCall,
        requirement: ActionRequirement,
    ) -> bool:
        normalized_statement = normalize_value(statement)
        if normalized_statement is None:
            return False
        for arg_name in requirement.exact_args:
            values = list(self._tool_argument_values(tool_call, arg_name))
            if not values:
                return False
            for value in values:
                normalized = normalize_value(value)
                if normalized is not None and not normalized_text_mentions_value(
                    normalized_statement,
                    normalized,
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
            for value in self._tool_argument_values(tool_call, arg_name)
        }
        exact_values.discard(None)
        if not exact_values:
            return True
        return bool(exact_values & inspection.normalized_values)

    def _record_verified_arguments(self, arguments: dict[str, Any]) -> None:
        for arg_name, value in arguments.items():
            identifier_name = canonical_identifier_name(
                tool_name=None,
                raw_name=arg_name,
            )
            for item in iter_values(value):
                self._record_verified_identifier(identifier_name, item)

    def _record_verified_payload(self, payload: Any, *, tool_name: str) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                identifier_name = canonical_identifier_name(
                    tool_name=tool_name,
                    raw_name=key,
                )
                if isinstance(value, (dict, list)):
                    for item in iter_values(value):
                        self._record_verified_identifier(identifier_name, item)
                    self._record_verified_payload(value, tool_name=tool_name)
                else:
                    self._record_verified_identifier(identifier_name, value)
        elif isinstance(payload, list):
            for item in payload:
                self._record_verified_payload(item, tool_name=tool_name)

    def _tool_argument_values(
        self,
        tool_call: ToolCall,
        identifier_name: str,
    ) -> list[Any]:
        return [
            value
            for arg_name in argument_names_for_identifier(
                identifier_name,
                tool_call=tool_call,
                tools_by_name=self.tools_by_name,
            )
            if arg_name in tool_call.arguments
            for value in iter_values(tool_call.arguments[arg_name])
        ]

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
    if tool_name == SERVICE_TASK_WRITE_TOOL:
        return (SERVICE_TASK_REF,)
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
    }:
        return ("roaming_enabled",)
    if tool_name in {
        "resume_line",
        "suspend_line",
    }:
        return ("status", "contract_end_date")
    if tool_name == "refuel_data":
        return ("data_used_gb", "data_limit_gb")
    if tool_name == "send_payment_request":
        return ("status", "total_due")
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


def argument_names_for_identifier(
    identifier_name: str,
    *,
    tool_call: ToolCall,
    tools_by_name: dict[str, Tool],
) -> tuple[str, ...]:
    """Map internal identifier names to public domain-tool argument names."""
    if identifier_name != SERVICE_TASK_REF:
        return (identifier_name,)

    candidates = tuple(
        arg_name for arg_name in tool_call.arguments if is_identifier_arg_name(arg_name)
    )
    if candidates:
        return candidates

    tool = tools_by_name.get(tool_call.name)
    if tool is None:
        return (SERVICE_TASK_REF,)

    required = tool.params.model_json_schema().get("required", [])
    schema_candidates = tuple(
        arg_name
        for arg_name in required
        if isinstance(arg_name, str) and is_identifier_arg_name(arg_name)
    )
    return schema_candidates or (SERVICE_TASK_REF,)


def canonical_identifier_name(
    *,
    tool_name: Optional[str],
    raw_name: str,
) -> str:
    """Return the StageGate control name for a public domain identifier field."""
    if tool_name == SERVICE_TASK_READ_TOOL and is_identifier_arg_name(raw_name):
        return SERVICE_TASK_REF
    return raw_name


def is_identifier_arg_name(name: str) -> bool:
    """Return whether a public argument name has identifier shape."""
    return name.endswith("_id")


def equivalent_identifier_names(name: str) -> set[str]:
    """Return equivalent identifier keys seen across args and tool payloads."""
    equivalents = {name}
    if name == "id":
        equivalents.update(
            {"account_id", "bill_id", "customer_id", "line_id", "plan_id"}
        )
    if name.endswith("_ids"):
        equivalents.add(name.removesuffix("s"))
    if name == "orders":
        equivalents.add("order_id")
    if name == "reservations":
        equivalents.add("reservation_id")
    if name == "customer_id":
        equivalents.add("account_id")
    if name == "account_id":
        equivalents.add("customer_id")
    return equivalents


def precondition_field_present(keys: set[str], field: str) -> bool:
    """Return whether any accepted alias for a precondition field is visible."""
    aliases = {
        "data_usage": {"data_usage", "data_used_gb"},
        "total_amount_due": {"total_amount_due", "total_due"},
    }.get(field, {field})
    return bool(keys & aliases)


def normalized_text_mentions_value(text: str, value: str) -> bool:
    """Return whether normalized text contains value as a distinct token."""
    if not value:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])",
            text,
            flags=re.I,
        )
        is not None
    )


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
            f"Do not call {tool_call.name} again until the missing prerequisite is satisfied.",
            "Do not call advance_stage before retrying the original write tool.",
        ],
        exit_condition="The missing prerequisite is visible in conversation, ledger, or official read-tool state.",
        when_done=f"After the user confirms, retry {tool_call.name} directly.",
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
            "action and consequence, then ask the user for explicit confirmation. "
            f"After the user confirms, retry {tool_call.name} directly."
        )
    if reason == "missing_action_summary":
        return (
            "Restate exactly what will change and any consequence, ask for explicit "
            f"confirmation, then retry {tool_call.name} directly after the user "
            "confirms."
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


def trim_text(text: str, *, max_chars: int) -> str:
    """Keep only the recent transcript text needed for validator evidence."""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


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


def summary_requests_confirmation(normalized_content: str) -> bool:
    """Return whether an assistant summary asked the user to confirm/proceed."""
    return any(
        cue in normalized_content
        for cue in (
            "confirm",
            "please reply yes",
            "please say yes",
            "say yes",
            "reply yes",
            "proceed",
        )
    )


def meaningful_tokens(text: str) -> set[str]:
    """Return compact content tokens for matching summaries to read-tool payloads."""
    stopwords = {
        "and",
        "any",
        "are",
        "for",
        "from",
        "item",
        "items",
        "order",
        "the",
        "this",
        "that",
        "will",
        "with",
        "your",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords and not token.isdigit()
    }


def descriptor_strings_for_values(
    payload: Any,
    normalized_values: set[str],
) -> list[str]:
    """Collect string descriptors from payload objects containing target IDs."""
    descriptors: list[str] = []
    if isinstance(payload, dict):
        if payload_contains_normalized_value(payload, normalized_values):
            descriptors.extend(string_leaf_values(payload))
        for value in payload.values():
            descriptors.extend(descriptor_strings_for_values(value, normalized_values))
    elif isinstance(payload, list):
        for item in payload:
            descriptors.extend(descriptor_strings_for_values(item, normalized_values))
    return descriptors


def payload_contains_normalized_value(
    payload: dict[str, Any], values: set[str]
) -> bool:
    """Return whether a dictionary contains one of the target normalized values."""
    for value in payload.values():
        if isinstance(value, (dict, list)):
            continue
        normalized = normalize_value(value)
        if normalized in values:
            return True
    return False


def string_leaf_values(payload: Any) -> list[str]:
    """Return all non-ID string leaves from a payload subtree."""
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and not key.lower().endswith("_id"):
                values.append(value)
            elif isinstance(value, (dict, list)):
                values.extend(string_leaf_values(value))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(string_leaf_values(item))
    return values


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
