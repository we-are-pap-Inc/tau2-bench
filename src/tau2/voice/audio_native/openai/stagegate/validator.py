"""Pre-write validation for StageGate domain tool calls."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
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
PendingWriteStatus = Literal[
    "none",
    "needs_confirmation",
    "denied",
    "unclear",
    "consumed",
    "expired",
    "mismatched_retry",
]
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
    "commit_pending_write",
    "transfer_to_human_agents",
}
ADVANCE_STAGE_TOOL_NAME = "advance_stage"
COMMIT_PENDING_WRITE_TOOL_NAME = "commit_pending_write"
TRANSFER_TOOL_NAME = "transfer_to_human_agents"
TRANSFER_BLOCKING_PENDING_WRITE_STATUSES = {
    "needs_confirmation",
    "unclear",
    "mismatched_retry",
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


class ValidatorDecision(BaseModel):
    """A StageGate validator decision for one model tool call."""

    decision: ValidatorOutcome
    reason: str
    checks: dict[str, bool] = Field(default_factory=dict)
    corrective_packet: Optional[StagePacket] = None
    pending_write_id: Optional[str] = None
    args_fingerprint: Optional[str] = None
    matched_facets: list[str] = Field(default_factory=list)
    missing_facets: list[str] = Field(default_factory=list)

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
class PendingWriteConfirmation:
    """Concrete side-effecting write awaiting structured summary and confirmation."""

    pending_write_id: str
    tool_name: str
    original_args: dict[str, Any]
    normalized_args: dict[str, Any]
    args_fingerprint: str
    created_tick: Optional[int]
    created_stage: Optional[str]
    status: PendingWriteStatus
    confirmation_recorded_tick: Optional[int] = None
    commit_recorded_tick: Optional[int] = None
    last_block_reason: Optional[str] = None
    confirmation_basis: Optional[str] = None
    commit_decision: Optional[str] = None
    commit_basis: Optional[str] = None

    def snapshot(self) -> dict[str, object]:
        """Return trace-safe pending-write state without raw argument values."""
        return {
            "pending_write_id": self.pending_write_id,
            "tool_name": self.tool_name,
            "original_tool_name": self.tool_name,
            "args_fingerprint": self.args_fingerprint,
            "created_tick": self.created_tick,
            "created_stage": self.created_stage,
            "status": self.status,
            "confirmation_recorded_tick": self.confirmation_recorded_tick,
            "commit_recorded_tick": self.commit_recorded_tick,
            "commit_tick": self.commit_recorded_tick,
            "last_block_reason": self.last_block_reason,
            "confirmation_basis": self.confirmation_basis,
            "commit_decision": self.commit_decision,
            "commit_basis": self.commit_basis,
        }


@dataclass
class VisibleConversationState:
    """Validator state derived only from agent-visible conversation events."""

    read_inspections: list[InspectionRecord] = field(default_factory=list)
    verified_identifiers: dict[str, set[str]] = field(default_factory=dict)
    latest_user_turn_tick: Optional[int] = None
    user_turn_ticks: list[int] = field(default_factory=list)
    pending_write: Optional[PendingWriteConfirmation] = None


class PendingWriteToolResult(BaseModel):
    """Result for a StageGate internal pending-write protocol tool call."""

    ok: bool
    reason: str
    pending_write: Optional[dict[str, object]] = None


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
        self._pending_write_events: list[dict[str, object]] = []

    def set_domain_name(self, domain_name: Optional[str]) -> None:
        """Update the public domain name used for domain-specific rules."""
        self.domain_name = normalize_domain(domain_name)

    def validate(
        self,
        tool_call: ToolCall,
        *,
        ledger: Optional[EntityLedger] = None,
        tick_index: Optional[int] = None,
        current_stage: Optional[str] = None,
    ) -> ValidatorDecision:
        """Validate a domain tool call without calling the tool."""
        repair_snapshot = (
            ledger.entity_repair_snapshot() if ledger is not None else None
        )
        repeated_failed_lookup = (
            ledger.failed_lookup_for_tool_call(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
            )
            if ledger is not None
            else None
        )
        if repeated_failed_lookup is not None:
            return ValidatorDecision(
                decision="block",
                reason="repeated_failed_lookup_blocked",
                checks={
                    "side_effecting_tool": False,
                    "repairable_entity_state": True,
                    "repeated_failed_lookup": True,
                },
                corrective_packet=build_entity_repair_corrective_packet(
                    tool_call=tool_call,
                    reason="repeated_failed_lookup_blocked",
                    repair_snapshot=repeated_failed_lookup,
                    read_tools=self._read_tool_names(),
                ),
            )

        if tool_call.name == TRANSFER_TOOL_NAME:
            if (
                repair_snapshot is not None
                and int(repair_snapshot.get("failure_count", 0)) < 3
            ):
                return ValidatorDecision(
                    decision="block",
                    reason="transfer_blocked_entity_repair",
                    checks={
                        "side_effecting_tool": False,
                        "repairable_entity_state": True,
                    },
                    corrective_packet=build_entity_repair_corrective_packet(
                        tool_call=tool_call,
                        reason="transfer_blocked_entity_repair",
                        repair_snapshot=repair_snapshot,
                        read_tools=self._read_tool_names(),
                    ),
                )
            transfer_block = self._transfer_block_for_pending_write(
                tool_call,
                tick_index=tick_index,
            )
            if transfer_block is not None:
                return transfer_block

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
            "pending_write_exists": False,
            "pending_write_needs_confirmation": False,
            "pending_write_committed": False,
            "pending_write_consumed": False,
        }
        reason_by_check = {
            "tool_arguments_complete": "incomplete_tool_arguments",
            "tool_arguments_non_ambiguous": "ambiguous_tool_arguments",
            "identity_verified": "missing_verified_identity",
            "exact_identifiers_verified": "missing_verified_identifier",
            "policy_state_inspected": "missing_policy_state_inspection",
            "policy_preconditions_represented": "missing_policy_precondition_state",
        }
        for check_name, reason in reason_by_check.items():
            if not checks[check_name]:
                return self._block(tool_call, reason=reason, checks=checks)

        mismatch = self._pending_write_mismatch(
            tool_call,
            tick_index=tick_index,
            current_stage=current_stage,
        )
        if mismatch is not None:
            return self._block(
                tool_call,
                reason="pending_write_mismatch",
                checks=checks,
                pending_write=mismatch,
            )

        pending_write = self._ensure_pending_write(
            tool_call,
            tick_index=tick_index,
            current_stage=current_stage,
        )
        checks["pending_write_exists"] = True
        checks["pending_write_needs_confirmation"] = (
            pending_write.status == "needs_confirmation"
        )
        checks["pending_write_committed"] = pending_write.commit_decision == "confirmed"
        checks["pending_write_consumed"] = pending_write.status == "consumed"

        if pending_write.status == "denied":
            return self._block(
                tool_call,
                reason="pending_write_denied",
                checks=checks,
                pending_write=pending_write,
            )
        if pending_write.status == "unclear":
            return self._block(
                tool_call,
                reason="pending_write_unclear",
                checks=checks,
                pending_write=pending_write,
            )
        pending_write.status = "needs_confirmation"
        return self._block(
            tool_call,
            reason="missing_confirmation",
            checks=checks,
            pending_write=pending_write,
        )

    def record_visible_message(
        self,
        *,
        role: Literal["assistant", "user"],
        content: Optional[str],
        tick_index: Optional[int] = None,
    ) -> None:
        """Reject transcript text as validator evidence for pending writes."""
        if role == "assistant":
            return
        raise ValueError(
            "UserMessage.content from audio-native chunks is simulator gold text; "
            "use record_user_turn with AGENT_VISIBLE_TRANSCRIPT ordering only."
        )

    def record_assistant_utterance(
        self,
        *,
        content: Optional[str],
        tick_index: Optional[int] = None,
        source: EvidenceSource | str = EvidenceSource.ASSISTANT_UTTERANCE,
    ) -> None:
        """Accept assistant transcript events without using text for validation."""
        source = ensure_runtime_evidence_source(source)
        if source is not EvidenceSource.ASSISTANT_UTTERANCE:
            raise ValueError(
                "assistant action evidence must use ASSISTANT_UTTERANCE source"
            )

    def record_user_turn(
        self,
        *,
        content: Optional[str],
        tick_index: Optional[int] = None,
        source: EvidenceSource | str,
    ) -> None:
        """Record user-turn ordering without parsing transcript semantics."""
        source = ensure_runtime_evidence_source(source)
        if source is not EvidenceSource.AGENT_VISIBLE_TRANSCRIPT:
            return
        if tick_index is not None:
            self.state.latest_user_turn_tick = tick_index
            self.state.user_turn_ticks.append(tick_index)
            self.state.user_turn_ticks = self.state.user_turn_ticks[-50:]
            pending_write = self._active_pending_write()
            if pending_write is not None:
                return

    def record_user_confirmation_evidence(
        self,
        *,
        content: Optional[str],
        tick_index: Optional[int] = None,
        source: EvidenceSource | str,
    ) -> None:
        """Backward-compatible wrapper that records only user-turn ordering."""
        self.record_user_turn(
            content=content,
            tick_index=tick_index,
            source=source,
        )

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

    def pending_write_snapshot(self) -> Optional[dict[str, object]]:
        """Return trace-safe pending-write state, when present."""
        pending_write = self.state.pending_write
        if pending_write is None:
            return None
        return pending_write.snapshot()

    def drain_pending_write_events(self) -> list[dict[str, object]]:
        """Return and clear pending-write trace events accumulated by validator."""
        events = list(self._pending_write_events)
        self._pending_write_events = []
        return events

    def commit_pending_write(
        self,
        *,
        decision: str,
        basis: str,
        notes: Optional[str] = None,
        tick_index: Optional[int] = None,
    ) -> PendingWriteToolResult:
        """Record the model's structured commit decision for the active write."""
        if decision not in {"confirmed", "denied", "unclear"}:
            return PendingWriteToolResult(
                ok=False,
                reason="invalid_commit_decision",
                pending_write=self.pending_write_snapshot(),
            )
        if basis not in {
            "latest_user_turn",
            "user_corrected_details",
            "user_declined",
            "unclear_response",
        }:
            return PendingWriteToolResult(
                ok=False,
                reason="invalid_commit_basis",
                pending_write=self.pending_write_snapshot(),
            )
        pending_write = self._active_pending_write()
        if pending_write is None:
            return PendingWriteToolResult(
                ok=False,
                reason="pending_write_not_found",
                pending_write=self.pending_write_snapshot(),
            )
        self._queue_pending_write_event(
            "pending_write_commit_requested",
            pending_write,
            tick_index=tick_index,
            extra={"decision": decision, "basis": basis},
        )
        if not self._user_turn_after_pending_write_seen(pending_write):
            return PendingWriteToolResult(
                ok=False,
                reason="missing_user_turn_after_pending_write",
                pending_write=pending_write.snapshot(),
            )
        if (
            pending_write.status == "unclear"
            and pending_write.commit_recorded_tick is not None
            and not any(
                tick > pending_write.commit_recorded_tick
                for tick in self.state.user_turn_ticks
            )
        ):
            return PendingWriteToolResult(
                ok=False,
                reason="missing_user_turn_after_unclear_commit",
                pending_write=pending_write.snapshot(),
            )

        pending_write.commit_recorded_tick = tick_index
        pending_write.commit_decision = decision
        pending_write.commit_basis = basis
        pending_write.confirmation_recorded_tick = tick_index
        pending_write.confirmation_basis = basis

        if decision == "unclear":
            pending_write.status = "unclear"
            self._queue_pending_write_event(
                "pending_write_unclear",
                pending_write,
                tick_index=tick_index,
                extra={"decision": decision, "basis": basis},
            )
            return PendingWriteToolResult(
                ok=True,
                reason="commit_unclear",
                pending_write=pending_write.snapshot(),
            )
        if decision == "denied":
            pending_write.status = "denied"
            self._queue_pending_write_event(
                "pending_write_denied",
                pending_write,
                tick_index=tick_index,
                extra={"decision": decision, "basis": basis},
            )
            return PendingWriteToolResult(
                ok=True,
                reason="commit_denied",
                pending_write=pending_write.snapshot(),
            )

        pending_write.status = "needs_confirmation"
        self._queue_pending_write_event(
            "pending_write_committed",
            pending_write,
            tick_index=tick_index,
            extra={"decision": decision, "basis": basis},
        )
        return PendingWriteToolResult(
            ok=True,
            reason="commit_confirmed",
            pending_write=pending_write.snapshot(),
        )

    def pending_write_domain_tool_call(
        self,
        *,
        tool_call_id: str,
        requestor: Literal["assistant", "user"] = "assistant",
    ) -> Optional[ToolCall]:
        """Return the stored original domain write for a confirmed commit."""
        pending_write = self._active_pending_write()
        if pending_write is None:
            return None
        if pending_write.commit_decision != "confirmed":
            return None
        return ToolCall(
            id=tool_call_id,
            name=pending_write.tool_name,
            arguments=deepcopy(pending_write.original_args),
            requestor=requestor,
        )

    def mark_pending_write_commit_failed(
        self,
        *,
        tool_call: ToolCall,
        tick_index: Optional[int] = None,
    ) -> None:
        """Trace a committed pending write whose environment execution errored."""
        pending_write = self.state.pending_write
        if pending_write is None:
            return
        if not self._pending_write_matches_tool_call(pending_write, tool_call):
            return
        pending_write.last_block_reason = "pending_write_commit_failed"
        self._queue_pending_write_event(
            "pending_write_commit_failed",
            pending_write,
            tick_index=tick_index,
        )

    def mark_side_effecting_write_consumed(
        self,
        *,
        tool_call: ToolCall,
        tick_index: Optional[int] = None,
    ) -> None:
        """Mark a successful side-effecting domain-tool result as consumed."""
        pending_write = self.state.pending_write
        if pending_write is None:
            return
        if pending_write.commit_decision != "confirmed":
            return
        if not self._pending_write_matches_tool_call(pending_write, tool_call):
            return
        pending_write.status = "consumed"
        self._queue_pending_write_event(
            "pending_write_consumed",
            pending_write,
            tick_index=tick_index,
        )

    def _block(
        self,
        tool_call: ToolCall,
        *,
        reason: str,
        checks: dict[str, bool],
        pending_write: Optional[PendingWriteConfirmation] = None,
    ) -> ValidatorDecision:
        if pending_write is not None:
            pending_write.last_block_reason = reason
        return ValidatorDecision(
            decision="block",
            reason=reason,
            checks=checks,
            corrective_packet=build_corrective_packet(
                tool_call=tool_call,
                reason=reason,
                read_tools=self._read_tool_names(),
                pending_write_tool_name=None
                if pending_write is None
                else pending_write.tool_name,
            ),
            pending_write_id=None
            if pending_write is None
            else pending_write.pending_write_id,
            args_fingerprint=None
            if pending_write is None
            else pending_write.args_fingerprint,
        )

    def _active_pending_write(self) -> Optional[PendingWriteConfirmation]:
        pending_write = self.state.pending_write
        if pending_write is None:
            return None
        if pending_write.status in {"consumed", "expired", "none"}:
            return None
        return pending_write

    def _transfer_block_for_pending_write(
        self,
        tool_call: ToolCall,
        *,
        tick_index: Optional[int],
    ) -> Optional[ValidatorDecision]:
        pending_write = self._active_pending_write()
        if pending_write is None:
            return None
        if pending_write.status not in TRANSFER_BLOCKING_PENDING_WRITE_STATUSES:
            return None

        self._queue_pending_write_event(
            "transfer_blocked_pending_write",
            pending_write,
            tick_index=tick_index,
        )
        return self._block(
            tool_call,
            reason="transfer_blocked_pending_write",
            checks={
                "side_effecting_tool": False,
                "pending_write_exists": True,
                "pending_write_needs_confirmation": pending_write.status
                == "needs_confirmation",
                "pending_write_committed": pending_write.commit_decision == "confirmed",
                "pending_write_consumed": False,
            },
            pending_write=pending_write,
        )

    def _ensure_pending_write(
        self,
        tool_call: ToolCall,
        *,
        tick_index: Optional[int],
        current_stage: Optional[str],
    ) -> PendingWriteConfirmation:
        active_pending = self._active_pending_write()
        if active_pending is not None and self._pending_write_matches_tool_call(
            active_pending,
            tool_call,
        ):
            return active_pending

        pending_write = self._pending_write_from_tool_call(
            tool_call,
            tick_index=tick_index,
            current_stage=current_stage,
            status="needs_confirmation",
        )
        self.state.pending_write = pending_write
        self._queue_pending_write_event(
            "pending_write_created",
            pending_write,
            tick_index=tick_index,
        )
        return pending_write

    def _pending_write_mismatch(
        self,
        tool_call: ToolCall,
        *,
        tick_index: Optional[int],
        current_stage: Optional[str],
    ) -> Optional[PendingWriteConfirmation]:
        active_pending = self._active_pending_write()
        if active_pending is None:
            return None
        if self._pending_write_matches_tool_call(active_pending, tool_call):
            return None

        active_pending.status = "mismatched_retry"
        self._queue_pending_write_event(
            "pending_write_mismatch",
            active_pending,
            tick_index=tick_index,
        )
        active_pending.status = "expired"
        self._queue_pending_write_event(
            "pending_write_expired",
            active_pending,
            tick_index=tick_index,
        )
        replacement = self._pending_write_from_tool_call(
            tool_call,
            tick_index=tick_index,
            current_stage=current_stage,
            status="needs_confirmation",
        )
        replacement.last_block_reason = "pending_write_mismatch"
        self.state.pending_write = replacement
        self._queue_pending_write_event(
            "pending_write_created",
            replacement,
            tick_index=tick_index,
        )
        return replacement

    def _pending_write_from_tool_call(
        self,
        tool_call: ToolCall,
        *,
        tick_index: Optional[int],
        current_stage: Optional[str],
        status: PendingWriteStatus,
    ) -> PendingWriteConfirmation:
        normalized_args = canonicalize_for_fingerprint(tool_call.arguments)
        fingerprint = args_fingerprint(tool_call.arguments)
        return PendingWriteConfirmation(
            pending_write_id=f"{tool_call.name}:{fingerprint[:12]}",
            tool_name=tool_call.name,
            original_args=deepcopy(tool_call.arguments),
            normalized_args=normalized_args,
            args_fingerprint=fingerprint,
            created_tick=tick_index,
            created_stage=current_stage,
            status=status,
        )

    def _pending_write_matches_tool_call(
        self,
        pending_write: PendingWriteConfirmation,
        tool_call: ToolCall,
    ) -> bool:
        return (
            pending_write.tool_name == tool_call.name
            and pending_write.args_fingerprint == args_fingerprint(tool_call.arguments)
        )

    def _user_turn_after_pending_write_seen(
        self,
        pending_write: PendingWriteConfirmation,
    ) -> bool:
        if pending_write.created_tick is None:
            return bool(self.state.user_turn_ticks)
        return any(
            tick > pending_write.created_tick for tick in self.state.user_turn_ticks
        )

    def _queue_pending_write_event(
        self,
        event_type: str,
        pending_write: PendingWriteConfirmation,
        *,
        tick_index: Optional[int],
        extra: Optional[dict[str, object]] = None,
    ) -> None:
        snapshot = pending_write.snapshot()
        snapshot["event_type"] = event_type
        snapshot["tick_index"] = tick_index
        if extra:
            snapshot.update(extra)
        self._pending_write_events.append(snapshot)

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
        return False

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


def canonicalize_for_fingerprint(value: Any) -> Any:
    """Return stable, trace-safe normalized data for pending-write matching."""
    if isinstance(value, dict):
        return {
            str(key): canonicalize_for_fingerprint(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, list):
        return [canonicalize_for_fingerprint(item) for item in value]
    if isinstance(value, str):
        return normalize_value(value) or ""
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return normalize_value(value) or str(value)


def args_fingerprint(arguments: dict[str, Any]) -> str:
    """Return a stable fingerprint for pending write arguments."""
    canonical = canonicalize_for_fingerprint(arguments)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def pending_write_next_required_steps(
    *,
    tool_name: str,
    status: str,
) -> list[dict[str, object]]:
    """Return model-actionable pending-write protocol steps for a status."""
    commit_args: dict[str, object] = {
        "decision": "confirmed",
        "basis": "latest_user_turn",
    }
    commit_step = {
        "step": "call_tool_if_user_confirms",
        "tool_name": COMMIT_PENDING_WRITE_TOOL_NAME,
        "arguments": commit_args,
        "instruction": (
            "If the user confirms, commit the active pending write. StageGate "
            "will execute the stored original write arguments."
        ),
    }
    no_advance_step = {
        "step": "do_not_advance_stage",
        "tool_name": ADVANCE_STAGE_TOOL_NAME,
        "instruction": "Do not call advance_stage before commit_pending_write.",
    }
    no_transfer_step = {
        "step": "do_not_transfer",
        "tool_name": TRANSFER_TOOL_NAME,
        "instruction": (
            "Do not transfer unless the user explicitly asks for a human or an "
            "unrecoverable error occurs."
        ),
    }

    if status in {"needs_confirmation", "mismatched_retry"}:
        return [
            {
                "step": "tell_user_pending_action",
                "instruction": "Tell the user the pending action and consequence.",
            },
            {
                "step": "ask_user_to_confirm",
                "instruction": "Ask the user to confirm.",
            },
            {
                "step": "wait_for_user_response",
                "instruction": "Wait for the user's response.",
            },
            commit_step,
            no_advance_step,
            no_transfer_step,
        ]
    if status == "unclear":
        return [
            {
                "step": "ask_one_clarification",
                "instruction": "Ask one concise clarification question about whether the user confirms the pending action.",
            },
            {
                "step": "wait_for_user_response",
                "instruction": "Wait for the user's clarified response.",
            },
            commit_step,
            no_advance_step,
            no_transfer_step,
        ]
    return []


def pending_write_allowed_internal_tools(*, status: str) -> list[str]:
    """Return StageGate tools allowed for the active pending-write status."""
    if status in {"needs_confirmation", "unclear", "mismatched_retry"}:
        return [COMMIT_PENDING_WRITE_TOOL_NAME]
    return []


def pending_write_disallowed_tools(*, status: str) -> list[str]:
    """Return tools the model should not call while pending write is active."""
    if status in {
        "needs_confirmation",
        "unclear",
        "mismatched_retry",
    }:
        return [ADVANCE_STAGE_TOOL_NAME, TRANSFER_TOOL_NAME]
    return []


def pending_write_next_tool_call(
    *,
    tool_name: str,
    status: str,
) -> Optional[dict[str, object]]:
    """Return the next concrete tool call the model should make, if any."""
    if status in {"needs_confirmation", "unclear", "mismatched_retry"}:
        return {
            "name": COMMIT_PENDING_WRITE_TOOL_NAME,
            "arguments": {
                "decision": "confirmed",
                "basis": "latest_user_turn",
            },
            "when": "after_user_confirms",
        }
    return None


def build_corrective_packet(
    *,
    tool_call: ToolCall,
    reason: str,
    read_tools: list[str],
    pending_write_tool_name: Optional[str] = None,
) -> StagePacket:
    """Build a corrective StageGate packet for a blocked tool call."""
    protocol_tool_name = pending_write_tool_name or tool_call.name
    protocol_status_by_reason = {
        "missing_confirmation": "needs_confirmation",
        "pending_write_mismatch": "mismatched_retry",
        "pending_write_unclear": "unclear",
        "transfer_blocked_pending_write": "needs_confirmation",
    }
    protocol_status = protocol_status_by_reason.get(reason)
    do_not = [
        f"Do not call {tool_call.name} again directly while the write is pending.",
        "Do not call advance_stage before commit_pending_write.",
    ]
    if protocol_status is not None:
        do_not.append(
            "Do not transfer to a human agent unless the user asks for human help "
            "or an unrecoverable error occurs."
        )

    return StagePacket(
        stage=stage_for_reason(reason),
        objective="Recover the missing prerequisite before executing a write/action tool.",
        known_facts={},
        missing_facts=[reason],
        ambiguous_facts=[],
        ask_next=corrective_instruction(
            tool_name=protocol_tool_name,
            reason=reason,
        ),
        allowed_read_tools=[
            tool_name
            for tool_name in read_tools
            if not (
                reason == "transfer_blocked_pending_write"
                and tool_name == TRANSFER_TOOL_NAME
            )
        ],
        allowed_write_tools=[],
        allowed_internal_tools=[]
        if protocol_status is None
        else pending_write_allowed_internal_tools(status=protocol_status),
        disallowed_tools=[]
        if protocol_status is None
        else pending_write_disallowed_tools(status=protocol_status),
        next_required_steps=[]
        if protocol_status is None
        else pending_write_next_required_steps(
            tool_name=protocol_tool_name,
            status=protocol_status,
        ),
        next_tool_call=None
        if protocol_status is None
        else pending_write_next_tool_call(
            tool_name=protocol_tool_name,
            status=protocol_status,
        ),
        do_not=do_not,
        exit_condition=(
            "commit_pending_write has been called with a structured decision after "
            "a later user turn."
        ),
        when_done=(
            "If commit_pending_write is called with decision=confirmed, StageGate "
            f"will execute the stored {protocol_tool_name} arguments once."
        ),
    )


def build_entity_repair_corrective_packet(
    *,
    tool_call: ToolCall,
    reason: str,
    repair_snapshot: dict[str, object],
    read_tools: list[str],
) -> StagePacket:
    """Build a corrective packet for failed lookup repair."""
    source_tool = str(repair_snapshot.get("source_tool") or tool_call.name)
    value = str(repair_snapshot.get("normalized_value", "the failed value"))
    field = str(repair_snapshot.get("field", "identifier"))
    return StagePacket(
        stage=str(
            repair_snapshot.get("preferred_stage") or "collect_required_exact_entities"
        ),
        objective="Repair the failed exact entity lookup before continuing.",
        known_facts={},
        missing_facts=[reason, field],
        ambiguous_facts=[],
        ask_next=entity_repair_instruction(repair_snapshot),
        allowed_read_tools=[
            read_tool for read_tool in read_tools if read_tool != source_tool
        ],
        allowed_write_tools=[],
        allowed_internal_tools=[],
        disallowed_tools=["transfer_to_human_agents"],
        next_required_steps=[
            {
                "step": "do_not_retry_failed_lookup",
                "tool_name": source_tool,
                "value": value,
                "instruction": "Do not retry the same lookup with the same failed value.",
            },
            {
                "step": "repair_exact_entity",
                "field": field,
                "instruction": (
                    f"Ask the user to spell the {field} one character at a time."
                ),
            },
        ],
        next_tool_call=None,
        do_not=[
            f"Do not call {source_tool} again with {value}.",
            "Do not guess another exact identifier.",
            "Do not transfer while the entity lookup can still be repaired.",
        ],
        exit_condition="The failed exact value is corrected or recovered from official context.",
        when_done="Retry only with a corrected exact value or official context.",
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


def entity_repair_instruction(repair_snapshot: dict[str, object]) -> str:
    """Return repair instruction for a failed lookup corrective packet."""
    field = str(repair_snapshot.get("field", "identifier"))
    if field in {"customer_name", "passenger_name"}:
        return (
            "Ask the user to spell the name one character at a time. Use the "
            "latest corrected name candidate unless that lookup also fails."
        )
    if field == "order_id":
        return (
            "Ask the user to spell the order ID one character at a time. If "
            "authenticated user context is available, inspect that user's orders "
            "rather than guessing another order ID."
        )
    return f"Ask the user to spell the {field} one character at a time."


def corrective_instruction(*, tool_name: str, reason: str) -> str:
    """Return concise corrective text for the model."""
    if reason == "missing_confirmation":
        return (
            "Tell the user the pending action and consequence, ask for "
            "confirmation, wait for the user's response, then call "
            'commit_pending_write({"decision": "confirmed", '
            '"basis": "latest_user_turn"}) if confirmed. Use decision=denied '
            "or unclear when appropriate. Do not retry the original domain write "
            "directly; StageGate executes the stored write after a confirmed commit."
        )
    if reason == "pending_write_mismatch":
        return (
            "A different write was attempted while another pending write was "
            "active. Tell the user the changed pending action and consequence, "
            "ask for confirmation, then call commit_pending_write with "
            "decision=confirmed, denied, or unclear after the user responds."
        )
    if reason == "transfer_blocked_pending_write":
        return (
            "A resolvable pending write is active. Do not transfer yet. Follow "
            "the active pending-write protocol: tell the user the pending action "
            "and consequence, ask for confirmation, wait for the user response, "
            "then call commit_pending_write."
        )
    if reason == "pending_write_denied":
        return (
            "The user declined the pending write. Do not retry the write unless "
            "a changed pending action is started and committed through "
            "commit_pending_write."
        )
    if reason == "pending_write_unclear":
        return (
            "The user response was unclear. Ask for clarification, then call "
            "commit_pending_write again after the next user response."
        )
    if reason == "missing_policy_state_inspection":
        return (
            "Use the narrowest read tool to inspect the current official state first."
        )
    if reason == "missing_policy_precondition_state":
        return "Inspect the policy-relevant state needed to establish the precondition first."
    if reason == "missing_verified_identifier":
        return "Verify the exact identifier with an official read tool."
    if reason == "missing_verified_identity":
        return "Authenticate or verify the account identity with an official read tool first."
    if reason == "ambiguous_tool_arguments":
        return "Ask one clarification question to resolve the ambiguous tool argument."
    return "Collect the missing required tool argument before retrying."


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
