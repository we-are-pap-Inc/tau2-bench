"""Typed entity ledger for StageGate."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from tau2.voice.audio_native.openai.stagegate.stage_schema import EvidenceSource


class LedgerStatus(str, Enum):
    """Confirmation state for a ledger slot."""

    MISSING = "missing"
    HYPOTHESIZED = "hypothesized"
    HEARD_NOT_CONFIRMED = "heard_not_confirmed"
    REPEATED_BACK = "repeated_back"
    USER_CONFIRMED = "user_confirmed"
    TOOL_VERIFIED = "tool_verified"
    CONTRADICTED = "contradicted"
    FAILED_LOOKUP = "failed_lookup"
    SUPERSEDED = "superseded"
    INVALIDATED_BY_CORRECTION = "invalidated_by_correction"
    STALE = "stale"


class LedgerEvidence(BaseModel):
    """One visible event supporting a ledger slot value."""

    source: str
    event_id: Optional[str] = None
    turn_index: Optional[int] = None
    tick_index: Optional[int] = None
    tool_name: Optional[str] = None
    normalized_value: Optional[str] = None


class LedgerCandidate(BaseModel):
    """One observed value for a slot, including inactive repair state."""

    value: Any
    normalized_value: str
    status: LedgerStatus
    confidence: float = 0.0
    evidence: list[LedgerEvidence] = Field(default_factory=list)
    first_seen_tick: Optional[int] = None
    last_updated_tick: Optional[int] = None
    failure_count: int = 0

    def observe(
        self,
        *,
        value: Any,
        status: LedgerStatus,
        evidence: LedgerEvidence,
        confidence: float,
    ) -> None:
        """Update this candidate with another visible observation."""
        self.value = value
        self.evidence.append(evidence)
        if self.first_seen_tick is None:
            self.first_seen_tick = evidence.tick_index
        self.last_updated_tick = evidence.tick_index
        if self.status is LedgerStatus.FAILED_LOOKUP and status_rank(
            status
        ) < status_rank(LedgerStatus.TOOL_VERIFIED):
            self.confidence = min(self.confidence, confidence)
            return
        if status_rank(status) >= status_rank(self.status):
            self.status = status
            self.confidence = confidence

    def deactivate(
        self,
        *,
        status: LedgerStatus,
        evidence: LedgerEvidence,
    ) -> None:
        """Mark this candidate inactive using visible repair evidence."""
        self.status = status
        self.evidence.append(evidence)
        self.last_updated_tick = evidence.tick_index

    def mark_failed(self, *, evidence: LedgerEvidence) -> None:
        """Mark this candidate as failed by an official lookup result."""
        self.status = LedgerStatus.FAILED_LOOKUP
        self.confidence = 0.0
        self.failure_count += 1
        self.evidence.append(evidence)
        if self.first_seen_tick is None:
            self.first_seen_tick = evidence.tick_index
        self.last_updated_tick = evidence.tick_index

    def packet_fact(self) -> dict[str, object]:
        """Return a compact candidate representation for repair packets."""
        return {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "status": self.status.value,
            "failure_count": self.failure_count,
            "last_updated_tick": self.last_updated_tick,
        }


class LedgerSlot(BaseModel):
    """A typed dialogue-state slot."""

    field: str
    value: Any = None
    normalized_value: Optional[str] = None
    status: LedgerStatus = LedgerStatus.MISSING
    confidence: float = 0.0
    evidence: list[LedgerEvidence] = Field(default_factory=list)
    last_updated_turn: Optional[int] = None
    last_updated_tick: Optional[int] = None
    alternatives: list[str] = Field(default_factory=list)
    candidates: list[LedgerCandidate] = Field(default_factory=list)
    pending_events: list[dict[str, object]] = Field(default_factory=list, exclude=True)

    def observe(
        self,
        *,
        value: Any,
        status: LedgerStatus,
        evidence: LedgerEvidence,
        confidence: float,
    ) -> bool:
        """Record a visible observation and update slot state."""
        normalized_value = normalize_value(value)
        if normalized_value is None:
            return False

        previous_state = self._state_marker()
        evidence.normalized_value = normalized_value
        self.evidence.append(evidence)
        self.last_updated_turn = evidence.turn_index
        self.last_updated_tick = evidence.tick_index
        candidate = self._candidate_for(normalized_value)
        if candidate is None:
            candidate = LedgerCandidate(
                value=value,
                normalized_value=normalized_value,
                status=status,
                confidence=confidence,
                evidence=[],
                first_seen_tick=evidence.tick_index,
                last_updated_tick=evidence.tick_index,
            )
            self.candidates.append(candidate)
        candidate.observe(
            value=value,
            status=status,
            evidence=evidence,
            confidence=confidence,
        )

        if status is LedgerStatus.TOOL_VERIFIED:
            self._deactivate_other_candidates(
                normalized_value=normalized_value,
                status=LedgerStatus.SUPERSEDED,
                evidence=evidence,
                event_type="ledger_value_superseded",
                reason="tool_verified_value_dominates",
            )
        elif self.field in SOFT_CORRECTION_FIELDS:
            self._deactivate_other_candidates(
                normalized_value=normalized_value,
                status=LedgerStatus.INVALIDATED_BY_CORRECTION,
                evidence=evidence,
                event_type="ledger_value_superseded",
                reason="later_corrected_value",
                only_lower_confidence=True,
            )
        elif (
            self.status is LedgerStatus.TOOL_VERIFIED
            and normalized_value != self.normalized_value
        ):
            candidate.deactivate(
                status=LedgerStatus.SUPERSEDED,
                evidence=evidence,
            )
            self._queue_event(
                event_type="ledger_value_superseded",
                candidate=candidate,
                evidence=evidence,
                reason="existing_tool_verified_value_dominates",
            )

        self._refresh_active_state()
        return self._state_marker() != previous_state

    def mark_failed_lookup(
        self,
        *,
        value: Any,
        evidence: LedgerEvidence,
        reason: str,
    ) -> bool:
        """Mark a candidate as failed by an official lookup result."""
        normalized_value = normalize_value(value)
        if normalized_value is None:
            return False
        previous_state = self._state_marker()
        evidence.normalized_value = normalized_value
        self.evidence.append(evidence)
        self.last_updated_turn = evidence.turn_index
        self.last_updated_tick = evidence.tick_index
        candidate = self._candidate_for(normalized_value)
        if candidate is None:
            candidate = LedgerCandidate(
                value=value,
                normalized_value=normalized_value,
                status=LedgerStatus.FAILED_LOOKUP,
                confidence=0.0,
                evidence=[],
                first_seen_tick=evidence.tick_index,
                last_updated_tick=evidence.tick_index,
            )
            self.candidates.append(candidate)
        candidate.mark_failed(evidence=evidence)
        self._queue_event(
            event_type="ledger_value_failed_lookup",
            candidate=candidate,
            evidence=evidence,
            reason=reason,
        )
        self._refresh_active_state(prefer_failed=normalized_value)
        return self._state_marker() != previous_state

    def packet_fact(self) -> dict[str, object]:
        """Return the compact slot form used in stage packets."""
        fact: dict[str, object] = {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "status": self.status.value,
        }
        if self.evidence:
            fact["source"] = self.evidence[-1].source
        return fact

    def active_candidates(self) -> list[LedgerCandidate]:
        """Return candidates still usable for ordinary packet facts."""
        return [
            candidate
            for candidate in self.candidates
            if candidate.status not in INACTIVE_CANDIDATE_STATUSES
        ]

    def failed_candidates(self) -> list[LedgerCandidate]:
        """Return candidates failed by official lookup evidence."""
        return [
            candidate
            for candidate in self.candidates
            if candidate.status is LedgerStatus.FAILED_LOOKUP
        ]

    def failed_lookup_for_value(self, value: Any) -> Optional[LedgerCandidate]:
        """Return a failed candidate matching a proposed value."""
        normalized_value = normalize_value(value)
        if normalized_value is None:
            return None
        for candidate in self.failed_candidates():
            if candidate.normalized_value == normalized_value:
                return candidate
        return None

    def drain_pending_events(self) -> list[dict[str, object]]:
        """Return and clear candidate repair events."""
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def _candidate_for(self, normalized_value: str) -> Optional[LedgerCandidate]:
        for candidate in self.candidates:
            if candidate.normalized_value == normalized_value:
                return candidate
        return None

    def _deactivate_other_candidates(
        self,
        *,
        normalized_value: str,
        status: LedgerStatus,
        evidence: LedgerEvidence,
        event_type: str,
        reason: str,
        only_lower_confidence: bool = False,
    ) -> None:
        for candidate in self.candidates:
            if candidate.normalized_value == normalized_value:
                continue
            if candidate.status in INACTIVE_CANDIDATE_STATUSES:
                continue
            if only_lower_confidence and candidate.confidence > 0.7:
                continue
            candidate.deactivate(status=status, evidence=evidence)
            self._queue_event(
                event_type=event_type,
                candidate=candidate,
                evidence=evidence,
                reason=reason,
            )

    def _refresh_active_state(self, *, prefer_failed: Optional[str] = None) -> None:
        active = self.active_candidates()
        if not active:
            failed = self.failed_candidates()
            if failed:
                preferred = self._latest_candidate(
                    [
                        candidate
                        for candidate in failed
                        if candidate.normalized_value == prefer_failed
                    ]
                    or failed
                )
                self.value = preferred.value
                self.normalized_value = preferred.normalized_value
                self.status = LedgerStatus.FAILED_LOOKUP
                self.confidence = 0.0
                self.alternatives = []
                return
            self.value = None
            self.normalized_value = None
            self.status = LedgerStatus.MISSING
            self.confidence = 0.0
            self.alternatives = []
            return
        if len(active) == 1:
            candidate = active[0]
            self.value = candidate.value
            self.normalized_value = candidate.normalized_value
            self.status = candidate.status
            self.confidence = candidate.confidence
            self.alternatives = []
            return
        primary = active[0]
        self.value = primary.value
        self.normalized_value = primary.normalized_value
        self.status = LedgerStatus.CONTRADICTED
        self.confidence = min(candidate.confidence for candidate in active)
        self.alternatives = [
            candidate.normalized_value
            for candidate in active[1:]
            if candidate.normalized_value
        ]

    def _latest_candidate(
        self,
        candidates: list[LedgerCandidate],
    ) -> LedgerCandidate:
        return max(
            candidates,
            key=lambda candidate: (
                candidate.last_updated_tick is not None,
                candidate.last_updated_tick or -1,
            ),
        )

    def _queue_event(
        self,
        *,
        event_type: str,
        candidate: LedgerCandidate,
        evidence: LedgerEvidence,
        reason: str,
    ) -> None:
        self.pending_events.append(
            {
                "event_type": event_type,
                "field": self.field,
                "normalized_value": candidate.normalized_value,
                "status": candidate.status.value,
                "source_tool": evidence.tool_name,
                "failure_reason": reason,
                "active_candidate_count": len(self.active_candidates()),
                "tick_index": evidence.tick_index,
            }
        )

    def _state_marker(
        self,
    ) -> tuple[Optional[str], str, tuple[str, ...], tuple[tuple[str, str, int], ...]]:
        return (
            self.normalized_value,
            self.status.value,
            tuple(self.alternatives),
            tuple(
                (
                    candidate.normalized_value,
                    candidate.status.value,
                    candidate.failure_count,
                )
                for candidate in self.candidates
            ),
        )


class EntityLedger(BaseModel):
    """Domain-specific collection of typed ledger slots."""

    domain_name: Optional[str]
    slots: dict[str, LedgerSlot] = Field(default_factory=dict)

    @classmethod
    def for_domain(cls, domain_name: Optional[str]) -> "EntityLedger":
        """Initialize a ledger from public domain name only."""
        normalized_domain = (domain_name or "").strip().lower() or None
        return cls(
            domain_name=normalized_domain,
            slots={
                field: LedgerSlot(field=field)
                for field in DOMAIN_SLOTS.get(normalized_domain or "", [])
            },
        )

    def update_from_tool_args(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        event_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        tick_index: Optional[int] = None,
    ) -> list[dict[str, object]]:
        """Update slots from model-emitted function-call arguments."""
        return self._update_from_payload(
            payload=arguments,
            tool_name=tool_name,
            source=EvidenceSource.MODEL_TOOL_ARGUMENT.value,
            status=LedgerStatus.HEARD_NOT_CONFIRMED,
            confidence=0.7,
            event_id=event_id,
            turn_index=turn_index,
            tick_index=tick_index,
        )

    def update_from_tool_result(
        self,
        *,
        tool_name: str,
        content: Optional[str],
        event_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        tick_index: Optional[int] = None,
    ) -> list[dict[str, object]]:
        """Update slots from successful official domain-tool output."""
        payload = parse_tool_result(content)
        if payload is None:
            return []
        return self._update_from_payload(
            payload=payload,
            tool_name=tool_name,
            source=EvidenceSource.DOMAIN_TOOL_OUTPUT.value,
            status=LedgerStatus.TOOL_VERIFIED,
            confidence=1.0,
            event_id=event_id,
            turn_index=turn_index,
            tick_index=tick_index,
        )

    def update_from_failed_tool_result(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        content: Optional[str],
        error: bool,
        event_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        tick_index: Optional[int] = None,
    ) -> list[dict[str, object]]:
        """Update slots from official failed lookup output."""
        if not self.slots or not is_lookup_failure_result(
            tool_name=tool_name,
            content=content,
            error=error,
        ):
            return []
        facts = failed_lookup_facts_from_tool_args(
            domain_name=self.domain_name,
            tool_name=tool_name,
            arguments=arguments,
        )
        deltas: list[dict[str, object]] = []
        for field, value in facts.items():
            slot = self.slots.get(field)
            if slot is None:
                continue
            evidence = LedgerEvidence(
                source=EvidenceSource.DOMAIN_TOOL_OUTPUT.value,
                event_id=event_id,
                turn_index=turn_index,
                tick_index=tick_index,
                tool_name=tool_name,
            )
            slot.mark_failed_lookup(
                value=value,
                evidence=evidence,
                reason="official_lookup_not_found",
            )
            deltas.append(
                slot_delta(slot, source=EvidenceSource.DOMAIN_TOOL_OUTPUT.value)
            )
        return deltas

    def known_facts(self) -> dict[str, dict[str, object]]:
        """Return non-missing, non-ambiguous facts for a stage packet."""
        return {
            field: slot.packet_fact()
            for field, slot in self.slots.items()
            if slot.value is not None
            and slot.status
            not in {
                LedgerStatus.MISSING,
                LedgerStatus.CONTRADICTED,
                LedgerStatus.STALE,
                LedgerStatus.FAILED_LOOKUP,
                LedgerStatus.SUPERSEDED,
                LedgerStatus.INVALIDATED_BY_CORRECTION,
            }
        }

    def missing_facts(self) -> list[str]:
        """Return slots that still lack a usable value."""
        return [
            field
            for field, slot in self.slots.items()
            if slot.status
            in {
                LedgerStatus.MISSING,
                LedgerStatus.STALE,
                LedgerStatus.FAILED_LOOKUP,
            }
        ]

    def ambiguous_facts(self) -> list[str]:
        """Return slots with contradictory observed values."""
        ambiguous: list[str] = []
        for field, slot in self.slots.items():
            if slot.status is LedgerStatus.CONTRADICTED:
                active_values = [
                    candidate.normalized_value
                    for candidate in slot.active_candidates()
                    if candidate.normalized_value
                ]
                values = active_values or [slot.normalized_value, *slot.alternatives]
                values_text = ", ".join(value for value in values if value)
                ambiguous.append(f"{field}: {values_text}")
        return ambiguous

    def entity_repair_snapshot(self) -> Optional[dict[str, object]]:
        """Return the highest-priority repairable failed lookup state."""
        failed: list[tuple[str, LedgerCandidate]] = []
        for field, slot in self.slots.items():
            active = slot.active_candidates()
            for candidate in slot.failed_candidates():
                later_active = [
                    active_candidate
                    for active_candidate in active
                    if (active_candidate.last_updated_tick or -1)
                    > (candidate.last_updated_tick or -1)
                ]
                if later_active:
                    continue
                failed.append((field, candidate))
        if not failed:
            return None
        field, candidate = max(
            failed,
            key=lambda item: (
                item[1].last_updated_tick is not None,
                item[1].last_updated_tick or -1,
            ),
        )
        return {
            "field": field,
            "value": candidate.value,
            "normalized_value": candidate.normalized_value,
            "status": candidate.status.value,
            "source_tool": (
                candidate.evidence[-1].tool_name if candidate.evidence else None
            ),
            "failure_count": candidate.failure_count,
            "active_candidate_count": len(self.slots[field].active_candidates()),
            "preferred_stage": repair_stage_for_field(field),
        }

    def failed_lookup_for_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Optional[dict[str, object]]:
        """Return a repair snapshot if this call repeats a failed lookup."""
        facts = failed_lookup_facts_from_tool_args(
            domain_name=self.domain_name,
            tool_name=tool_name,
            arguments=arguments,
        )
        for field, value in facts.items():
            slot = self.slots.get(field)
            if slot is None:
                continue
            candidate = slot.failed_lookup_for_value(value)
            if candidate is None:
                continue
            return {
                "field": field,
                "value": candidate.value,
                "normalized_value": candidate.normalized_value,
                "status": candidate.status.value,
                "source_tool": tool_name,
                "failure_count": candidate.failure_count,
                "active_candidate_count": len(slot.active_candidates()),
                "preferred_stage": repair_stage_for_field(field),
            }
        return None

    def _update_from_payload(
        self,
        *,
        payload: Any,
        tool_name: str,
        source: str,
        status: LedgerStatus,
        confidence: float,
        event_id: Optional[str],
        turn_index: Optional[int],
        tick_index: Optional[int],
    ) -> list[dict[str, object]]:
        if not self.slots:
            return []
        facts = extract_domain_facts(
            domain_name=self.domain_name,
            tool_name=tool_name,
            payload=payload,
        )
        deltas: list[dict[str, object]] = []
        for field, value in facts.items():
            slot = self.slots.get(field)
            if slot is None:
                continue
            evidence = LedgerEvidence(
                source=source,
                event_id=event_id,
                turn_index=turn_index,
                tick_index=tick_index,
                tool_name=tool_name,
            )
            slot.observe(
                value=value,
                status=status,
                evidence=evidence,
                confidence=confidence,
            )
            deltas.append(slot_delta(slot, source=source))
        return deltas


DOMAIN_SLOTS = {
    "retail": [
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
    ],
    "airline": [
        "passenger_name",
        "reservation_id",
        "flight_number",
        "origin",
        "destination",
        "date",
        "requested_change",
        "fare_class",
        "payment_or_fee_acknowledgment",
        "confirmation",
    ],
    "telecom": [
        "account_id",
        "customer_name",
        "phone_line",
        "service_address",
        "plan_name",
        "device_id",
        "issue_type",
        "requested_change",
        "confirmation",
    ],
}

FIELD_ALIASES = {
    "retail": {
        "email": {"email"},
        "phone": {"phone", "phone_number"},
        "order_id": {"order_id"},
        "item_id": {"item_id"},
        "return_reason": {"return_reason", "reason", "cancel_reason"},
        "address": {"address", "new_address"},
        "payment_method": {
            "payment_method",
            "payment_method_id",
            "return_payment_method_id",
            "exchange_payment_method_id",
        },
        "confirmation": {"confirmation", "confirmed", "user_confirmation"},
    },
    "airline": {
        "reservation_id": {"reservation_id"},
        "flight_number": {"flight_number"},
        "origin": {"origin"},
        "destination": {"destination"},
        "date": {"date"},
        "fare_class": {"fare_class", "cabin", "cabin_class"},
        "payment_or_fee_acknowledgment": {
            "payment_or_fee_acknowledgment",
            "payment_id",
            "payment_ids",
            "payment",
            "amount",
        },
        "confirmation": {"confirmation", "confirmed", "user_confirmation"},
    },
    "telecom": {
        "account_id": {"account_id", "customer_id"},
        "phone_line": {"phone_line", "line_id", "phone_number", "phone"},
        "plan_name": {"plan_name", "plan", "plan_id"},
        "device_id": {"device_id"},
        "issue_type": {"issue_type", "issue", "problem", "reason"},
        "confirmation": {"confirmation", "confirmed", "user_confirmation"},
    },
}

INACTIVE_CANDIDATE_STATUSES = {
    LedgerStatus.MISSING,
    LedgerStatus.STALE,
    LedgerStatus.FAILED_LOOKUP,
    LedgerStatus.SUPERSEDED,
    LedgerStatus.INVALIDATED_BY_CORRECTION,
}

SOFT_CORRECTION_FIELDS = {
    "customer_name",
    "passenger_name",
}

IDENTITY_REPAIR_FIELDS = {
    "customer_name",
    "passenger_name",
    "email",
    "phone",
}

EXACT_ID_REPAIR_FIELDS = {
    "order_id",
    "reservation_id",
    "account_id",
    "phone_line",
    "flight_number",
}

LOOKUP_TOOL_PREFIXES = (
    "find_",
    "get_",
    "search_",
)

LOOKUP_FAILURE_CUES = (
    "not found",
    "no matching",
    "no record",
    "does not exist",
    "unable to find",
)

TRANSIENT_ERROR_CUES = (
    "api",
    "connection",
    "not connected",
    "provider",
    "rate limit",
    "temporary",
    "timeout",
    "try again",
    "unavailable",
)


def status_rank(status: LedgerStatus) -> int:
    """Return precedence used to avoid downgrading a slot."""
    return {
        LedgerStatus.MISSING: 0,
        LedgerStatus.STALE: 0,
        LedgerStatus.FAILED_LOOKUP: 0,
        LedgerStatus.SUPERSEDED: 0,
        LedgerStatus.INVALIDATED_BY_CORRECTION: 0,
        LedgerStatus.HYPOTHESIZED: 1,
        LedgerStatus.HEARD_NOT_CONFIRMED: 2,
        LedgerStatus.REPEATED_BACK: 3,
        LedgerStatus.USER_CONFIRMED: 4,
        LedgerStatus.TOOL_VERIFIED: 5,
        LedgerStatus.CONTRADICTED: 6,
    }[status]


def normalize_value(value: Any) -> Optional[str]:
    """Normalize a slot value for contradiction checks."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = " ".join(value.strip().lower().split())
        return normalized or None
    if isinstance(value, (int, float, bool)):
        return str(value).lower()
    if isinstance(value, list):
        values = [normalize_value(item) for item in value]
        filtered = sorted(value for value in values if value)
        return json.dumps(filtered, separators=(",", ":")) if filtered else None
    if isinstance(value, dict):
        return json.dumps(
            value, sort_keys=True, default=str, separators=(",", ":")
        ).lower()
    return str(value).strip().lower() or None


def parse_tool_result(content: Optional[str]) -> Any:
    """Parse a visible tool result payload if it carries structured data."""
    if not content:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, str):
        return None
    return payload


def is_lookup_failure_result(
    *,
    tool_name: str,
    content: Optional[str],
    error: bool,
) -> bool:
    """Return whether official tool output says a lookup failed."""
    if not tool_name.startswith(LOOKUP_TOOL_PREFIXES):
        return False
    text = tool_result_text(content)
    if text and any(cue in text for cue in LOOKUP_FAILURE_CUES):
        return True
    if not error:
        return False
    if text and any(cue in text for cue in TRANSIENT_ERROR_CUES):
        return False
    return text is None


def tool_result_text(content: Optional[str]) -> Optional[str]:
    """Return normalized informative text from a tool result payload."""
    if not content:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = content
    if isinstance(payload, str):
        text = payload.strip().lower()
        return text or None
    if isinstance(payload, dict):
        if not payload:
            return None
        return json.dumps(payload, sort_keys=True, default=str).lower()
    if isinstance(payload, list):
        if not payload:
            return None
        return json.dumps(payload, sort_keys=True, default=str).lower()
    if payload is None:
        return None
    text = str(payload).strip().lower()
    return text or None


def failed_lookup_facts_from_tool_args(
    *,
    domain_name: Optional[str],
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Extract exact values that were just used in a failed lookup."""
    domain = domain_name or ""
    if domain not in DOMAIN_SLOTS:
        return {}
    facts: dict[str, Any] = {}
    if tool_name == "find_user_id_by_name_zip":
        name = compose_name(arguments)
        if name:
            facts["customer_name"] = name
    for field, aliases in FIELD_ALIASES.get(domain, {}).items():
        values = unique_values(values_by_key(arguments, aliases))
        if values:
            facts[field] = values[0] if len(values) == 1 else values
    if not facts and tool_name.startswith("get_"):
        for key, value in arguments.items():
            if key.endswith("_id") and normalize_value(value) is not None:
                facts[key] = value
    return {
        field: value for field, value in facts.items() if field in DOMAIN_SLOTS[domain]
    }


def repair_stage_for_field(field: str) -> str:
    """Return the stage that should handle a failed lookup field."""
    if field in IDENTITY_REPAIR_FIELDS:
        return "identify_or_authenticate"
    if field in EXACT_ID_REPAIR_FIELDS:
        return "collect_required_exact_entities"
    return "inspect_state_with_read_tools"


def extract_domain_facts(
    *,
    domain_name: Optional[str],
    tool_name: str,
    payload: Any,
) -> dict[str, Any]:
    """Extract slot values from visible structured data for one domain."""
    domain = domain_name or ""
    if domain not in DOMAIN_SLOTS:
        return {}

    facts: dict[str, Any] = {}
    for slot, aliases in FIELD_ALIASES.get(domain, {}).items():
        if domain == "retail" and slot == "item_id":
            continue
        values = unique_values(values_by_key(payload, aliases))
        if values:
            facts[slot] = values[0] if len(values) == 1 else values

    if domain == "retail":
        facts.update(retail_item_id_facts(tool_name=tool_name, payload=payload))
        names = unique_values(extract_names(payload))
        if names:
            facts["customer_name"] = names[0] if len(names) == 1 else names
        intent = retail_intent_from_tool(tool_name)
        if intent:
            facts["refund_or_exchange_intent"] = intent
    elif domain == "airline":
        names = unique_values(extract_names(payload))
        if names:
            facts["passenger_name"] = names[0] if len(names) == 1 else names
        requested_change = requested_change_from_tool(tool_name)
        if requested_change:
            facts["requested_change"] = requested_change
    elif domain == "telecom":
        names = unique_values(extract_names(payload))
        if names:
            facts["customer_name"] = names[0] if len(names) == 1 else names
        plan_names = unique_values(extract_plan_names(payload))
        if plan_names:
            facts["plan_name"] = plan_names[0] if len(plan_names) == 1 else plan_names
        address = first_value(
            [
                *values_by_key(payload, {"service_address"}),
                *extract_address_dicts(payload),
            ]
        )
        if address is not None:
            facts["service_address"] = address
        requested_change = requested_change_from_tool(tool_name)
        if requested_change:
            facts["requested_change"] = requested_change

    return {
        field: value for field, value in facts.items() if field in DOMAIN_SLOTS[domain]
    }


def retail_item_id_facts(*, tool_name: str, payload: Any) -> dict[str, Any]:
    """Extract retail item identifiers into semantic slots."""
    facts: dict[str, Any] = {}
    if isinstance(payload, dict):
        if tool_name == "exchange_delivered_order_items":
            old_ids = flat_unique_values(values_by_key(payload, {"item_ids"}))
            new_ids = flat_unique_values(values_by_key(payload, {"new_item_ids"}))
            if old_ids:
                facts["selected_old_item_ids"] = old_ids
            if new_ids:
                facts["selected_new_item_ids"] = new_ids
        elif tool_name == "return_delivered_order_items":
            old_ids = flat_unique_values(values_by_key(payload, {"item_ids"}))
            if old_ids:
                facts["selected_old_item_ids"] = old_ids
        elif tool_name == "get_order_details":
            order_ids = flat_unique_values(
                values_by_key(payload, {"item_id", "item_ids"})
            )
            if order_ids:
                facts["order_item_ids"] = order_ids
                facts["item_id"] = order_ids[0] if len(order_ids) == 1 else order_ids
        elif tool_name == "get_item_details":
            item_ids = flat_unique_values(values_by_key(payload, {"item_id"}))
            if item_ids:
                facts["item_id"] = item_ids[0] if len(item_ids) == 1 else item_ids
        elif tool_name == "get_product_details":
            replacement_ids = flat_unique_values(values_by_key(payload, {"item_id"}))
            if replacement_ids:
                facts["candidate_replacement_item_ids"] = replacement_ids
    return facts


def values_by_key(payload: Any, aliases: set[str]) -> list[Any]:
    """Recursively collect values whose key matches an alias."""
    values: list[Any] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_normalized = key.lower()
            if key_normalized in aliases and value not in (None, "", []):
                values.append(value)
            values.extend(values_by_key(value, aliases))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(values_by_key(item, aliases))
    return values


def extract_names(payload: Any) -> list[str]:
    """Recursively collect customer/passenger names from structured payloads."""
    names: list[str] = []
    if isinstance(payload, dict):
        full_name = payload.get("full_name")
        if isinstance(full_name, str) and full_name.strip():
            names.append(full_name)
        name = payload.get("name")
        if isinstance(name, dict):
            composed = compose_name(name)
            if composed:
                names.append(composed)
        composed = compose_name(payload)
        if composed:
            names.append(composed)
        for value in payload.values():
            names.extend(extract_names(value))
    elif isinstance(payload, list):
        for item in payload:
            names.extend(extract_names(item))
    return names


def extract_plan_names(payload: Any) -> list[str]:
    """Recursively collect telecom plan display names."""
    plan_names: list[str] = []
    if isinstance(payload, dict):
        name = payload.get("name")
        if "plan_id" in payload and isinstance(name, str) and name.strip():
            plan_names.append(name)
        for value in payload.values():
            plan_names.extend(extract_plan_names(value))
    elif isinstance(payload, list):
        for item in payload:
            plan_names.extend(extract_plan_names(item))
    return plan_names


def compose_name(payload: dict[str, Any]) -> Optional[str]:
    """Build a display name from first and last name keys."""
    first_name = payload.get("first_name")
    last_name = payload.get("last_name")
    if isinstance(first_name, str) and isinstance(last_name, str):
        name = f"{first_name.strip()} {last_name.strip()}".strip()
        return name or None
    return None


def extract_address_dicts(payload: Any) -> list[dict[str, Any]]:
    """Collect address-shaped dictionaries from visible payloads."""
    addresses: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        address_keys = {"street", "address1", "city", "state", "zip", "zip_code"}
        if address_keys & {key.lower() for key in payload}:
            addresses.append(payload)
        for value in payload.values():
            addresses.extend(extract_address_dicts(value))
    elif isinstance(payload, list):
        for item in payload:
            addresses.extend(extract_address_dicts(item))
    return addresses


def retail_intent_from_tool(tool_name: str) -> Optional[str]:
    """Infer retail action intent from an agent-visible tool name."""
    lowered = tool_name.lower()
    for intent in ("exchange", "return", "refund", "cancel"):
        if intent in lowered:
            return intent
    return None


def requested_change_from_tool(tool_name: str) -> Optional[str]:
    """Infer requested change from an agent-visible domain tool name."""
    lowered = tool_name.lower()
    action_prefixes = (
        "book",
        "cancel",
        "disable",
        "enable",
        "modify",
        "refuel",
        "resume",
        "send",
        "suspend",
        "update",
    )
    if any(lowered.startswith(prefix) for prefix in action_prefixes):
        return tool_name
    return None


def unique_values(values: list[Any]) -> list[Any]:
    """Return values deduplicated by normalized representation."""
    seen: set[str] = set()
    unique: list[Any] = []
    for value in values:
        normalized = normalize_value(value)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique


def flat_unique_values(values: list[Any]) -> list[Any]:
    """Return unique scalar values from possibly nested list values."""
    flattened: list[Any] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    return unique_values(flattened)


def first_value(values: list[Any]) -> Any:
    """Return the first non-empty value from a list."""
    for value in values:
        if normalize_value(value) is not None:
            return value
    return None


def slot_delta(slot: LedgerSlot, *, source: str) -> dict[str, object]:
    """Return a trace-friendly representation of a slot update."""
    delta: dict[str, object] = {
        "field": slot.field,
        "value": slot.value,
        "normalized_value": slot.normalized_value,
        "status": slot.status.value,
        "source": source,
    }
    if slot.alternatives:
        delta["alternatives"] = list(slot.alternatives)
    if slot.evidence:
        delta["evidence"] = slot.evidence[-1].model_dump(mode="json")
    events = slot.drain_pending_events()
    if events:
        delta["events"] = events
    return delta
