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
    STALE = "stale"


class LedgerEvidence(BaseModel):
    """One visible event supporting a ledger slot value."""

    source: str
    event_id: Optional[str] = None
    turn_index: Optional[int] = None
    tick_index: Optional[int] = None
    tool_name: Optional[str] = None
    normalized_value: Optional[str] = None


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

        if (
            self.normalized_value is not None
            and normalized_value != self.normalized_value
            and self.status not in {LedgerStatus.MISSING, LedgerStatus.STALE}
        ):
            self.status = LedgerStatus.CONTRADICTED
            if normalized_value not in self.alternatives:
                self.alternatives.append(normalized_value)
            self.confidence = min(self.confidence, confidence)
            return self._state_marker() != previous_state

        self.value = value
        self.normalized_value = normalized_value
        if (
            self.status is LedgerStatus.CONTRADICTED
            and normalized_value == self.normalized_value
        ):
            self.status = LedgerStatus.CONTRADICTED
        elif status_rank(status) >= status_rank(self.status):
            self.status = status
            self.confidence = confidence
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

    def _state_marker(self) -> tuple[Optional[str], str, tuple[str, ...]]:
        return (
            self.normalized_value,
            self.status.value,
            tuple(self.alternatives),
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
            }
        }

    def missing_facts(self) -> list[str]:
        """Return slots that still lack a usable value."""
        return [
            field
            for field, slot in self.slots.items()
            if slot.status in {LedgerStatus.MISSING, LedgerStatus.STALE}
        ]

    def ambiguous_facts(self) -> list[str]:
        """Return slots with contradictory observed values."""
        ambiguous: list[str] = []
        for field, slot in self.slots.items():
            if slot.status is LedgerStatus.CONTRADICTED:
                values = [slot.normalized_value, *slot.alternatives]
                values_text = ", ".join(value for value in values if value)
                ambiguous.append(f"{field}: {values_text}")
        return ambiguous

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
        "item_id": {"item_id", "item_ids", "return_items", "exchange_items"},
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


def status_rank(status: LedgerStatus) -> int:
    """Return precedence used to avoid downgrading a slot."""
    return {
        LedgerStatus.MISSING: 0,
        LedgerStatus.STALE: 0,
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
        values = unique_values(values_by_key(payload, aliases))
        if values:
            facts[slot] = values[0] if len(values) == 1 else values

    if domain == "retail":
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
    return delta
