"""Stage packet generation for StageGate."""

import re
from dataclasses import dataclass
from typing import Optional

from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.stagegate.ledger import (
    EntityLedger,
    LedgerStatus,
)
from tau2.voice.audio_native.openai.stagegate.stage_schema import StagePacket

STAGES = [
    "understand_intent",
    "identify_or_authenticate",
    "collect_required_exact_entities",
    "inspect_state_with_read_tools",
    "check_policy_eligibility",
    "propose_action_and_confirm",
    "execute_write_action",
    "verify_result_and_close",
]

OBJECTIVES = {
    "understand_intent": "Understand the customer's requested outcome.",
    "identify_or_authenticate": "Establish identity before account-specific actions.",
    "collect_required_exact_entities": "Collect exact identifiers and action details.",
    "inspect_state_with_read_tools": "Inspect official current state before deciding.",
    "check_policy_eligibility": "Check the public policy requirements for the action.",
    "propose_action_and_confirm": "Summarize the intended change and get confirmation.",
    "execute_write_action": "Execute the confirmed write action.",
    "verify_result_and_close": "Verify the result and close the conversation.",
}

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

WRITE_TOOL_PREFIXES = (
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

NON_SIDE_EFFECTING_TOOL_NAMES = {
    "advance_stage",
    "calculate",
    "transfer_to_human_agents",
}

IDENTITY_READ_HINTS = {
    "retail": ("find_user", "get_user", "get_order"),
    "airline": ("get_reservation", "get_user"),
    "telecom": ("get_customer", "get_details"),
    "mock": ("get_account", "get_users"),
}

STAGE_SLOT_SCOPE = {
    "identify_or_authenticate": {
        "retail": ("customer_name", "email", "phone", "order_id"),
        "airline": ("passenger_name", "reservation_id"),
        "telecom": ("account_id", "customer_name", "phone_line"),
        "mock": ("account_id",),
    },
    "collect_required_exact_entities": {
        "retail": (
            "order_id",
            "item_id",
            "return_reason",
            "refund_or_exchange_intent",
        ),
        "airline": (
            "reservation_id",
            "flight_number",
            "origin",
            "destination",
            "date",
            "requested_change",
            "fare_class",
        ),
        "telecom": (
            "account_id",
            "phone_line",
            "service_address",
            "plan_name",
            "device_id",
            "issue_type",
            "requested_change",
        ),
        "mock": ("account_id",),
    },
    "inspect_state_with_read_tools": {
        "retail": ("order_id", "item_id", "payment_method"),
        "airline": ("reservation_id", "flight_number", "date"),
        "telecom": ("account_id", "phone_line", "device_id"),
        "mock": ("account_id",),
    },
    "check_policy_eligibility": {
        "retail": (
            "order_id",
            "item_id",
            "return_reason",
            "refund_or_exchange_intent",
            "payment_method",
        ),
        "airline": (
            "reservation_id",
            "flight_number",
            "date",
            "requested_change",
            "fare_class",
            "payment_or_fee_acknowledgment",
        ),
        "telecom": (
            "account_id",
            "phone_line",
            "service_address",
            "issue_type",
            "requested_change",
        ),
        "mock": ("account_id",),
    },
    "propose_action_and_confirm": {
        "retail": ("confirmation",),
        "airline": ("confirmation",),
        "telecom": ("confirmation",),
        "mock": ("confirmation",),
    },
    "execute_write_action": {
        "retail": ("confirmation",),
        "airline": ("confirmation",),
        "telecom": ("confirmation",),
        "mock": ("confirmation",),
    },
}

STAGE_MISSING_HINTS = {
    "understand_intent": {
        "default": ("customer request",),
    },
    "identify_or_authenticate": {
        "retail": ("customer email, phone, or name and ZIP", "order ID if known"),
        "airline": ("passenger name or reservation ID",),
        "telecom": ("account ID, customer name, or phone line",),
        "mock": ("account ID",),
    },
    "collect_required_exact_entities": {
        "retail": ("order ID", "item ID or requested order change"),
        "airline": ("reservation or flight details", "requested travel change"),
        "telecom": ("account or line ID", "requested service change"),
        "mock": ("account ID",),
    },
    "inspect_state_with_read_tools": {
        "retail": ("official order or item state",),
        "airline": ("official reservation or flight state",),
        "telecom": ("official account, line, bill, or device state",),
        "mock": ("official account state",),
    },
    "check_policy_eligibility": {
        "retail": ("policy-relevant order status and payment/refund facts",),
        "airline": ("policy-relevant itinerary, fare, and fee facts",),
        "telecom": ("policy-relevant account, line, bill, or service facts",),
        "mock": ("policy-relevant account status",),
    },
    "propose_action_and_confirm": {
        "default": ("explicit user confirmation",),
    },
    "execute_write_action": {
        "default": ("validated write/action call",),
    },
    "verify_result_and_close": {
        "default": ("final tool result or customer acknowledgement",),
    },
}

SLOT_LABELS = {
    "customer_name": "customer name",
    "email": "customer email",
    "phone": "customer phone",
    "order_id": "order ID",
    "item_id": "item ID",
    "return_reason": "return or cancellation reason",
    "refund_or_exchange_intent": "refund, exchange, return, or cancellation intent",
    "address": "address",
    "payment_method": "payment method",
    "passenger_name": "passenger name",
    "reservation_id": "reservation ID",
    "flight_number": "flight number",
    "origin": "origin",
    "destination": "destination",
    "date": "date",
    "requested_change": "requested change",
    "fare_class": "fare class",
    "payment_or_fee_acknowledgment": "payment or fee acknowledgment",
    "account_id": "account ID",
    "phone_line": "phone line",
    "service_address": "service address",
    "plan_name": "plan name",
    "device_id": "device ID",
    "issue_type": "issue type",
    "confirmation": "explicit user confirmation",
}

HINT_ALTERNATIVE_RE = re.compile(r"\s+or\s+|,")
HINT_TOKEN_RE = re.compile(r"[a-z0-9]+")
HINT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "customer",
        "fact",
        "facts",
        "for",
        "from",
        "if",
        "in",
        "is",
        "known",
        "of",
        "official",
        "on",
        "or",
        "passenger",
        "policy",
        "relevant",
        "the",
        "to",
        "user",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True)
class ToolInventory:
    """Public tools split into read and write/action candidates."""

    read_tools: tuple[str, ...]
    write_tools: tuple[str, ...]


class StagePacketOrchestrator:
    """Deterministic stage packet generator."""

    def build_packet(
        self,
        *,
        current_stage: str,
        observed_facts: list[str],
        last_action: str,
        blocker: str | None,
        tools: list[Tool],
        domain_name: Optional[str] = None,
        ledger: EntityLedger | None = None,
    ) -> StagePacket:
        """Build a compact packet from visible state."""
        stage = self._choose_stage(current_stage, blocker)
        domain = self._domain(domain_name=domain_name, ledger=ledger)
        inventory = self._split_tools(tools)
        allowed_read_tools = self._allowed_read_tools(
            stage=stage,
            domain=domain,
            inventory=inventory,
        )
        allowed_write_tools = (
            list(inventory.write_tools) if stage == "execute_write_action" else []
        )
        known_facts = self._known_facts(observed_facts)
        missing = self._stage_missing_facts(
            stage=stage,
            domain=domain,
            observed_facts=observed_facts,
            ledger=ledger,
        )
        ambiguous: list[str] = []
        if ledger is not None:
            known_facts = self._ledger_known_facts(
                ledger=ledger,
                stage=stage,
                domain=domain,
            )
            ambiguous.extend(self._stage_ambiguous_facts(ledger, stage, domain))
        if blocker:
            missing.insert(0, blocker)
        ask_next = self._ask_next(
            stage,
            missing,
            observed_facts,
            last_action,
            ambiguous,
            allowed_read_tools,
            allowed_write_tools,
            domain,
        )
        return StagePacket(
            stage=stage,
            objective=OBJECTIVES.get(stage, OBJECTIVES["understand_intent"]),
            known_facts=known_facts,
            missing_facts=missing,
            ambiguous_facts=ambiguous,
            ask_next=ask_next,
            allowed_read_tools=allowed_read_tools,
            allowed_write_tools=allowed_write_tools,
            do_not=self._do_not(stage, domain),
            exit_condition=self._exit_condition(stage, domain),
            when_done="Call advance_stage again with updated visible facts and the last tool result.",
        )

    def build_fallback_packet(
        self,
        *,
        packet: StagePacket,
        guard_reason: str,
    ) -> StagePacket:
        """Build a non-mutating fallback packet when stage looping is detected."""
        ask_next = self._fallback_ask_next(packet)
        allowed_read_tools: list[str] = []
        if packet.allowed_read_tools and "Call " in ask_next:
            allowed_read_tools = [packet.allowed_read_tools[0]]
        return StagePacket(
            stage=packet.stage,
            objective="Break the StageGate loop and make one safe next move.",
            known_facts=packet.known_facts,
            missing_facts=packet.missing_facts,
            ambiguous_facts=packet.ambiguous_facts,
            ask_next=ask_next,
            allowed_read_tools=allowed_read_tools,
            allowed_write_tools=[],
            do_not=[
                "Do not call advance_stage again immediately.",
                "Do not call write/action tools from this fallback packet.",
            ],
            exit_condition="One safe next action is completed, or the customer is gracefully handed off or closed.",
            when_done=(
                "Continue only after the customer answers, an allowed read tool returns, "
                "or the conversation is safely closed."
            ),
        )

    def _known_facts(self, observed_facts: list[str]) -> dict[str, dict[str, str]]:
        return {
            f"observed_fact_{idx}": {"value": fact, "source": "advance_stage_args"}
            for idx, fact in enumerate(observed_facts, start=1)
            if fact
        }

    def _choose_stage(self, current_stage: str, blocker: str | None) -> str:
        if current_stage not in STAGES:
            return STAGES[0]
        if blocker:
            return current_stage
        idx = STAGES.index(current_stage)
        return STAGES[min(idx + 1, len(STAGES) - 1)]

    def _split_tools(self, tools: list[Tool]) -> ToolInventory:
        read_tools: list[str] = []
        write_tools: list[str] = []
        for tool in tools:
            if tool.name in NON_SIDE_EFFECTING_TOOL_NAMES:
                if tool.name != "advance_stage":
                    read_tools.append(tool.name)
                continue
            if self._is_write_tool(tool):
                write_tools.append(tool.name)
            else:
                read_tools.append(tool.name)
        return ToolInventory(
            read_tools=tuple(sorted(read_tools)),
            write_tools=tuple(sorted(write_tools)),
        )

    def _ask_next(
        self,
        stage: str,
        missing: list[str],
        observed_facts: list[str],
        last_action: str,
        ambiguous: list[str],
        allowed_read_tools: list[str],
        allowed_write_tools: list[str],
        domain: str,
    ) -> str:
        if ambiguous:
            return f"Ask one concise clarification question for {ambiguous[0]}."
        if missing:
            return f"Ask the customer for {missing[0]}."
        if stage == "inspect_state_with_read_tools":
            if allowed_read_tools:
                return f"Call {allowed_read_tools[0]} to inspect the official current state."
            return "Ask for the exact identifier needed to inspect official state."
        if stage == "check_policy_eligibility":
            if allowed_read_tools:
                return f"Use {allowed_read_tools[0]} or the last read result to check the public policy requirements."
            return "Explain that the policy cannot be checked until official state is available."
        if stage == "propose_action_and_confirm":
            return "Summarize the intended change and consequence, then ask: Do you confirm?"
        if stage == "execute_write_action":
            if allowed_write_tools:
                return f"Call the confirmed write/action tool, choosing from: {', '.join(allowed_write_tools)}."
            return "Do not write yet; ask for the missing confirmed action details."
        if stage == "verify_result_and_close":
            return "State the outcome from the tool result and ask if anything else is needed."
        if stage == "identify_or_authenticate":
            return (
                f"Ask for one {self._identity_phrase(domain)} value to verify identity."
            )
        if stage == "collect_required_exact_entities":
            return "Ask for one exact ID, date, item, line, flight, or requested change needed next."
        if observed_facts:
            return (
                "Continue with the next policy-required step using only verified facts."
            )
        if last_action:
            return "Use the last action result to choose the next policy-required step."
        return "Ask one concise clarification question."

    def _do_not(self, stage: str, domain: str) -> list[str]:
        if stage == "execute_write_action":
            return [
                "Do not invent tool results or hidden benchmark facts.",
                "Do not call a write/action tool unless the validator prerequisites are satisfied.",
            ]
        if stage == "verify_result_and_close":
            return ["Do not reopen new work unless the customer asks for it."]
        if stage == "identify_or_authenticate":
            return [
                f"Do not modify {self._domain_record_name(domain)} state yet.",
                "Do not ask for later-stage action details at this stage unless the user already raised them.",
            ]
        return [
            f"Do not modify {self._domain_mutable_object(domain)} state yet.",
        ]

    def _exit_condition(self, stage: str, domain: str) -> str:
        if stage == "execute_write_action":
            return "The confirmed write tool has completed or returned an error."
        if stage == "verify_result_and_close":
            return (
                "The customer understands the outcome and no further action is needed."
            )
        if stage == "identify_or_authenticate":
            return f"Identity or the relevant {self._domain_record_name(domain)} is verified by official read-tool output."
        if stage == "inspect_state_with_read_tools":
            return (
                "The official current state for the relevant record has been inspected."
            )
        if stage == "propose_action_and_confirm":
            return "The assistant has summarized the action and the user explicitly confirmed."
        return "The stage-scoped visible facts are collected or verified."

    def _domain(
        self,
        *,
        domain_name: Optional[str],
        ledger: EntityLedger | None,
    ) -> str:
        if domain_name:
            return domain_name.strip().lower()
        if ledger is not None and ledger.domain_name:
            return ledger.domain_name.strip().lower()
        return "default"

    def _is_write_tool(self, tool: Tool) -> bool:
        if tool.name.startswith(READ_TOOL_PREFIXES):
            return False
        if tool.name.startswith(WRITE_TOOL_PREFIXES):
            return True
        description = f"{tool.short_desc}\n{tool.long_desc}".lower()
        return any(
            cue in description
            for cue in (
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
        )

    def _allowed_read_tools(
        self,
        *,
        stage: str,
        domain: str,
        inventory: ToolInventory,
    ) -> list[str]:
        if stage in {
            "understand_intent",
            "propose_action_and_confirm",
            "execute_write_action",
        }:
            return []
        if stage == "identify_or_authenticate":
            hints = IDENTITY_READ_HINTS.get(domain, ())
            filtered = [
                name
                for name in inventory.read_tools
                if any(hint in name for hint in hints)
            ]
            return filtered or list(inventory.read_tools)
        return list(inventory.read_tools)

    def _stage_missing_facts(
        self,
        *,
        stage: str,
        domain: str,
        observed_facts: list[str],
        ledger: EntityLedger | None,
    ) -> list[str]:
        if ledger is None:
            hints = STAGE_MISSING_HINTS.get(stage, {})
            return [
                hint
                for hint in hints.get(domain, hints.get("default", ()))
                if not self._observed_fact_mentions(observed_facts, hint)
            ]
        slots = self._stage_slots(stage, domain)
        missing: list[str] = []
        for slot_name in slots:
            slot = ledger.slots.get(slot_name)
            if slot is None:
                continue
            if slot.status in {LedgerStatus.MISSING, LedgerStatus.STALE}:
                missing.append(SLOT_LABELS.get(slot_name, slot_name))
        if not missing:
            return []
        return missing

    def _ledger_known_facts(
        self,
        *,
        ledger: EntityLedger,
        stage: str,
        domain: str,
    ) -> dict[str, dict[str, object]]:
        stage_slots = set(self._stage_slots(stage, domain))
        if not stage_slots:
            stage_slots = set(ledger.slots)
        return {
            field: fact
            for field, fact in ledger.known_facts().items()
            if field in stage_slots
        }

    def _stage_ambiguous_facts(
        self,
        ledger: EntityLedger,
        stage: str,
        domain: str,
    ) -> list[str]:
        stage_slots = set(self._stage_slots(stage, domain))
        if not stage_slots:
            return ledger.ambiguous_facts()
        return [
            fact
            for fact in ledger.ambiguous_facts()
            if fact.split(":", maxsplit=1)[0] in stage_slots
        ]

    def _stage_slots(self, stage: str, domain: str) -> tuple[str, ...]:
        stage_scope = STAGE_SLOT_SCOPE.get(stage, {})
        return stage_scope.get(domain, stage_scope.get("default", ()))

    def _observed_fact_mentions(self, observed_facts: list[str], hint: str) -> bool:
        text_tokens = set(self._meaningful_hint_tokens(" ".join(observed_facts)))
        if not text_tokens:
            return False
        for alternative in HINT_ALTERNATIVE_RE.split(hint):
            hint_tokens = self._meaningful_hint_tokens(alternative)
            if hint_tokens and all(token in text_tokens for token in hint_tokens):
                return True
        return False

    def _meaningful_hint_tokens(self, text: str) -> tuple[str, ...]:
        return tuple(
            token
            for token in HINT_TOKEN_RE.findall(text.lower())
            if token not in HINT_STOPWORDS and (len(token) > 2 or token == "id")
        )

    def _fallback_ask_next(self, packet: StagePacket) -> str:
        if packet.ambiguous_facts:
            return f"Ask one concise clarification question for {packet.ambiguous_facts[0]}."
        if packet.missing_facts:
            return f"Ask the customer for {packet.missing_facts[0]}."
        if packet.allowed_read_tools:
            return f"Call {packet.allowed_read_tools[0]} with the best verified identifier available."
        return "Tell the customer you cannot safely continue and offer to close or transfer the conversation."

    def _identity_phrase(self, domain: str) -> str:
        return {
            "retail": "customer email, phone, name and ZIP, or order ID",
            "airline": "passenger name or reservation ID",
            "telecom": "account ID, customer name, or phone line",
            "mock": "account ID",
        }.get(domain, "identity")

    def _domain_mutable_object(self, domain: str) -> str:
        return {
            "retail": "order, address, payment, refund, return, or exchange",
            "airline": "reservation, flight, passenger, baggage, or certificate",
            "telecom": "account, line, plan, bill, roaming, or data",
            "mock": "account or task",
        }.get(domain, "customer")

    def _domain_record_name(self, domain: str) -> str:
        return {
            "retail": "customer or order",
            "airline": "passenger or reservation",
            "telecom": "account or phone line",
            "mock": "account",
        }.get(domain, "customer record")
