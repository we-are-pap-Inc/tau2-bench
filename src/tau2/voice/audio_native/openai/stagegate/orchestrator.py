"""Stage packet generation for StageGate."""

from tau2.environment.tool import Tool
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

WRITE_HINTS = (
    "book",
    "cancel",
    "create",
    "delete",
    "disable",
    "enable",
    "exchange",
    "modify",
    "refund",
    "resume",
    "return",
    "send",
    "suspend",
    "transfer",
    "update",
)


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
    ) -> StagePacket:
        """Build a compact packet from visible state."""
        stage = self._choose_stage(current_stage, blocker)
        read_tools, write_tools = self._split_tools(tools)
        missing: list[str] = []
        if blocker:
            missing.append(blocker)
        ask_next = self._ask_next(stage, missing, observed_facts, last_action)
        return StagePacket(
            stage=stage,
            objective=OBJECTIVES.get(stage, OBJECTIVES["understand_intent"]),
            known_facts=self._known_facts(observed_facts),
            missing_facts=missing,
            ask_next=ask_next,
            allowed_read_tools=read_tools,
            allowed_write_tools=write_tools if stage == "execute_write_action" else [],
            do_not=self._do_not(stage),
            exit_condition=self._exit_condition(stage),
            when_done="Call advance_stage again with updated visible facts and the last tool result.",
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

    def _split_tools(self, tools: list[Tool]) -> tuple[list[str], list[str]]:
        read_tools: list[str] = []
        write_tools: list[str] = []
        for tool in tools:
            if tool.name == "advance_stage":
                continue
            if any(hint in tool.name for hint in WRITE_HINTS):
                write_tools.append(tool.name)
            else:
                read_tools.append(tool.name)
        return sorted(read_tools), sorted(write_tools)

    def _ask_next(
        self,
        stage: str,
        missing: list[str],
        observed_facts: list[str],
        last_action: str,
    ) -> str:
        if missing:
            return f"Ask for or verify: {missing[0]}."
        if stage == "inspect_state_with_read_tools":
            return "Use the narrowest read tool needed to inspect the official current state."
        if stage == "propose_action_and_confirm":
            return "Briefly summarize the intended action and consequence, then ask for explicit confirmation."
        if stage == "execute_write_action":
            return "Call the exact confirmed write tool with complete arguments."
        if observed_facts:
            return (
                "Continue with the next policy-required step using only verified facts."
            )
        if last_action:
            return "Use the last action result to choose the next policy-required step."
        return "Ask one concise clarification question."

    def _do_not(self, stage: str) -> list[str]:
        if stage != "execute_write_action":
            return [
                "Do not modify account, order, reservation, plan, or service state yet."
            ]
        return ["Do not invent tool results or hidden benchmark facts."]

    def _exit_condition(self, stage: str) -> str:
        if stage == "execute_write_action":
            return "The confirmed write tool has completed or returned an error."
        if stage == "verify_result_and_close":
            return (
                "The customer understands the outcome and no further action is needed."
            )
        return "The required visible facts for this stage are collected or verified."
