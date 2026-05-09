"""StageGate controller shared by the OpenAI audio-native agent and orchestrator."""

import os
from typing import Optional

from loguru import logger

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.stagegate.orchestrator import (
    StagePacketOrchestrator,
)
from tau2.voice.audio_native.openai.stagegate.stage_schema import (
    StageGateCondition,
    TraceEvent,
)
from tau2.voice.audio_native.openai.stagegate.trace import JsonlTraceWriter

CONDITION_ENV_VAR = "TAU2_STAGEGATE_CONDITION"
ADVANCE_STAGE_TOOL_NAME = "advance_stage"
VALID_CONDITIONS = {"baseline", "stage_only", "stagegate"}

STAGEGATE_PROMPT_ADDITION = """
StageGate operating rules:
- Do not reveal internal stages, packets, benchmark internals, or orchestration.
- When the next procedural step is unclear, call advance_stage.
- Before changing account, order, reservation, plan, or service state, make sure policy prerequisites and confirmation requirements are satisfied.
""".strip()


def advance_stage(
    current_stage: str,
    observed_facts: list[str],
    last_action: str,
    blocker: Optional[str] = None,
) -> str:
    """Request the next stage-specific operating instructions from the server-side orchestrator.

    This tool does not modify customer records or domain state.

    Args:
        current_stage: The stage the assistant believes it is currently in.
        observed_facts: Visible facts from conversation or official tool results.
        last_action: The last assistant action or official tool result summary.
        blocker: Optional reason the assistant cannot proceed.

    Returns:
        A compact StageGate stage packet.
    """
    return ""


class StageGateController:
    """Coordinates StageOnly and StageGate behavior."""

    def __init__(
        self,
        *,
        condition: StageGateCondition,
        domain_policy: str,
        tools: list[Tool],
        domain_name: Optional[str] = None,
        trace_writer: Optional[JsonlTraceWriter] = None,
    ):
        self.condition = condition
        self.domain_policy = domain_policy
        self.domain_name = domain_name
        self.tools = list(tools)
        self.advance_stage_tool = Tool(advance_stage)
        self.packet_orchestrator = StagePacketOrchestrator()
        self.trace_writer = trace_writer or JsonlTraceWriter.from_env()

    @classmethod
    def from_env(
        cls,
        *,
        provider: str,
        domain_policy: str,
        tools: list[Tool],
    ) -> "StageGateController":
        """Build a controller from TAU2_STAGEGATE_CONDITION."""
        raw_condition = os.environ.get(CONDITION_ENV_VAR, "baseline").strip().lower()
        condition = raw_condition or "baseline"
        if condition not in VALID_CONDITIONS:
            raise ValueError(
                f"Invalid {CONDITION_ENV_VAR}={raw_condition!r}. "
                f"Expected one of {sorted(VALID_CONDITIONS)}."
            )
        if provider != "openai" and condition != "baseline":
            logger.warning(
                f"StageGate condition {condition!r} requested for provider "
                f"{provider!r}; StageGate is only active for provider='openai'."
            )
            condition = "baseline"
        return cls(condition=condition, domain_policy=domain_policy, tools=tools)

    @property
    def enabled(self) -> bool:
        """Whether any StageGate behavior is active."""
        return self.condition in {"stage_only", "stagegate"}

    def set_domain_name(self, domain_name: str) -> None:
        """Attach the public domain name for trace context."""
        self.domain_name = domain_name

    def prompt_addendum(self) -> str:
        """Return extra prompt text for active StageGate conditions."""
        if not self.enabled:
            return ""
        return STAGEGATE_PROMPT_ADDITION

    def session_tools(self, tools: list[Tool]) -> list[Tool]:
        """Return session tools, adding advance_stage only when active."""
        if not self.enabled:
            return tools
        if any(tool.name == ADVANCE_STAGE_TOOL_NAME for tool in tools):
            return tools
        return list(tools) + [self.advance_stage_tool]

    def is_advance_stage(self, tool_call: ToolCall) -> bool:
        """Whether this call targets the StageGate orchestration tool."""
        return tool_call.name == ADVANCE_STAGE_TOOL_NAME

    def handle_advance_stage(
        self, tool_call: ToolCall, *, tick_id: Optional[int] = None
    ) -> ToolMessage:
        """Return a stage packet for an advance_stage call."""
        args = tool_call.arguments
        current_stage = str(args.get("current_stage", "understand_intent"))
        observed_facts = args.get("observed_facts", [])
        if not isinstance(observed_facts, list):
            observed_facts = [str(observed_facts)]
        observed_facts = [str(fact) for fact in observed_facts]
        last_action = str(args.get("last_action", ""))
        blocker = args.get("blocker")
        blocker_text = str(blocker) if blocker is not None else None

        self._trace(
            "advance_stage_call",
            tick_id=tick_id,
            payload={
                "tool_call_id": tool_call.id,
                "current_stage": current_stage,
                "last_action": last_action,
                "blocker": blocker_text,
            },
        )

        packet = self.packet_orchestrator.build_packet(
            current_stage=current_stage,
            observed_facts=observed_facts,
            last_action=last_action,
            blocker=blocker_text,
            tools=self.session_tools(self.tools),
        )
        self._trace(
            "stage_packet_returned",
            tick_id=tick_id,
            payload={
                "tool_call_id": tool_call.id,
                "packet": packet.model_dump(mode="json"),
            },
        )
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            requestor=tool_call.requestor,
            content=packet.model_dump_json(),
            error=False,
        )

    def _trace(
        self,
        event_type: str,
        *,
        tick_id: Optional[int] = None,
        payload: Optional[dict] = None,
    ) -> None:
        self.trace_writer.write(
            TraceEvent(
                event_type=event_type,
                condition=self.condition,
                domain=self.domain_name,
                tick_id=tick_id,
                payload=payload or {},
            )
        )
