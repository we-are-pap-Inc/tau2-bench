"""StageGate controller shared by the OpenAI audio-native agent and orchestrator."""

import os
import time
from typing import Optional

from loguru import logger

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.simulation import SimulationRun
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.stagegate.orchestrator import (
    StagePacketOrchestrator,
)
from tau2.voice.audio_native.openai.stagegate.stage_schema import (
    StageGateCondition,
    TraceEvent,
)
from tau2.voice.audio_native.openai.stagegate.trace import (
    JsonlTraceWriter,
    get_trace_run_id,
)

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
        self.task_id: Optional[str] = None
        self.sim_id: Optional[str] = None
        self.trial: Optional[int] = None

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

    @property
    def tracing_enabled(self) -> bool:
        """Whether JSONL trace writing is active."""
        return self.trace_writer.enabled

    def set_domain_name(self, domain_name: str) -> None:
        """Attach the public domain name for trace context."""
        self.domain_name = domain_name

    def set_trace_context(
        self,
        *,
        domain_name: Optional[str] = None,
        task_id: Optional[str] = None,
        sim_id: Optional[str] = None,
        trial: Optional[int] = None,
    ) -> None:
        """Attach run context used on subsequent trace events."""
        if domain_name is not None:
            self.domain_name = domain_name
        if task_id is not None:
            self.task_id = task_id
        if sim_id is not None:
            self.sim_id = sim_id
        if trial is not None:
            self.trial = trial

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
        start = time.perf_counter()
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
            tick_index=tick_id,
            stage=current_stage,
            source="advance_stage_call",
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
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
            tick_index=tick_id,
            stage=packet.stage,
            source="stage_packet_returned",
            tool_name=tool_call.name,
            latency_ms=self._elapsed_ms(start),
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

    def trace_run_start(self) -> None:
        """Emit run_start with public run context."""
        self._trace(
            "run_start",
            visible_to_agent=False,
            source="orchestrator",
        )

    def trace_run_end(
        self,
        *,
        termination_reason: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Emit run_end before posthoc evaluation output is available."""
        payload: dict[str, object] = {}
        if termination_reason is not None:
            payload["termination_reason"] = termination_reason
        if duration_seconds is not None:
            payload["duration_seconds"] = duration_seconds
        self._trace(
            "run_end",
            visible_to_agent=False,
            source="orchestrator",
            payload=payload,
        )

    def trace_model_function_call(
        self, tool_call: ToolCall, *, tick_id: Optional[int] = None
    ) -> None:
        """Emit a model_function_call event for an agent-visible function call."""
        self._trace(
            "model_function_call",
            tick_index=tick_id,
            source="model_function_call",
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
            payload={"tool_call_id": tool_call.id},
        )

    def trace_domain_tool_call(
        self, tool_call: ToolCall, *, tick_id: Optional[int] = None
    ) -> None:
        """Emit a domain_tool_call event before executing a domain tool."""
        self._trace(
            "domain_tool_call",
            tick_index=tick_id,
            source="domain_tool_call",
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
            payload={"tool_call_id": tool_call.id},
        )

    def trace_domain_tool_result(
        self,
        tool_call: ToolCall,
        tool_result: ToolMessage,
        *,
        tick_id: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Emit a domain_tool_result event after official tool execution."""
        self._trace(
            "domain_tool_result",
            tick_index=tick_id,
            source="domain_tool_result",
            tool_name=tool_call.name,
            latency_ms=latency_ms,
            payload={
                "tool_call_id": tool_call.id,
                "tool_result_id": tool_result.id,
                "tool_result": tool_result.content,
                "tool_error": tool_result.error,
            },
        )

    def trace_final_outcome(self, simulation: SimulationRun) -> None:
        """Emit posthoc evaluator outcome only after evaluation has completed."""
        if simulation.reward_info is None:
            reward = None
            passed = None
            reward_breakdown = None
        else:
            reward = simulation.reward_info.reward
            passed = reward == 1.0
            reward_breakdown = simulation.reward_info.reward_breakdown
        self._trace(
            "final_outcome",
            visible_to_agent=False,
            source="evaluator",
            leakage_risk="posthoc_evaluator",
            reward=reward,
            passed=passed,
            payload={
                "termination_reason": simulation.termination_reason,
                "reward_breakdown": reward_breakdown,
            },
        )

    def _trace(
        self,
        event_type: str,
        *,
        stage: Optional[str] = None,
        turn_index: Optional[int] = None,
        tick_index: Optional[int] = None,
        visible_to_agent: bool = True,
        source: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        latency_ms: Optional[float] = None,
        leakage_risk: str = "none",
        reward: Optional[float] = None,
        passed: Optional[bool] = None,
        failure_type: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        self.trace_writer.write(
            TraceEvent(
                event_type=event_type,
                condition=self.condition,
                run_id=get_trace_run_id(sim_id=self.sim_id),
                domain=self.domain_name,
                task_id=self.task_id,
                sim_id=self.sim_id,
                trial=self.trial,
                stage=stage,
                turn_index=turn_index,
                tick_index=tick_index,
                visible_to_agent=visible_to_agent,
                source=source,
                tool_name=tool_name,
                tool_args=tool_args,
                latency_ms=latency_ms,
                leakage_risk=leakage_risk,
                reward=reward,
                passed=passed,
                failure_type=failure_type,
                payload=payload or {},
            )
        )

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)
