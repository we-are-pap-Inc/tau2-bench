"""StageGate controller shared by the OpenAI audio-native agent and orchestrator."""

import os
import time
from typing import Optional

from loguru import logger

from tau2.data_model.message import Message, ToolCall, ToolMessage
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.stagegate.ledger import EntityLedger
from tau2.voice.audio_native.openai.stagegate.orchestrator import (
    StagePacketOrchestrator,
)
from tau2.voice.audio_native.openai.stagegate.stage_schema import (
    EvidenceSource,
    StageGateCondition,
    TraceEvent,
)
from tau2.voice.audio_native.openai.stagegate.trace import (
    JsonlTraceWriter,
    get_trace_run_id,
)
from tau2.voice.audio_native.openai.stagegate.validator import (
    PreWriteValidator,
    ValidatorDecision,
)

CONDITION_ENV_VAR = "TAU2_STAGEGATE_CONDITION"
ADVANCE_STAGE_TOOL_NAME = "advance_stage"
VALID_CONDITIONS = {"baseline", "stage_only", "stagegate"}
MAX_ADVANCE_STAGE_CALLS_ENV_VAR = "TAU2_STAGEGATE_MAX_ADVANCE_STAGE_CALLS"
MAX_REPEATED_STAGE_ENV_VAR = "TAU2_STAGEGATE_MAX_REPEATED_STAGE"
MAX_REPEATED_BLOCKER_ENV_VAR = "TAU2_STAGEGATE_MAX_REPEATED_BLOCKER"
DEFAULT_MAX_ADVANCE_STAGE_CALLS_PER_SIM = 12
DEFAULT_MAX_REPEATED_SAME_STAGE = 3
DEFAULT_MAX_REPEATED_SAME_BLOCKER = 2

STAGEGATE_PROMPT_ADDITION = """
StageGate operating rules:
- Do not reveal internal stages, packets, benchmark internals, or orchestration.
- Once the user's request can be summarized from visible conversation or official tool results, call advance_stage before the first customer-specific domain tool call.
- In advance_stage observed_facts, include only facts visible in the conversation, model tool arguments, or official tool results.
- Follow each returned stage packet. Call advance_stage again only after its exit condition is met or a blocker appears.
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
        max_advance_stage_calls_per_sim: Optional[int] = None,
        max_repeated_same_stage: Optional[int] = None,
        max_repeated_same_blocker: Optional[int] = None,
    ):
        self.condition = condition
        self.domain_policy = domain_policy
        self.domain_name = domain_name
        self.tools = list(tools)
        self.advance_stage_tool = Tool(advance_stage)
        self.packet_orchestrator = StagePacketOrchestrator()
        self.trace_writer = trace_writer or JsonlTraceWriter.from_env()
        self.benchmark_task_id: Optional[str] = None
        self.sim_id: Optional[str] = None
        self.trial: Optional[int] = None
        self.max_advance_stage_calls_per_sim = _resolve_int_setting(
            explicit=max_advance_stage_calls_per_sim,
            env_var=MAX_ADVANCE_STAGE_CALLS_ENV_VAR,
            default=DEFAULT_MAX_ADVANCE_STAGE_CALLS_PER_SIM,
        )
        self.max_repeated_same_stage = _resolve_int_setting(
            explicit=max_repeated_same_stage,
            env_var=MAX_REPEATED_STAGE_ENV_VAR,
            default=DEFAULT_MAX_REPEATED_SAME_STAGE,
        )
        self.max_repeated_same_blocker = _resolve_int_setting(
            explicit=max_repeated_same_blocker,
            env_var=MAX_REPEATED_BLOCKER_ENV_VAR,
            default=DEFAULT_MAX_REPEATED_SAME_BLOCKER,
        )
        self.advance_stage_call_count = 0
        self.stage_sequence: list[str] = []
        self.repeated_stage_count = 0
        self.repeated_blocker_count = 0
        self._last_stage: Optional[str] = None
        self._current_repeated_stage_count = 0
        self._last_blocker_pattern: Optional[tuple[str, ...]] = None
        self._current_repeated_blocker_count = 0
        self.loop_guard_triggered = False
        self.validator_block_count = 0
        self.validator_allow_count = 0
        self.ledger_update_count = 0
        self.final_stage: Optional[str] = None
        self.last_stage_packet: Optional[dict[str, object]] = None
        self.last_validator_decision: Optional[dict[str, object]] = None
        self.last_corrective_packet: Optional[dict[str, object]] = None
        self.last_blocked_side_effecting_tool: Optional[dict[str, object]] = None
        self.last_successful_side_effecting_tool: Optional[dict[str, object]] = None
        if self.condition == "stagegate":
            self.ledger = EntityLedger.for_domain(domain_name)
            self.validator = PreWriteValidator(
                domain_name=domain_name,
                tools=self.tools,
                domain_policy=domain_policy,
            )

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
        self._set_ledger_domain(domain_name)
        self._set_validator_domain(domain_name)

    def set_trace_context(
        self,
        *,
        domain_name: Optional[str] = None,
        benchmark_task_id: Optional[str] = None,
        sim_id: Optional[str] = None,
        trial: Optional[int] = None,
    ) -> None:
        """Attach run context used on subsequent trace events."""
        if domain_name is not None:
            self.domain_name = domain_name
            self._set_ledger_domain(domain_name)
            self._set_validator_domain(domain_name)
        if benchmark_task_id is not None:
            self.benchmark_task_id = benchmark_task_id
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
        forced_stage = self._forced_stage_for_advance(
            current_stage=current_stage,
            blocker=blocker_text,
        )

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
            domain_name=self.domain_name,
            ledger=self._active_ledger(),
            forced_stage=forced_stage,
        )
        guard_reason = self._record_advance_stage_guard_state(
            packet=packet,
            blocker=blocker_text,
        )
        if guard_reason is not None:
            packet = self.packet_orchestrator.build_fallback_packet(
                packet=packet,
                guard_reason=guard_reason,
            )
            self.loop_guard_triggered = True
            self._remember_stage_packet(packet)
            self.last_corrective_packet = packet.model_dump(mode="json")
            self._trace(
                "stage_loop_guard_triggered",
                tick_index=tick_id,
                stage=packet.stage,
                source="advance_stage_guard",
                tool_name=tool_call.name,
                tool_args=tool_call.arguments,
                payload={
                    "tool_call_id": tool_call.id,
                    "guard_reason": guard_reason,
                    "summary": self.trace_summary_payload(),
                    "fallback_packet": packet.model_dump(mode="json"),
                },
            )
        else:
            self._remember_stage_packet(packet)
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
        summary = self.trace_summary_payload()
        self._trace(
            "trace_summary",
            visible_to_agent=False,
            source="stagegate_summary",
            payload=summary,
        )
        payload: dict[str, object] = dict(summary)
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
            source=EvidenceSource.MODEL_TOOL_ARGUMENT.value,
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
            payload={"tool_call_id": tool_call.id},
        )
        ledger = self._active_ledger()
        if ledger is None:
            return
        deltas = ledger.update_from_tool_args(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            event_id=tool_call.id,
            tick_index=tick_id,
        )
        self._trace_ledger_updates(
            deltas,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            tick_id=tick_id,
        )

    def trace_domain_tool_call(
        self, tool_call: ToolCall, *, tick_id: Optional[int] = None
    ) -> None:
        """Emit a domain_tool_call event before executing a domain tool."""
        self._trace(
            "domain_tool_call",
            tick_index=tick_id,
            source=EvidenceSource.MODEL_TOOL_ARGUMENT.value,
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
            source=EvidenceSource.DOMAIN_TOOL_OUTPUT.value,
            tool_name=tool_call.name,
            latency_ms=latency_ms,
            payload={
                "tool_call_id": tool_call.id,
                "tool_result_id": tool_result.id,
                "tool_result": tool_result.content,
                "tool_error": tool_result.error,
            },
        )
        validator = self._active_validator()
        if (
            validator is not None
            and not tool_result.error
            and validator.is_side_effecting_tool(tool_call.name)
        ):
            self.last_successful_side_effecting_tool = {
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "tick_index": tick_id,
            }
            self.last_blocked_side_effecting_tool = None
        ledger = self._active_ledger()
        if ledger is None or tool_result.error:
            return
        deltas = ledger.update_from_tool_result(
            tool_name=tool_call.name,
            content=tool_result.content,
            event_id=tool_result.id,
            tick_index=tick_id,
        )
        self._trace_ledger_updates(
            deltas,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            tick_id=tick_id,
        )
        if validator is not None:
            validator.record_tool_result(
                tool_call=tool_call,
                tool_result=tool_result,
                tick_index=tick_id,
            )

    def record_visible_message(
        self,
        message: Message,
        *,
        is_agent: bool,
        tick_id: Optional[int] = None,
    ) -> None:
        """Record legacy assistant text and reject audio-native user chunk text."""
        if is_agent:
            self.record_assistant_utterance(message, tick_id=tick_id)
            return
        raise ValueError(
            "Audio-native UserMessage.content is simulator gold text and cannot "
            "be used as StageGate runtime evidence."
        )

    def record_assistant_utterance(
        self,
        message: Message,
        *,
        tick_id: Optional[int] = None,
    ) -> None:
        """Record assistant utterance text from model output."""
        content = getattr(message, "content", None)
        if content:
            self._trace(
                "assistant_audio_event",
                tick_index=tick_id,
                source=EvidenceSource.ASSISTANT_UTTERANCE.value,
                payload={"content": content},
            )
        validator = self._active_validator()
        if validator is None:
            return
        validator.record_assistant_utterance(
            content=content,
            tick_index=tick_id,
            source=EvidenceSource.ASSISTANT_UTTERANCE,
        )

    def record_agent_visible_user_transcript(
        self,
        transcript: str,
        *,
        tick_id: Optional[int] = None,
    ) -> None:
        """Record user transcript only when the adapter exposes it to the model path."""
        if transcript:
            self._trace(
                "user_transcript_event",
                tick_index=tick_id,
                source=EvidenceSource.AGENT_VISIBLE_TRANSCRIPT.value,
                payload={"transcript": transcript},
            )
        validator = self._active_validator()
        if validator is None:
            return
        validator.record_user_confirmation_evidence(
            content=transcript,
            tick_index=tick_id,
            source=EvidenceSource.AGENT_VISIBLE_TRANSCRIPT,
        )

    def validate_tool_call(
        self,
        tool_call: ToolCall,
        *,
        tick_id: Optional[int] = None,
    ) -> ValidatorDecision:
        """Validate a tool call and emit validator trace rows when active."""
        validator = self._active_validator()
        if validator is None:
            return ValidatorDecision(
                decision="allow",
                reason="validator_inactive",
                checks={},
            )

        start = time.perf_counter()
        decision = validator.validate(tool_call, ledger=self._active_ledger())
        payload = {
            "tool_call_id": tool_call.id,
            "checks": decision.checks,
        }
        if decision.corrective_packet is not None:
            payload["corrective_packet"] = decision.corrective_packet.model_dump(
                mode="json"
            )
            self.last_corrective_packet = payload["corrective_packet"]

        if decision.allowed:
            self.validator_allow_count += 1
        else:
            self.validator_block_count += 1
            if decision.checks.get(
                "side_effecting_tool"
            ) is True or validator.is_side_effecting_tool(tool_call.name):
                self.last_blocked_side_effecting_tool = {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "tick_index": tick_id,
                    "reason": decision.reason,
                }
        self.last_validator_decision = {
            "tool_name": tool_call.name,
            "decision": decision.decision,
            "reason": decision.reason,
            "checks": decision.checks,
        }

        self._trace(
            "validator_check",
            tick_index=tick_id,
            source="validator",
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
            validator_decision=decision.decision,
            validator_reason=decision.reason,
            payload=payload,
        )
        self._trace(
            "validator_allow" if decision.allowed else "validator_block",
            tick_index=tick_id,
            source="validator",
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
            validator_decision=decision.decision,
            validator_reason=decision.reason,
            latency_ms=self._elapsed_ms(start),
            payload=payload,
        )
        return decision

    def blocked_tool_message(
        self,
        tool_call: ToolCall,
        decision: ValidatorDecision,
    ) -> ToolMessage:
        """Convert a block decision to the tool result returned to the model."""
        if decision.corrective_packet is None:
            content = decision.model_dump_json()
        else:
            content = decision.corrective_packet.model_dump_json()
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            requestor=tool_call.requestor,
            content=content,
            error=True,
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
        ledger_delta: Optional[dict[str, object]] = None,
        validator_decision: Optional[str] = None,
        validator_reason: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        self.trace_writer.write(
            TraceEvent(
                event_type=event_type,
                condition=self.condition,
                run_id=get_trace_run_id(sim_id=self.sim_id),
                domain=self.domain_name,
                benchmark_task_id=self.benchmark_task_id,
                sim_id=self.sim_id,
                trial=self.trial,
                stage=stage,
                turn_index=turn_index,
                tick_index=tick_index,
                visible_to_agent=visible_to_agent,
                source=source,
                tool_name=tool_name,
                tool_args=tool_args,
                ledger_delta=ledger_delta,
                validator_decision=validator_decision,
                validator_reason=validator_reason,
                latency_ms=latency_ms,
                leakage_risk=leakage_risk,
                payload=payload or {},
            )
        )

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)

    def _active_ledger(self) -> Optional[EntityLedger]:
        if self.condition != "stagegate":
            return None
        return getattr(self, "ledger", None)

    def _active_validator(self) -> Optional[PreWriteValidator]:
        if self.condition != "stagegate":
            return None
        return getattr(self, "validator", None)

    def trace_summary_payload(self) -> dict[str, object]:
        """Return trace-only StageGate triage counters for the current run."""
        return {
            "advance_stage_call_count": self.advance_stage_call_count,
            "repeated_stage_count": self.repeated_stage_count,
            "repeated_blocker_count": self.repeated_blocker_count,
            "loop_guard_triggered": self.loop_guard_triggered,
            "validator_block_count": self.validator_block_count,
            "validator_allow_count": self.validator_allow_count,
            "ledger_update_count": self.ledger_update_count,
            "stage_sequence": list(self.stage_sequence),
            "final_stage": self.final_stage,
            "last_stage_packet": self.last_stage_packet,
            "last_validator_decision": self.last_validator_decision,
            "last_corrective_packet": self.last_corrective_packet,
            "last_blocked_side_effecting_tool": self.last_blocked_side_effecting_tool,
            "last_successful_side_effecting_tool": (
                self.last_successful_side_effecting_tool
            ),
        }

    def _set_ledger_domain(self, domain_name: Optional[str]) -> None:
        if self.condition != "stagegate":
            return
        normalized_domain = (domain_name or "").strip().lower() or None
        ledger = getattr(self, "ledger", None)
        if ledger is None or ledger.domain_name != normalized_domain:
            self.ledger = EntityLedger.for_domain(normalized_domain)

    def _set_validator_domain(self, domain_name: Optional[str]) -> None:
        validator = self._active_validator()
        if validator is not None:
            validator.set_domain_name(domain_name)

    def _trace_ledger_updates(
        self,
        deltas: list[dict[str, object]],
        *,
        tool_call_id: str,
        tool_name: str,
        tick_id: Optional[int],
    ) -> None:
        for delta in deltas:
            self.ledger_update_count += 1
            self._trace(
                "ledger_update",
                tick_index=tick_id,
                source=str(delta.get("source", "ledger")),
                tool_name=tool_name,
                ledger_delta={str(delta["field"]): delta},
                payload={"tool_call_id": tool_call_id},
            )

    def _record_advance_stage_guard_state(
        self,
        *,
        packet,
        blocker: Optional[str],
    ) -> Optional[str]:
        self.advance_stage_call_count += 1
        self._record_stage_streak(packet.stage)
        self._record_blocker_streak(blocker=blocker, missing_facts=packet.missing_facts)
        self.stage_sequence.append(packet.stage)
        self.final_stage = packet.stage

        if self.advance_stage_call_count > self.max_advance_stage_calls_per_sim:
            return "max_advance_stage_calls"
        if self._current_repeated_stage_count > self.max_repeated_same_stage:
            return "repeated_same_stage"
        if self._current_repeated_blocker_count > self.max_repeated_same_blocker:
            return "repeated_same_blocker"
        return None

    def _record_stage_streak(self, stage: str) -> None:
        if stage == self._last_stage:
            self._current_repeated_stage_count += 1
        else:
            self._last_stage = stage
            self._current_repeated_stage_count = 1
        self.repeated_stage_count = max(
            self.repeated_stage_count,
            self._current_repeated_stage_count,
        )

    def _record_blocker_streak(
        self,
        *,
        blocker: Optional[str],
        missing_facts: list[str],
    ) -> None:
        pattern = self._blocker_pattern(blocker=blocker, missing_facts=missing_facts)
        if pattern is None:
            self._last_blocker_pattern = None
            self._current_repeated_blocker_count = 0
            return
        if pattern == self._last_blocker_pattern:
            self._current_repeated_blocker_count += 1
        else:
            self._last_blocker_pattern = pattern
            self._current_repeated_blocker_count = 1
        self.repeated_blocker_count = max(
            self.repeated_blocker_count,
            self._current_repeated_blocker_count,
        )

    def _blocker_pattern(
        self,
        *,
        blocker: Optional[str],
        missing_facts: list[str],
    ) -> Optional[tuple[str, ...]]:
        values = [blocker or "", *missing_facts]
        normalized = tuple(
            " ".join(value.strip().lower().split())
            for value in values
            if value and value.strip()
        )
        return normalized or None

    def _remember_stage_packet(self, packet) -> None:
        self.final_stage = packet.stage
        if self.stage_sequence:
            self.stage_sequence[-1] = packet.stage
        else:
            self.stage_sequence.append(packet.stage)
        self.last_stage_packet = packet.model_dump(mode="json")

    def _forced_stage_for_advance(
        self,
        *,
        current_stage: str,
        blocker: Optional[str],
    ) -> Optional[str]:
        if self.condition != "stagegate":
            return None
        if blocker is not None:
            return None
        if current_stage not in {"execute_write_action", "verify_result_and_close"}:
            return None
        if self.last_successful_side_effecting_tool is not None:
            return None
        return "execute_write_action"


def _resolve_int_setting(
    *,
    explicit: Optional[int],
    env_var: str,
    default: int,
) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    raw_value = os.environ.get(env_var)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning(f"Invalid {env_var}={raw_value!r}; using default {default}.")
        return default
