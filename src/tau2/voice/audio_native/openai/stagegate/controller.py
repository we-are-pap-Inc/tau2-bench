"""StageGate controller shared by the OpenAI audio-native agent and orchestrator."""

import json
import os
import time
from typing import Literal, Optional

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
    pending_write_allowed_internal_tools,
    pending_write_disallowed_tools,
    pending_write_next_required_steps,
)

CONDITION_ENV_VAR = "TAU2_STAGEGATE_CONDITION"
ADVANCE_STAGE_TOOL_NAME = "advance_stage"
RECORD_PENDING_WRITE_SUMMARY_TOOL_NAME = "record_pending_write_summary"
RECORD_PENDING_WRITE_CONFIRMATION_TOOL_NAME = "record_pending_write_confirmation"
PENDING_WRITE_TOOL_NAMES = {
    RECORD_PENDING_WRITE_SUMMARY_TOOL_NAME,
    RECORD_PENDING_WRITE_CONFIRMATION_TOOL_NAME,
}
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
- In StageGate mode, a blocked write creates one active pending write. After you verbally summarize that exact pending write and ask for confirmation, call record_pending_write_summary; no pending_write_id is needed. After the user's next response, call record_pending_write_confirmation; no pending_write_id is needed. If confirmed, retry the same original domain write tool directly with unchanged arguments.
- Do not transfer to a human agent while an active pending write can still be completed through the structured pending-write tools.
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


def record_pending_write_summary(
    summary_presented: bool,
    action_type: str,
    consequence_presented: bool,
    confirmation_requested: bool,
    notes: Optional[str] = None,
    pending_write_id: Optional[str] = None,
) -> str:
    """Record that the assistant summarized a pending write and asked confirmation.

    This StageGate-only orchestration tool does not modify domain state.
    This applies to the active pending write. pending_write_id is not required.
    Use record_pending_write_summary after you have told the user the pending
    action and consequence and asked for confirmation. If the user later
    confirms, record that response with record_pending_write_confirmation and
    retry the same original write tool directly.

    Args:
        summary_presented: Whether the assistant presented the pending write summary.
        action_type: The structured action type the assistant summarized.
        consequence_presented: Whether the consequence was included.
        confirmation_requested: Whether explicit user confirmation was requested.
        notes: Optional brief model-visible note.
        pending_write_id: Optional trace/debug handle for the active pending write.

    Returns:
        A structured StageGate pending-write protocol result.
    """
    return ""


def record_pending_write_confirmation(
    decision: Literal["confirmed", "denied", "unclear"],
    basis: Literal[
        "latest_user_turn",
        "user_corrected_details",
        "user_declined",
        "unclear_response",
    ],
    notes: Optional[str] = None,
    pending_write_id: Optional[str] = None,
) -> str:
    """Record the assistant's structured decision after the user's response.

    This StageGate-only orchestration tool does not modify domain state.
    This applies to the active pending write. pending_write_id is not required.
    Use record_pending_write_confirmation after the user responds to that
    confirmation request. If confirmed, retry the same original write tool
    directly.

    Args:
        decision: Whether the user confirmed, denied, or gave an unclear response.
        basis: The event-order basis for the decision.
        notes: Optional brief model-visible note.
        pending_write_id: Optional trace/debug handle for the active pending write.

    Returns:
        A structured StageGate pending-write protocol result.
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
        self.pending_write_summary_tool = Tool(record_pending_write_summary)
        self.pending_write_confirmation_tool = Tool(record_pending_write_confirmation)
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
        """Return session tools, adding StageGate tools only when active."""
        if not self.enabled:
            return tools
        session_tools = list(tools)
        existing_names = {tool.name for tool in session_tools}
        if ADVANCE_STAGE_TOOL_NAME not in existing_names:
            session_tools.append(self.advance_stage_tool)
            existing_names.add(ADVANCE_STAGE_TOOL_NAME)
        if self.condition == "stagegate":
            for internal_tool in (
                self.pending_write_summary_tool,
                self.pending_write_confirmation_tool,
            ):
                if internal_tool.name not in existing_names:
                    session_tools.append(internal_tool)
                    existing_names.add(internal_tool.name)
        return session_tools

    def is_advance_stage(self, tool_call: ToolCall) -> bool:
        """Whether this call targets the StageGate orchestration tool."""
        return tool_call.name == ADVANCE_STAGE_TOOL_NAME

    def is_pending_write_tool(self, tool_call: ToolCall) -> bool:
        """Whether this call targets a StageGate pending-write protocol tool."""
        return (
            self.condition == "stagegate" and tool_call.name in PENDING_WRITE_TOOL_NAMES
        )

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
        packet = self._apply_pending_write_guidance(packet)
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

    def handle_pending_write_tool(
        self,
        tool_call: ToolCall,
        *,
        tick_id: Optional[int] = None,
    ) -> ToolMessage:
        """Execute a StageGate-only pending-write protocol tool."""
        validator = self._active_validator()
        if validator is None:
            return ToolMessage(
                id=tool_call.id,
                role="tool",
                requestor=tool_call.requestor,
                content='{"ok": false, "reason": "validator_inactive"}',
                error=True,
            )
        args = tool_call.arguments
        if tool_call.name == RECORD_PENDING_WRITE_SUMMARY_TOOL_NAME:
            result = validator.record_pending_write_summary(
                pending_write_id=(
                    None
                    if args.get("pending_write_id") is None
                    else str(args.get("pending_write_id"))
                ),
                summary_presented=bool(args.get("summary_presented", False)),
                action_type=str(args.get("action_type", "")),
                consequence_presented=bool(args.get("consequence_presented", False)),
                confirmation_requested=bool(args.get("confirmation_requested", False)),
                notes=None if args.get("notes") is None else str(args.get("notes")),
                tick_index=tick_id,
            )
        elif tool_call.name == RECORD_PENDING_WRITE_CONFIRMATION_TOOL_NAME:
            result = validator.record_pending_write_confirmation(
                pending_write_id=(
                    None
                    if args.get("pending_write_id") is None
                    else str(args.get("pending_write_id"))
                ),
                decision=str(args.get("decision", "unclear")),
                basis=str(args.get("basis", "unclear_response")),
                notes=None if args.get("notes") is None else str(args.get("notes")),
                tick_index=tick_id,
            )
        else:
            result = None

        self._trace_pending_write_events(validator.drain_pending_write_events())
        if result is None:
            content = '{"ok": false, "reason": "unknown_pending_write_tool"}'
            error = True
        else:
            content = result.model_dump_json()
            error = not result.ok
        self._trace(
            "pending_write_tool_result",
            tick_index=tick_id,
            source="stagegate_pending_write_tool",
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
            payload=json.loads(content),
        )
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            requestor=tool_call.requestor,
            content=content,
            error=error,
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
            validator.mark_side_effecting_write_consumed(
                tool_call=tool_call,
                tick_index=tick_id,
            )
            self._trace_pending_write_events(validator.drain_pending_write_events())
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
        self._trace_pending_write_events(validator.drain_pending_write_events())

    def record_agent_visible_user_transcript(
        self,
        transcript: str,
        *,
        tick_id: Optional[int] = None,
    ) -> None:
        """Record agent-visible user-turn ordering without parsing transcript text."""
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
        validator.record_user_turn(
            content=transcript,
            tick_index=tick_id,
            source=EvidenceSource.AGENT_VISIBLE_TRANSCRIPT,
        )
        self._trace_pending_write_events(validator.drain_pending_write_events())

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
        decision = validator.validate(
            tool_call,
            ledger=self._active_ledger(),
            tick_index=tick_id,
            current_stage=self.final_stage,
        )
        payload = {
            "tool_call_id": tool_call.id,
            "checks": decision.checks,
        }
        if decision.pending_write_id is not None:
            payload["pending_write_id"] = decision.pending_write_id
            payload["args_fingerprint"] = decision.args_fingerprint
            payload["matched_facets"] = decision.matched_facets
            payload["missing_facets"] = decision.missing_facets
        self._trace_pending_write_events(validator.drain_pending_write_events())
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
            "pending_write_id": decision.pending_write_id,
            "args_fingerprint": decision.args_fingerprint,
            "matched_facets": decision.matched_facets,
            "missing_facets": decision.missing_facets,
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
            "pending_write": self._pending_write_snapshot(),
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

    def _trace_pending_write_events(self, events: list[dict[str, object]]) -> None:
        for event in events:
            event_type = str(event.get("event_type", "pending_write_event"))
            tick_index = event.get("tick_index")
            self._trace(
                event_type,
                tick_index=tick_index if isinstance(tick_index, int) else None,
                source="validator",
                tool_name=str(event.get("tool_name", "")) or None,
                payload=event,
            )

    def _pending_write_snapshot(self) -> Optional[dict[str, object]]:
        validator = self._active_validator()
        if validator is None:
            return None
        return validator.pending_write_snapshot()

    def _active_pending_write_snapshot(self) -> Optional[dict[str, object]]:
        snapshot = self._pending_write_snapshot()
        if snapshot is None:
            return None
        if snapshot.get("status") in {"consumed", "expired", "none"}:
            return None
        return snapshot

    def _apply_pending_write_guidance(self, packet):
        pending_write = self._active_pending_write_snapshot()
        if pending_write is None:
            return packet
        status = str(pending_write.get("status", ""))
        tool_name = str(pending_write.get("tool_name", "the write tool"))
        if status in {"needs_summary", "mismatched_retry"}:
            return packet.model_copy(
                update={
                    "stage": "propose_action_and_confirm",
                    "missing_facts": [
                        "pending_write_summary",
                        "pending_write_confirmation",
                    ],
                    "ask_next": (
                        "State the pending action and consequence, ask for "
                        "explicit confirmation, then call "
                        "record_pending_write_summary with summary_presented=true, "
                        f"action_type={tool_name!r}, consequence_presented=true, "
                        "and confirmation_requested=true."
                    ),
                    "allowed_write_tools": [],
                    "allowed_internal_tools": pending_write_allowed_internal_tools(
                        status=status
                    ),
                    "disallowed_tools": pending_write_disallowed_tools(status=status),
                    "next_required_steps": pending_write_next_required_steps(
                        tool_name=tool_name,
                        status=status,
                    ),
                    "do_not": [
                        "Do not call a write/action tool before structured confirmation.",
                        "Do not call advance_stage before retrying the original write tool.",
                        "Do not transfer to a human agent unless the pending write protocol is structurally impossible or the pending write is denied or unclear.",
                    ],
                    "exit_condition": (
                        "The pending action has been summarized through "
                        "record_pending_write_summary and the next user response "
                        "has been recorded through record_pending_write_confirmation."
                    ),
                    "when_done": (
                        "After record_pending_write_confirmation returns confirmed, "
                        f"retry {tool_name} directly."
                    ),
                }
            )
        if status == "summarized":
            return packet.model_copy(
                update={
                    "stage": "propose_action_and_confirm",
                    "missing_facts": ["pending_write_confirmation"],
                    "ask_next": (
                        "Wait for or ask for the user's response, then call "
                        "record_pending_write_confirmation with confirmed, denied, "
                        "or unclear. If the user confirms, use decision=confirmed "
                        "and basis=latest_user_turn."
                    ),
                    "allowed_write_tools": [],
                    "allowed_internal_tools": pending_write_allowed_internal_tools(
                        status=status
                    ),
                    "disallowed_tools": pending_write_disallowed_tools(status=status),
                    "next_required_steps": pending_write_next_required_steps(
                        tool_name=tool_name,
                        status=status,
                    ),
                    "do_not": [
                        "Do not call a write/action tool before structured confirmation.",
                        "Do not call advance_stage before retrying the original write tool.",
                        "Do not transfer to a human agent unless the pending write protocol is structurally impossible or the pending write is denied or unclear.",
                    ],
                    "exit_condition": "record_pending_write_confirmation returned confirmed.",
                    "when_done": f"If confirmed, retry {tool_name} directly.",
                }
            )
        if status == "confirmed":
            return packet.model_copy(
                update={
                    "stage": "execute_write_action",
                    "missing_facts": [],
                    "ask_next": f"Retry {tool_name} now with the same confirmed arguments.",
                    "allowed_write_tools": [tool_name],
                    "allowed_internal_tools": pending_write_allowed_internal_tools(
                        status=status
                    ),
                    "disallowed_tools": pending_write_disallowed_tools(status=status),
                    "next_required_steps": pending_write_next_required_steps(
                        tool_name=tool_name,
                        status=status,
                    ),
                    "do_not": [
                        "Do not change the confirmed write arguments.",
                        "Do not call advance_stage before retrying the original write tool.",
                        "Do not transfer to a human agent unless the pending write protocol is structurally impossible or the pending write is denied or unclear.",
                    ],
                    "exit_condition": "The confirmed write tool has completed or returned an error.",
                    "when_done": "After the tool returns, call advance_stage with the visible tool result.",
                }
            )
        if status == "unclear":
            return packet.model_copy(
                update={
                    "stage": "propose_action_and_confirm",
                    "missing_facts": ["pending_write_unclear"],
                    "ask_next": (
                        "Ask one concise clarification question. After the user "
                        "responds, call record_pending_write_confirmation again "
                        "with confirmed, denied, or unclear."
                    ),
                    "allowed_write_tools": [],
                    "allowed_internal_tools": pending_write_allowed_internal_tools(
                        status=status
                    ),
                    "disallowed_tools": pending_write_disallowed_tools(status=status),
                    "next_required_steps": pending_write_next_required_steps(
                        tool_name=tool_name,
                        status=status,
                    ),
                    "do_not": [
                        "Do not retry the unclear write tool before structured confirmation.",
                        "Do not call advance_stage before resolving the pending write.",
                        "Do not transfer to a human agent unless the pending write protocol is structurally impossible or the pending write is denied or unclear.",
                    ],
                    "exit_condition": "record_pending_write_confirmation returned confirmed or denied.",
                    "when_done": f"If confirmed, retry {tool_name} directly.",
                }
            )
        if status == "denied":
            return packet.model_copy(
                update={
                    "stage": "propose_action_and_confirm",
                    "missing_facts": ["pending_write_denied"],
                    "ask_next": (
                        "Do not retry the denied pending write. Gracefully close "
                        "or offer alternative help that does not execute this write."
                    ),
                    "allowed_write_tools": [],
                    "allowed_internal_tools": [],
                    "disallowed_tools": [],
                    "next_required_steps": [
                        {
                            "step": "do_not_retry_denied_write",
                            "tool_name": tool_name,
                            "instruction": "Do not retry the denied pending write.",
                        },
                        {
                            "step": "non_write_resolution",
                            "instruction": "Gracefully close or offer alternative non-write help.",
                        },
                    ],
                    "do_not": [
                        "Do not call the denied write tool.",
                    ],
                    "exit_condition": "The denied write is not executed.",
                    "when_done": "Close or continue only with non-write assistance.",
                }
            )
        return packet

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
        pending_write = self._active_pending_write_snapshot()
        if pending_write is not None:
            status = pending_write.get("status")
            if status in {
                "needs_summary",
                "summarized",
                "denied",
                "unclear",
                "mismatched_retry",
            }:
                return "propose_action_and_confirm"
            if status == "confirmed":
                return "execute_write_action"
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
