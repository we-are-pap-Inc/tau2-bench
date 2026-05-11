"""Typed StageGate schemas."""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

StageGateCondition = Literal["baseline", "stage_only", "stagegate"]


class EvidenceSource(str, Enum):
    """Source class for evidence admitted into StageGate runtime state."""

    AGENT_VISIBLE_TRANSCRIPT = "agent_visible_transcript"
    MODEL_TOOL_ARGUMENT = "model_tool_argument"
    DOMAIN_TOOL_OUTPUT = "domain_tool_output"
    ASSISTANT_UTTERANCE = "assistant_utterance"
    SIMULATOR_GOLD_TEXT = "simulator_gold_text"
    POSTHOC_ORACLE = "posthoc_oracle"


RUNTIME_FORBIDDEN_EVIDENCE_SOURCES = {
    EvidenceSource.SIMULATOR_GOLD_TEXT,
    EvidenceSource.POSTHOC_ORACLE,
}


def ensure_runtime_evidence_source(source: EvidenceSource | str) -> EvidenceSource:
    """Return a typed runtime evidence source or reject forbidden oracle sources."""
    typed_source = EvidenceSource(source)
    if typed_source in RUNTIME_FORBIDDEN_EVIDENCE_SOURCES:
        raise ValueError(
            f"{typed_source.value} is not an allowed StageGate runtime evidence source"
        )
    return typed_source


def get_trace_ts() -> str:
    """Return an RFC 3339 UTC timestamp for JSONL trace events."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class StagePacket(BaseModel):
    """Compact server-side stage instructions returned by advance_stage."""

    schema_version: str = "stagegate.stage_packet.v1"
    stage: str
    objective: str
    known_facts: dict[str, dict[str, object]] = Field(default_factory=dict)
    missing_facts: list[str] = Field(default_factory=list)
    ambiguous_facts: list[str] = Field(default_factory=list)
    ask_next: str
    allowed_read_tools: list[str] = Field(default_factory=list)
    allowed_write_tools: list[str] = Field(default_factory=list)
    allowed_internal_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    next_required_steps: list[dict[str, object]] = Field(default_factory=list)
    do_not: list[str] = Field(default_factory=list)
    exit_condition: str
    when_done: str


class TraceEvent(BaseModel):
    """One JSONL trace event."""

    schema_version: str = "stagegate.trace.v1"
    ts: str = Field(default_factory=get_trace_ts)
    event_type: str
    condition: StageGateCondition
    run_id: Optional[str] = None
    domain: Optional[str] = None
    benchmark_task_id: Optional[str] = None
    sim_id: Optional[str] = None
    trial: Optional[int] = None
    stage: Optional[str] = None
    turn_index: Optional[int] = None
    tick_index: Optional[int] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    visible_to_agent: bool = True
    source: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, object]] = None
    ledger_delta: Optional[dict[str, object]] = None
    validator_decision: Optional[str] = None
    validator_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    leakage_risk: str = "none"
    payload: dict[str, object] = Field(default_factory=dict)
