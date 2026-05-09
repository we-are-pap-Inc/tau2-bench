"""Typed StageGate schemas."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.utils.utils import get_now

StageGateCondition = Literal["baseline", "stage_only", "stagegate"]


class StagePacket(BaseModel):
    """Compact server-side stage instructions returned by advance_stage."""

    schema_version: str = "stagegate.stage_packet.v1"
    stage: str
    objective: str
    known_facts: dict[str, dict[str, object]] = Field(default_factory=dict)
    missing_facts: list[str] = Field(default_factory=list)
    ask_next: str
    allowed_read_tools: list[str] = Field(default_factory=list)
    allowed_write_tools: list[str] = Field(default_factory=list)
    do_not: list[str] = Field(default_factory=list)
    exit_condition: str
    when_done: str


class TraceEvent(BaseModel):
    """One JSONL trace event."""

    schema_version: str = "stagegate.trace_event.v1"
    event_type: str
    timestamp: str = Field(default_factory=get_now)
    condition: StageGateCondition
    domain: Optional[str] = None
    tick_id: Optional[int] = None
    payload: dict[str, object] = Field(default_factory=dict)
