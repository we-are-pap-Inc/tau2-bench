"""StageGate scaffold for the OpenAI audio-native path."""

from tau2.voice.audio_native.openai.stagegate.controller import StageGateController
from tau2.voice.audio_native.openai.stagegate.ledger import (
    EntityLedger,
    LedgerEvidence,
    LedgerSlot,
    LedgerStatus,
)
from tau2.voice.audio_native.openai.stagegate.stage_schema import (
    StagePacket,
    TraceEvent,
)
from tau2.voice.audio_native.openai.stagegate.validator import (
    PreWriteValidator,
    ValidatorDecision,
)

__all__ = [
    "EntityLedger",
    "LedgerEvidence",
    "LedgerSlot",
    "LedgerStatus",
    "PreWriteValidator",
    "StageGateController",
    "StagePacket",
    "TraceEvent",
    "ValidatorDecision",
]
