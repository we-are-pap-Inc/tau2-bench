"""StageGate scaffold for the OpenAI audio-native path."""

from tau2.voice.audio_native.openai.stagegate.controller import StageGateController
from tau2.voice.audio_native.openai.stagegate.stage_schema import (
    StagePacket,
    TraceEvent,
)

__all__ = [
    "StageGateController",
    "StagePacket",
    "TraceEvent",
]
