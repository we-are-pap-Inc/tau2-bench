"""
OpenAI Realtime API implementation for audio native adapters.
"""

from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    BaseRealtimeEvent,
    ErrorEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    InputAudioTranscriptionCompletedEvent,
    OutputItemAddedEvent,
    RealtimeEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TextDeltaEvent,
    TimeoutEvent,
    UnknownEvent,
    parse_realtime_event,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


def __getattr__(name: str):
    """Lazy-load provider classes so core imports do not require voice extras."""
    if name == "DiscreteTimeOpenAIAdapter":
        from tau2.voice.audio_native.openai.discrete_time_adapter import (
            DiscreteTimeOpenAIAdapter,
        )

        return DiscreteTimeOpenAIAdapter
    if name in {"OpenAIRealtimeProvider", "OpenAIVADConfig", "OpenAIVADMode"}:
        from tau2.voice.audio_native.openai.provider import (
            OpenAIRealtimeProvider,
            OpenAIVADConfig,
            OpenAIVADMode,
        )

        return {
            "OpenAIRealtimeProvider": OpenAIRealtimeProvider,
            "OpenAIVADConfig": OpenAIVADConfig,
            "OpenAIVADMode": OpenAIVADMode,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Adapters
    "DiscreteTimeOpenAIAdapter",
    # Config (in provider.py)
    "OpenAIVADConfig",
    "OpenAIVADMode",
    # Events
    "BaseRealtimeEvent",
    "TextDeltaEvent",
    "AudioDeltaEvent",
    "AudioDoneEvent",
    "AudioTranscriptDeltaEvent",
    "AudioTranscriptDoneEvent",
    "FunctionCallArgumentsDeltaEvent",
    "FunctionCallArgumentsDoneEvent",
    "InputAudioTranscriptionCompletedEvent",
    "OutputItemAddedEvent",
    "ResponseDoneEvent",
    "SpeechStartedEvent",
    "SpeechStoppedEvent",
    "ErrorEvent",
    "TimeoutEvent",
    "UnknownEvent",
    "RealtimeEvent",
    "parse_realtime_event",
    # Provider
    "OpenAIRealtimeProvider",
    # Tick-based simulation
    "TickResult",
    "UtteranceTranscript",
]
