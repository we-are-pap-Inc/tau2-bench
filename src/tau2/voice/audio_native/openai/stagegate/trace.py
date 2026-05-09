"""JSONL trace writer for StageGate."""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.voice.audio_native.openai.stagegate.stage_schema import TraceEvent

TRACE_ENV_VAR = "TAU2_TRACE_JSONL"


class JsonlTraceWriter:
    """Append StageGate events to JSONL when enabled by environment."""

    def __init__(self, path: Optional[Path]):
        self.path = path

    @classmethod
    def from_env(cls) -> "JsonlTraceWriter":
        """Create a writer using TAU2_TRACE_JSONL, or a no-op writer."""
        path = os.environ.get(TRACE_ENV_VAR)
        if not path:
            return cls(None)
        return cls(Path(path))

    @property
    def enabled(self) -> bool:
        """Whether trace writing is active."""
        return self.path is not None

    def write(self, event: TraceEvent) -> None:
        """Write one event if tracing is enabled."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except OSError as exc:
            logger.error(f"Failed to write StageGate trace event: {exc}")
            raise
