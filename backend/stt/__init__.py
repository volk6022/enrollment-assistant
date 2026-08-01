"""STT layer: dedicated single-worker whisper pool (`plan.md` §2/§9, T-05).

Public surface:

    from backend.stt import WhisperWorker, TranscriptionResult

Nothing in this package reads `.env` directly (FR-32) -- callers pass
model/device/throttle parameters explicitly, sourced from
`backend.config.settings.audio` (T-01).
"""

from __future__ import annotations

from backend.stt.whisper_worker import (
    TranscriptionResult,
    WhisperModelProtocol,
    WhisperSegmentProtocol,
    WhisperWorker,
)

__all__ = [
    "WhisperWorker",
    "TranscriptionResult",
    "WhisperModelProtocol",
    "WhisperSegmentProtocol",
]
