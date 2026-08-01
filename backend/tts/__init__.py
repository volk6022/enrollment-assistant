"""TTS layer: dedicated single-worker CPU silero pool (`plan.md` §2/§3.4, T-05).

Public surface:

    from backend.tts import SileroWorker, SynthesisResult, normalize_for_tts

Nothing in this package reads `.env` directly (FR-32) -- callers pass
speaker/device/sample-rate parameters explicitly, sourced from
`backend.config.settings.audio` (T-01).
"""

from __future__ import annotations

from backend.tts.silero_worker import SileroModelProtocol, SileroWorker, SynthesisResult
from backend.tts.text_normalize import normalize_for_tts

__all__ = [
    "SileroWorker",
    "SynthesisResult",
    "SileroModelProtocol",
    "normalize_for_tts",
]
