"""Continuous audio capture: session clock, ring buffer, VAD gate.

Public surface for the rest of the backend (STT worker, dialogue session):

    from backend.audio import SessionClock, AudioRing, VadGate, SileroVadModel
    from backend.audio import SpeechStarted, SpeechEnded, Overlap, VadEvent

Nothing in this package reads `.env` directly (FR-32) -- callers pass
`sample_rate`, `capacity_seconds`, thresholds etc. explicitly, sourced from
`backend/config.py` (T-01).
"""
from __future__ import annotations

from backend.audio.clock import SessionClock
from backend.audio.ring import AudioRing
from backend.audio.vad import (
    Overlap,
    SileroVadModel,
    SpeechEnded,
    SpeechProbabilityModel,
    SpeechStarted,
    VadEvent,
    VadGate,
)

__all__ = [
    "SessionClock",
    "AudioRing",
    "VadGate",
    "VadEvent",
    "SpeechStarted",
    "SpeechEnded",
    "Overlap",
    "SpeechProbabilityModel",
    "SileroVadModel",
]
