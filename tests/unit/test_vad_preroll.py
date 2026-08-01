"""Unit tests for `backend.audio.vad.VadGate`.

The load-bearing test here is `test_preroll_captures_full_word_started_before_vad_fired`
(A-01 / FR-05): detection lags the real acoustic onset, and `VAD_PREROLL_MS`
must back the reported `speech_start_ms` up far enough that reading
`AudioRing` from that mark captures the whole word, not just the tail of it
after the model noticed.

`VadGate` is driven by an injected `SpeechProbabilityModel` fake -- no real
silero-vad model or network access needed to verify the preroll/turn/overlap
state machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.audio.ring import AudioRing
from backend.audio.vad import Overlap, SpeechEnded, SpeechStarted, VadGate


class ScriptedModel:
    """Returns 1.0 (speech) from the `speech_from_call_index`-th call
    onward, 0.0 before that -- simulates a VAD that needs some number of
    calls to become confident, independent of what's actually in `pcm`.
    """

    def __init__(self, speech_from_call_index: int, speech_until_call_index: int | None = None) -> None:
        self._call_index = 0
        self._speech_from = speech_from_call_index
        self._speech_until = speech_until_call_index

    def __call__(self, pcm: np.ndarray) -> float:
        i = self._call_index
        self._call_index += 1
        if i < self._speech_from:
            return 0.0
        if self._speech_until is not None and i >= self._speech_until:
            return 0.0
        return 1.0


def test_preroll_captures_full_word_started_before_vad_fired() -> None:
    sample_rate = 16000
    chunk_ms = 50
    chunk_samples = sample_rate * chunk_ms // 1000  # 800

    real_onset_ms = 1000
    detection_lag_ms = 250
    # The model reports speech from the chunk covering
    # [real_onset_ms + detection_lag_ms, ...) onward.
    speech_from_chunk = (real_onset_ms + detection_lag_ms) // chunk_ms  # index 25

    ring = AudioRing(sample_rate=sample_rate, capacity_seconds=5)
    gate = VadGate(
        model=ScriptedModel(speech_from_call_index=speech_from_chunk),
        sample_rate=sample_rate,
        preroll_ms=300,
    )

    SPEECH_MARKER = np.int16(12345)
    total_chunks = 30  # 1500 ms of timeline
    event: SpeechStarted | None = None
    for i in range(total_chunks):
        at_ms = i * chunk_ms
        is_real_speech = at_ms >= real_onset_ms
        value = SPEECH_MARKER if is_real_speech else np.int16(0)
        chunk = np.full(chunk_samples, value, dtype=np.int16)
        ring.append(chunk.tobytes(), at_ms=at_ms)

        result = gate.feed(chunk.tobytes(), at_ms=at_ms)
        if isinstance(result, SpeechStarted):
            event = result
            break

    assert event is not None, "VadGate never fired SpeechStarted"
    # VAD only "noticed" at real_onset_ms + detection_lag_ms; with a 300 ms
    # preroll the reported start must still land at or before the real onset.
    assert event.speech_start_ms <= real_onset_ms

    window = ring.snapshot(event.speech_start_ms, real_onset_ms + chunk_ms)
    first_word_ms = real_onset_ms - event.speech_start_ms
    first_word_samples = first_word_ms * sample_rate // 1000

    # Everything before the real onset inside the window is silence (proves
    # the window starts early enough on its own, not by accident)...
    assert np.all(window[:first_word_samples] == 0)
    # ...and the first sample of the real word is present in the window,
    # not cut off. Without the preroll this would fail: a window starting at
    # the trigger point (1250 ms) would miss the word entirely.
    assert window[first_word_samples] == SPEECH_MARKER


def test_speech_started_backdates_by_preroll_ms() -> None:
    gate = VadGate(model=ScriptedModel(speech_from_call_index=0), preroll_ms=300)
    event = gate.feed(np.zeros(1600, dtype=np.int16).tobytes(), at_ms=1000)
    assert isinstance(event, SpeechStarted)
    # chunk covers [1000, 1100); preroll backs the 1100 trigger point up by 300
    assert event.speech_start_ms == 800


def test_speech_started_preroll_never_goes_negative() -> None:
    gate = VadGate(model=ScriptedModel(speech_from_call_index=0), preroll_ms=300)
    event = gate.feed(np.zeros(1600, dtype=np.int16).tobytes(), at_ms=0)
    assert isinstance(event, SpeechStarted)
    assert event.speech_start_ms == 0


def test_speech_ended_ignores_brief_pause_under_silence_threshold() -> None:
    # speech, one quiet 100 ms chunk (well under the 500 ms threshold), speech resumes.
    calls = iter([1.0, 1.0, 0.0, 1.0, 1.0])

    class ManualModel:
        def __call__(self, pcm: np.ndarray) -> float:
            return next(calls)

    gate = VadGate(model=ManualModel(), silence_end_ms=500)
    chunk = np.zeros(1600, dtype=np.int16).tobytes()  # 100 ms per call

    assert isinstance(gate.feed(chunk, at_ms=0), SpeechStarted)
    assert gate.feed(chunk, at_ms=100) is None
    assert gate.feed(chunk, at_ms=200) is None  # brief silence, below threshold
    assert gate.feed(chunk, at_ms=300) is None  # speech resumes: silence timer resets
    assert gate.feed(chunk, at_ms=400) is None


def test_speech_ended_fires_after_continuous_silence_threshold() -> None:
    calls = iter([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 2 speech, then 5x100ms silence

    class ManualModel:
        def __call__(self, pcm: np.ndarray) -> float:
            return next(calls)

    gate = VadGate(model=ManualModel(), silence_end_ms=500)
    chunk = np.zeros(1600, dtype=np.int16).tobytes()  # 100 ms per call

    started = gate.feed(chunk, at_ms=0)
    assert isinstance(started, SpeechStarted)
    assert gate.feed(chunk, at_ms=100) is None  # still speech

    # Silence starts at at_ms=200. Needs 500ms continuous (until 700) to confirm.
    assert gate.feed(chunk, at_ms=200) is None
    assert gate.feed(chunk, at_ms=300) is None
    assert gate.feed(chunk, at_ms=400) is None
    assert gate.feed(chunk, at_ms=500) is None
    ended = gate.feed(chunk, at_ms=600)
    assert isinstance(ended, SpeechEnded)
    assert ended.end_ms == 200  # the acoustic boundary, not the confirmation time


def test_overlap_fires_exactly_once_per_episode() -> None:
    gate = VadGate(model=ScriptedModel(speech_from_call_index=0), overlap_threshold_ms=300)
    chunk = np.zeros(1600, dtype=np.int16).tobytes()  # 100 ms per call

    gate.notify_agent_speaking(True)
    # First call also fires SpeechStarted (turn transition wins this chunk),
    # so the overlap counter only actually starts running from the next call.
    first = gate.feed(chunk, at_ms=0)
    assert isinstance(first, SpeechStarted)

    second = gate.feed(chunk, at_ms=100)  # overlap so far: 100-200 = 100ms < 300
    assert second is None

    third = gate.feed(chunk, at_ms=200)  # overlap so far: 100-300 = 200ms < 300
    assert third is None

    fourth = gate.feed(chunk, at_ms=300)  # overlap so far: 100-400 = 300ms >= 300
    assert isinstance(fourth, Overlap)

    fifth = gate.feed(chunk, at_ms=400)  # still overlapping, but already fired
    assert fifth is None

    gate.notify_agent_speaking(False)
    gate.notify_agent_speaking(True)
    sixth = gate.feed(chunk, at_ms=500)  # new episode, not enough duration yet
    assert sixth is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
