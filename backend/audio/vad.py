"""Voice activity detection over the continuous mic stream: turn-taking
(speech start / end) and the barge-in overlap gate.

`contracts/memory.md` §3 is the literal contract: `VadGate.feed(pcm, at_ms)
-> VadEvent | None`. Two things in this file exist only because of it:

1. **Preroll (FR-05).** Detection always lags the real acoustic onset by
   however long the model needs to become confident. `speech_start_ms` is
   backdated by `VAD_PREROLL_MS` from the moment detection fires, so the STT
   worker reads the ring from before the trigger and doesn't lose the first
   word (A-01). Getting this backwards -- backdating from when *we* notice
   instead of subtracting a fixed preroll -- would silently reintroduce the
   exact bug this class exists to prevent.
2. **Overlap fires once, not as a stream** (FR-17, plan.md §4.4). It is the
   cheap non-LLM gate that keeps the barge-in classifier from being polled in
   a loop, which plan.md §4.4 measured as a 2.09x generation slowdown. Once
   `Overlap` has fired for a given overlap episode, `feed()` stays silent
   until the episode ends (agent stops talking, or the caller's speech
   drops), even if the overlap keeps going.

`VadGate` depends on `SpeechProbabilityModel`, not on silero-vad directly.
Production wiring passes a `SileroVadModel` (silero-vad on CPU, per
`plan.md` §1/§3.2); unit tests inject a deterministic fake so preroll/overlap
logic can be verified without loading real model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

import numpy as np


@dataclass(frozen=True)
class SpeechStarted:
    """Preroll already applied: `speech_start_ms` is where the STT window
    should begin, not the moment the model actually fired.
    """

    speech_start_ms: int


@dataclass(frozen=True)
class SpeechEnded:
    """`end_ms` is the acoustic boundary (when audio actually went quiet),
    not the moment `VAD_SILENCE_END_MS` of silence was confirmed -- the
    confirmation delay is detection latency, not part of the turn.
    """

    end_ms: int


@dataclass(frozen=True)
class Overlap:
    """Fires exactly once per overlap episode, when continuous overlap
    reaches the configured threshold (FR-17).
    """

    duration_ms: int


VadEvent = Union[SpeechStarted, SpeechEnded, Overlap]


class SpeechProbabilityModel(Protocol):
    """Structural type for anything that can score a PCM16 mono chunk at the
    gate's configured sample rate. `VadGate` never imports silero-vad or
    torch itself -- only a concrete implementation of this protocol does.
    """

    def __call__(self, pcm: np.ndarray) -> float: ...


class VadGate:
    """Turn-taking and barge-in-overlap detector. Stateful per session --
    one instance per `ws/dialogue` connection.
    """

    def __init__(
        self,
        model: SpeechProbabilityModel,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        preroll_ms: int = 300,
        silence_end_ms: int = 500,
        overlap_threshold_ms: int = 1000,
    ) -> None:
        self._model = model
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._preroll_ms = preroll_ms
        self._silence_end_ms = silence_end_ms
        self._overlap_threshold_ms = overlap_threshold_ms

        self._in_speech = False
        self._silence_since_ms: int | None = None

        self._agent_speaking = False
        self._overlap_start_ms: int | None = None
        self._overlap_fired = False

    def notify_agent_speaking(self, speaking: bool) -> None:
        """Tells the gate whether the agent's own voice is currently being
        played back. Not part of the `feed()` signature in `memory.md` §3 --
        overlap detection needs this and `VadGate` has no other way to learn
        it, since it only ever sees the caller's microphone audio. The
        session orchestrator (T-09) calls this when agent playback starts
        and stops.
        """
        self._agent_speaking = speaking
        if not speaking:
            self._overlap_start_ms = None
            self._overlap_fired = False

    def feed(self, pcm: bytes, at_ms: int) -> VadEvent | None:
        """`pcm` is PCM16 mono little-endian at `sample_rate`, `at_ms` is
        this chunk's start on the session clock. At most one event per call,
        matching the contract signature -- if a turn transition and an
        overlap threshold would both fire on the same chunk, the turn
        transition wins and the overlap check resumes on the next call.
        """
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return None
        chunk_duration_ms = samples.size * 1000 // self._sample_rate
        chunk_end_ms = at_ms + chunk_duration_ms
        is_speech_now = self._model(samples) >= self._threshold

        turn_event = self._update_turn_state(is_speech_now, at_ms, chunk_end_ms)
        if turn_event is not None:
            return turn_event
        return self._update_overlap_state(is_speech_now, at_ms, chunk_end_ms)

    def _update_turn_state(
        self, is_speech_now: bool, chunk_start_ms: int, chunk_end_ms: int
    ) -> VadEvent | None:
        if not self._in_speech:
            if not is_speech_now:
                return None
            self._in_speech = True
            self._silence_since_ms = None
            speech_start_ms = max(0, chunk_end_ms - self._preroll_ms)
            return SpeechStarted(speech_start_ms=speech_start_ms)

        if is_speech_now:
            self._silence_since_ms = None
            return None

        if self._silence_since_ms is None:
            self._silence_since_ms = chunk_start_ms
        silence_duration = chunk_end_ms - self._silence_since_ms
        if silence_duration < self._silence_end_ms:
            return None
        end_ms = self._silence_since_ms
        self._in_speech = False
        self._silence_since_ms = None
        return SpeechEnded(end_ms=end_ms)

    def _update_overlap_state(
        self, is_speech_now: bool, chunk_start_ms: int, chunk_end_ms: int
    ) -> VadEvent | None:
        if not (self._agent_speaking and is_speech_now):
            self._overlap_start_ms = None
            self._overlap_fired = False
            return None
        if self._overlap_start_ms is None:
            self._overlap_start_ms = chunk_start_ms
        duration = chunk_end_ms - self._overlap_start_ms
        if duration >= self._overlap_threshold_ms and not self._overlap_fired:
            self._overlap_fired = True
            return Overlap(duration_ms=duration)
        return None


class SileroVadModel:
    """Wraps silero-vad (CPU -- plan.md §1: RTF 0.066, cheap enough to be the
    "not-LLM gate" that protects the GPU). Loaded lazily: importing this
    module never requires `silero-vad`/`torch` to be installed, only
    constructing this class does. Unit tests use a fake `SpeechProbabilityModel`
    instead and never hit this class at all.

    silero-vad v5 scores fixed-size windows only (512 samples at 16kHz, 256
    at 8kHz). `VadGate` feeds it whatever chunk size the caller uses
    (`AUDIO_CHUNK_SIZE_MS`, typically 100ms = 1600 samples at 16kHz), so this
    class buffers the remainder between calls and reports the max
    probability across the whole-window(s) it consumed this call -- "was
    there speech anywhere in this chunk".
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        if sample_rate == 16000:
            window_samples = 512
        elif sample_rate == 8000:
            window_samples = 256
        else:
            raise ValueError("silero-vad only supports 8000 or 16000 Hz")
        self._sample_rate = sample_rate
        self._window_samples = window_samples
        self._carry = np.zeros(0, dtype=np.int16)
        self._model = self._load()

    @staticmethod
    def _load():
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad is not installed. Add it to pyproject.toml (plan.md "
                "§1: CPU, RTF 0.066) before constructing SileroVadModel; unit "
                "tests should inject a fake SpeechProbabilityModel instead."
            ) from exc
        model = load_silero_vad()
        model.reset_states()
        return model

    def __call__(self, pcm: np.ndarray) -> float:
        import torch

        combined = np.concatenate((self._carry, pcm)) if self._carry.size else pcm
        usable = (combined.size // self._window_samples) * self._window_samples
        if usable == 0:
            self._carry = combined
            return 0.0
        self._carry = combined[usable:].copy()
        windows = combined[:usable].reshape(-1, self._window_samples)
        audio_f32 = windows.astype(np.float32) / 32768.0
        probabilities = [
            float(self._model(torch.from_numpy(row), self._sample_rate).item())
            for row in audio_f32
        ]
        return max(probabilities)
