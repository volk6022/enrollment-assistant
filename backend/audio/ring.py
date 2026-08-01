"""Continuous PCM16 ring buffer for the raw microphone stream.

`contracts/memory.md` §2 is the literal contract for this class; §7 lists the
mistakes that break the "no races" guarantee silently. Two rules matter more
than the rest of the file:

1. Recording never stops (FR-09) -- there is no `pause()` here on purpose.
   `Speaking`, `Closing`, an LLM decision in flight: none of them get to
   interrupt `append()`. Stopping even briefly would make it impossible to
   recover the start of the next user turn from the ring, which is the whole
   point of the VAD preroll (FR-05).
2. A worker thread (the whisper pool) only ever sees the return value of
   `snapshot()`, never `self._buffer`. `snapshot()` always returns a fresh
   `np.ndarray` copy -- that single rule is the entire reason this system has
   no audio data races, by construction rather than by careful scheduling.

`append()` is documented as loop-only, but `snapshot()` may legitimately be
called from a different thread than `append()` (the STT worker pool asks for
a window of audio before running whisper). Both methods take the same
`threading.Lock`, so a `snapshot()` racing an in-flight `append()` never
observes a half-written sample range or a `_head_index` that has moved but
whose samples haven't landed yet.
"""
from __future__ import annotations

import threading

import numpy as np


class AudioRing:
    """PCM16 mono ring buffer, `capacity_seconds` long (`AUDIO_RING_SECONDS`,
    default 60). Time is tracked in the caller's millisecond axis
    (`audio.clock.SessionClock.now_ms()`), not in samples -- callers never
    need to know the sample rate to address the buffer.
    """

    def __init__(self, sample_rate: int = 16000, capacity_seconds: int = 60) -> None:
        self._sample_rate = sample_rate
        self._capacity_samples = sample_rate * capacity_seconds
        self._buffer = np.zeros(self._capacity_samples, dtype=np.int16)
        self._lock = threading.Lock()
        # Absolute sample index one past the last sample ever written.
        # Monotonically non-decreasing; never resets for the life of the ring.
        self._head_index = 0
        # Absolute sample index of the very first sample ever written.
        # `None` until the first `append()`; bounds `snapshot()` from below
        # so a caller can't read "valid-looking" zeros from before the ring
        # ever had data.
        self._first_index: int | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def capacity_ms(self) -> int:
        return self._capacity_samples * 1000 // self._sample_rate

    def _index_for_ms(self, ms: int) -> int:
        return (ms * self._sample_rate) // 1000

    def append(self, pcm: bytes, at_ms: int) -> None:
        """Only from the event loop. `pcm` is PCM16 mono little-endian at
        `sample_rate`; `at_ms` is this chunk's start on the session clock.
        """
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return
        start_index = self._index_for_ms(at_ms)
        with self._lock:
            if self._first_index is None:
                self._first_index = start_index
            end_index = self._write_locked(samples, start_index)
            if end_index > self._head_index:
                self._head_index = end_index

    def _write_locked(self, samples: np.ndarray, start_index: int) -> int:
        n = samples.size
        if n >= self._capacity_samples:
            # A single chunk larger than the whole ring: only its tail could
            # ever be read back before being overwritten anyway, so drop the
            # unreadable head of it up front instead of writing it twice.
            overflow = n - self._capacity_samples
            samples = samples[overflow:]
            start_index += overflow
            n = samples.size
        start_pos = start_index % self._capacity_samples
        end_pos = start_pos + n
        if end_pos <= self._capacity_samples:
            self._buffer[start_pos:end_pos] = samples
        else:
            first_part = self._capacity_samples - start_pos
            self._buffer[start_pos:] = samples[:first_part]
            self._buffer[: end_pos - self._capacity_samples] = samples[first_part:]
        return start_index + n

    def snapshot(self, from_ms: int, to_ms: int | None = None) -> np.ndarray:
        """Returns a COPY covering `[from_ms, to_ms)` (or `[from_ms, now)` if
        `to_ms` is omitted). Silently clamps to whatever is still valid --
        audio older than `capacity_seconds` has already been overwritten and
        is gone (§2: "переполнение затирает старое молча"), and this is the
        one place that has to tolerate that quietly.
        """
        with self._lock:
            head_index = self._head_index
            first_index = self._first_index
            if first_index is None:
                return np.zeros(0, dtype=np.int16)
            from_index = self._index_for_ms(from_ms)
            to_index = head_index if to_ms is None else self._index_for_ms(to_ms)
            valid_start = max(first_index, head_index - self._capacity_samples)
            from_index = max(from_index, valid_start)
            to_index = min(to_index, head_index)
            if to_index <= from_index:
                return np.zeros(0, dtype=np.int16)
            n = to_index - from_index
            start_pos = from_index % self._capacity_samples
            end_pos = start_pos + n
            if end_pos <= self._capacity_samples:
                return self._buffer[start_pos:end_pos].copy()
            first_part = self._capacity_samples - start_pos
            return np.concatenate(
                (self._buffer[start_pos:], self._buffer[: end_pos - self._capacity_samples])
            )
