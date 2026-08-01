"""Unit tests for `backend.audio.ring.AudioRing`.

Covers the two properties `contracts/memory.md` §2/§7 call out as the reason
this system has no audio data races: `snapshot()` returns an independent
copy, and ring overflow overwrites the oldest audio without corrupting what
is still valid.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.audio.ring import AudioRing


def test_snapshot_returns_independent_copy() -> None:
    ring = AudioRing(sample_rate=16000, capacity_seconds=1)
    original = np.arange(100, dtype=np.int16)
    ring.append(original.tobytes(), at_ms=0)

    snap = ring.snapshot(0, None)
    assert np.array_equal(snap, original)
    assert not np.shares_memory(snap, ring._buffer)  # not a view into ring storage

    snap[:] = -1  # mutate aggressively; must not reach the ring

    snap_again = ring.snapshot(0, None)
    assert np.array_equal(snap_again, original)
    assert not np.array_equal(snap_again, snap)


def test_snapshot_across_wraparound_is_also_a_copy() -> None:
    sample_rate = 16000
    ring = AudioRing(sample_rate=sample_rate, capacity_seconds=1)
    chunk_samples = sample_rate // 10  # 100 ms
    # Fill past capacity so the read below straddles the wrap point.
    for i in range(12):
        ring.append(np.full(chunk_samples, i + 1, dtype=np.int16).tobytes(), at_ms=i * 100)

    snap = ring.snapshot(300, 900)
    assert snap.size == (900 - 300) * sample_rate // 1000
    mutated = snap.copy()
    snap[:] = 0
    snap_again = ring.snapshot(300, 900)
    assert np.array_equal(snap_again, mutated)


def test_overflow_overwrites_oldest_correctly() -> None:
    sample_rate = 16000
    capacity_seconds = 1
    ring = AudioRing(sample_rate=sample_rate, capacity_seconds=capacity_seconds)
    chunk_ms = 100
    chunk_samples = sample_rate * chunk_ms // 1000  # 1600

    total_chunks = 15  # 1500 ms of audio into a 1000 ms ring -> 500 ms overwritten
    for i in range(total_chunks):
        value = i + 1  # nonzero so "never written" (0) is distinguishable
        ring.append(np.full(chunk_samples, value, dtype=np.int16).tobytes(), at_ms=i * chunk_ms)

    total_ms = total_chunks * chunk_ms  # 1500

    # The oldest 500 ms (chunks 0..4) no longer exist -- overwritten by wraparound.
    stale = ring.snapshot(0, 500)
    assert stale.size == 0

    # The remaining full capacity (1000 ms) must be exactly chunks 5..14, in order,
    # not some mangled interleaving of old and new samples at the wrap boundary.
    recent = ring.snapshot(total_ms - 1000, total_ms)
    expected = np.concatenate(
        [np.full(chunk_samples, i + 1, dtype=np.int16) for i in range(5, total_chunks)]
    )
    assert np.array_equal(recent, expected)


def test_snapshot_before_any_data_is_empty() -> None:
    ring = AudioRing(sample_rate=16000, capacity_seconds=1)
    assert ring.snapshot(0, 100).size == 0


def test_snapshot_open_ended_reads_up_to_now() -> None:
    ring = AudioRing(sample_rate=16000, capacity_seconds=1)
    ring.append(np.full(1600, 7, dtype=np.int16).tobytes(), at_ms=0)
    ring.append(np.full(1600, 9, dtype=np.int16).tobytes(), at_ms=100)

    snap = ring.snapshot(0)
    assert snap.size == 3200
    assert np.all(snap[:1600] == 7)
    assert np.all(snap[1600:] == 9)


def test_append_ignores_empty_payload() -> None:
    ring = AudioRing(sample_rate=16000, capacity_seconds=1)
    ring.append(b"", at_ms=0)
    assert ring.snapshot(0).size == 0


def test_default_capacity_is_sixty_seconds() -> None:
    ring = AudioRing()
    assert ring.capacity_ms == 60_000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
