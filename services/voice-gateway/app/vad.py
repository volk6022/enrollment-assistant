from __future__ import annotations

import math
import struct


def rms_level(raw_pcm16: bytes) -> float:
    if not raw_pcm16:
        return 0.0
    sample_count = len(raw_pcm16) // 2
    if sample_count <= 0:
        return 0.0
    samples = struct.unpack("<" + "h" * sample_count, raw_pcm16[: sample_count * 2])
    squares = sum(float(s) * float(s) for s in samples)
    return math.sqrt(squares / sample_count)


def has_speech(raw_pcm16: bytes, threshold: float = 500.0) -> bool:
    return rms_level(raw_pcm16) >= threshold
