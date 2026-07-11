from __future__ import annotations

import io
import struct
import wave
from typing import Tuple


def pcm16_to_wav_bytes(raw: bytes, sample_rate_hz: int = 8000, channels: int = 1, sample_width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(raw)
    return buffer.getvalue()


def _downmix_stereo_pcm16(frames: bytes) -> bytes:
    sample_count = len(frames) // 2
    if sample_count <= 1:
        return frames
    samples = struct.unpack("<" + "h" * sample_count, frames)
    mono = []
    for i in range(0, len(samples) - 1, 2):
        mono.append(int((samples[i] + samples[i + 1]) / 2))
    return struct.pack("<" + "h" * len(mono), *mono)


def wav_bytes_to_pcm16_mono(wav_bytes: bytes) -> Tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported in demo mode")
    if channels == 1:
        return frames, sample_rate
    if channels == 2:
        return _downmix_stereo_pcm16(frames), sample_rate
    raise ValueError("Only mono or stereo WAV is supported in demo mode")


def normalize_audio_for_stt(
    audio_bytes: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    sample_rate_hertz: int = 8000,
    audio_format: str = "lpcm",
) -> Tuple[bytes, int, str]:
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    fmt = (audio_format or "lpcm").lower()

    if name.endswith(".wav") or ctype in {"audio/wav", "audio/x-wav", "audio/wave"}:
        pcm, sr = wav_bytes_to_pcm16_mono(audio_bytes)
        return pcm, sr, "lpcm"

    if name.endswith(".ogg") or "ogg" in ctype:
        return audio_bytes, sample_rate_hertz, "oggopus"

    if name.endswith(".mp3") or ctype == "audio/mpeg":
        return audio_bytes, sample_rate_hertz, "mp3"

    return audio_bytes, sample_rate_hertz, fmt
