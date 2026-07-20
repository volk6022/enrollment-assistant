"""Audio I/O and conversion utilities for the TTS benchmark harness."""

import struct
import wave
from pathlib import Path

import numpy as np


def audio_duration_s(audio_data: np.ndarray, sample_rate: int) -> float:
    """Return the playback duration of *audio_data* in seconds."""
    return len(audio_data) / sample_rate


def save_wav(audio_data: np.ndarray, path: Path, sample_rate: int) -> None:
    """Save *audio_data* (float32 or int16) to a WAV file at *path*.

    float32 arrays are assumed to be in [-1, 1] and converted to int16 before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    elif audio_data.dtype == np.int16:
        pcm = audio_data
    else:
        pcm = audio_data.astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw int16 PCM bytes with a WAV header and return the result as bytes.

    The returned bytes form a complete, valid WAV file that can be written directly
    to disk or sent over a network without further processing.
    """
    num_samples = len(pcm_bytes) // 2  # int16 → 2 bytes per sample
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    chunk_size = 36 + data_size  # RIFF chunk size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,               # sub-chunk 1 size for PCM
        1,                # audio format: PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def save_wav_from_bytes(pcm_bytes: bytes, path: Path, sample_rate: int, channels: int = 1) -> None:
    """Wrap *pcm_bytes* in a WAV header and write to *path*.

    Convenience wrapper around pcm_to_wav() + file I/O.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=sample_rate, channels=channels)
    path.write_bytes(wav_bytes)
