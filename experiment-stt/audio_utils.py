"""Audio loading and resampling utilities for the STT benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np


TARGET_SR = 16_000  # all models expect 16 kHz mono


def load_audio_float32(audio_path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, float]:
    """Load an audio file, resample to target_sr, convert to mono float32.

    Returns a tuple of (waveform, duration_seconds).
    waveform shape: (num_samples,) — 1-D float32 numpy array, values in [-1, 1].
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("Install librosa: uv add librosa") from exc

    # librosa loads as float32 mono by default; resamples when sr != None
    waveform, _ = librosa.load(str(audio_path), sr=target_sr, mono=True, dtype=np.float32)
    duration_s = len(waveform) / target_sr
    return waveform, duration_s


def load_audio_soundfile(audio_path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, float]:
    """Load audio with soundfile + resampy (lighter dependency than librosa).

    Falls back to librosa if resampy is not installed.
    Returns (waveform float32 1-D, duration_seconds).
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("Install soundfile: uv add soundfile") from exc

    waveform, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)

    # Convert stereo to mono by averaging channels
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if sr != target_sr:
        try:
            import resampy
            waveform = resampy.resample(waveform, sr, target_sr)
        except ImportError:
            # Fall back to librosa for resampling
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    duration_s = len(waveform) / target_sr
    return waveform.astype(np.float32), duration_s


def waveform_to_tensor(waveform: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
    """Convert a float32 numpy waveform to a 1-D torch.Tensor (CPU)."""
    import torch
    return torch.from_numpy(waveform)


def audio_duration_s(audio_path: Path) -> float:
    """Return audio duration in seconds without fully decoding the file."""
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        _, duration = load_audio_soundfile(audio_path)
        return duration
