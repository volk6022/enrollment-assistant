"""VRAM, RAM, and audio duration measurement helpers."""

from pathlib import Path


# ---------------------------------------------------------------------------
# VRAM helpers (torch.cuda)
# ---------------------------------------------------------------------------

def vram_snapshot() -> float:
    """Return currently allocated VRAM in MB (0.0 if CUDA is unavailable)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 2
    except ImportError:
        pass
    return 0.0


def vram_peak_reset() -> None:
    """Reset the torch peak VRAM counter so the next vram_peak_mb() call is accurate."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def vram_peak_mb() -> float:
    """Return peak VRAM allocated since the last vram_peak_reset() call, in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except ImportError:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# RAM helper
# ---------------------------------------------------------------------------

def ram_mb() -> float:
    """Return the current process RSS (resident set size) in MB."""
    try:
        import psutil
        import os
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        return 0.0


# ---------------------------------------------------------------------------
# Audio duration helper
# ---------------------------------------------------------------------------

def measure_audio_duration(wav_path_or_bytes: "Path | bytes", sample_rate: int) -> float:
    """Return the duration in seconds of a WAV file or raw int16 PCM bytes.

    *wav_path_or_bytes* can be a Path to a WAV file or raw int16 PCM bytes.
    When it is a Path, the sample rate is read from the WAV header (the *sample_rate*
    argument is ignored).  When it is raw bytes, *sample_rate* is required.
    """
    import numpy as np

    if isinstance(wav_path_or_bytes, (str, Path)):
        try:
            import soundfile as sf
            data, sr = sf.read(str(wav_path_or_bytes), dtype="int16")
            return len(data) / sr
        except Exception:
            # Fallback via scipy
            from scipy.io import wavfile  # type: ignore[import]
            sr, data = wavfile.read(str(wav_path_or_bytes))
            return len(data) / sr
    else:
        # Raw int16 PCM bytes
        audio_np = np.frombuffer(wav_path_or_bytes, dtype=np.int16)
        return len(audio_np) / sample_rate
