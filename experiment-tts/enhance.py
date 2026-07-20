"""Enhancement & diagnostic primitives for the TTS experiment grid.

Three orthogonal knobs the agent recommended, each isolated so the grid can
switch them on/off independently:

  1. accent   — RUAccent neural stress placement (helps Silero's `+` format).
  2. stitch    — chunk concatenation: naive vs raised-cosine crossfade.
  3. denoise  — DeepFilterNet post-filter against vocoder "digital shimmer".

Plus `clipping_stats()` to *diagnose* whether noise is a quantisation/clipping
artefact (fix upstream) before reaching for a denoiser.

Everything works on float32 mono in [-1, 1]. Heavy deps (ruaccent, deepfilternet)
are imported lazily; if absent, the wrapper reports `available == False` instead
of crashing, so the grid degrades to "skip that cell".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# int16 PCM bytes  <->  float32 mono
# ---------------------------------------------------------------------------

def pcm_bytes_to_f32(pcm: bytes) -> np.ndarray:
    """int16 PCM bytes → float32 mono in [-1, 1]."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def f32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """float32 mono in [-1, 1] → int16 PCM bytes (clipped)."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# 1. Accentuation — RUAccent
# ---------------------------------------------------------------------------

class AccentProcessor:
    """Wraps RUAccent to insert `+` stress marks Silero understands natively.

    Lazy: the model is loaded on first `process()` call. `available` is False if
    the `ruaccent` package is not installed.
    """

    def __init__(self, model_size: str = "turbo3.1") -> None:
        self.model_size = model_size
        self._accentizer = None
        self._load_failed = False

    @property
    def available(self) -> bool:
        try:
            import ruaccent  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._accentizer is not None or self._load_failed:
            return
        try:
            from ruaccent import RUAccent
            acc = RUAccent()
            # workdir=None → downloads/loads to the package cache
            acc.load(omograph_model_size=self.model_size, use_dictionary=True)
            self._accentizer = acc
        except Exception:  # noqa: BLE001 — any load failure disables the knob
            self._load_failed = True

    def process(self, text: str) -> str:
        """Return *text* with `+` stress marks, or the input unchanged on failure."""
        self._ensure_loaded()
        if self._accentizer is None:
            return text
        try:
            return self._accentizer.process_all(text)
        except Exception:  # noqa: BLE001
            return text


# ---------------------------------------------------------------------------
# 2. Stitching — naive concat vs raised-cosine crossfade
# ---------------------------------------------------------------------------

def concat(chunks: list[np.ndarray]) -> np.ndarray:
    """Naive concatenation — the baseline that can click at seams."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def crossfade(chunks: list[np.ndarray], sample_rate: int, ms: float = 20.0) -> np.ndarray:
    """Raised-cosine (equal-power) crossfade between consecutive chunks.

    Overlaps `ms` milliseconds at each junction with cos/sin ramps, which keeps
    the seam continuous (no click) without a perceptible dip. Falls back to plain
    concat when a chunk is shorter than the overlap window.
    """
    chunks = [c.astype(np.float32) for c in chunks if len(c) > 0]
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    n = max(1, int(sample_rate * ms / 1000.0))
    out = chunks[0].copy()
    for nxt in chunks[1:]:
        if len(out) < n or len(nxt) < n:
            out = np.concatenate([out, nxt])
            continue
        t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
        fade_out = np.cos(t)   # 1 → 0, equal-power
        fade_in = np.sin(t)    # 0 → 1
        head = out[:-n]
        seam = out[-n:] * fade_out + nxt[:n] * fade_in
        tail = nxt[n:]
        out = np.concatenate([head, seam, tail])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Denoise — DeepFilterNet post-filter
# ---------------------------------------------------------------------------

class Denoiser:
    """Wraps DeepFilterNet (real-time speech enhancement, ~48 kHz).

    Lazy load; `available` is False if `deepfilternet` is not installed. Audio at
    a different sample rate is resampled to 48 kHz for the model and back.
    """

    DFN_SR = 48000

    def __init__(self) -> None:
        self._model = None
        self._df_state = None
        self._load_failed = False

    @property
    def available(self) -> bool:
        try:
            import df  # noqa: F401  (the deepfilternet import name)
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_failed:
            return
        try:
            from df.enhance import init_df
            self._model, self._df_state, _ = init_df()
        except Exception:  # noqa: BLE001
            self._load_failed = True

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise float32 mono *audio*; return same sample rate. No-op on failure."""
        self._ensure_loaded()
        if self._model is None:
            return audio
        try:
            import torch
            from df.enhance import enhance

            wav = audio
            if sample_rate != self.DFN_SR:
                wav = _resample(wav, sample_rate, self.DFN_SR)
            t = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)
            out = enhance(self._model, self._df_state, t)
            out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
            if sample_rate != self.DFN_SR:
                out_np = _resample(out_np, self.DFN_SR, sample_rate)
            return out_np
        except Exception:  # noqa: BLE001
            return audio


def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Scale *audio* so its peak sits at *target_dbfs* — the fix for hard clipping.

    When a vocoder (e.g. Piper/VITS) emits samples at/over full scale, the int16
    conversion clips and produces a "digital shimmer". Normalising the float32
    peak to just under 0 dBFS removes the clipping at its source, which is the
    right fix here — cheaper and cleaner than a denoiser. No-op on silence.
    """
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    target = 10.0 ** (target_dbfs / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 8) -> np.ndarray:
    """Zero-phase Butterworth low-pass — the smoothing knob against vocoder metal.

    The metallic buzz lives in the top of the band (HF harmonics / aliasing of the
    vocoder). A clean low-pass at ~8–10 kHz shaves it while leaving speech (mostly
    < 8 kHz, sibilants ~10 kHz) intact. Zero-phase (sosfiltfilt) avoids smearing.
    No-op if the cutoff is at/above Nyquist.
    """
    if len(audio) == 0:
        return audio
    nyq = sample_rate / 2.0
    if cutoff_hz >= nyq:
        return audio
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
        return sosfiltfilt(sos, audio).astype(np.float32)
    except Exception:  # noqa: BLE001
        return audio


def high_shelf(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 7000.0,
               gain_db: float = -5.0) -> np.ndarray:
    """RBJ high-shelf biquad — gently lower everything above *cutoff_hz*.

    Softer than a low-pass: it *tilts* the top down by gain_db instead of cutting
    it off, so some "air" survives while the metallic top is tamed. gain_db < 0.
    """
    if len(audio) == 0 or gain_db == 0.0:
        return audio
    try:
        from scipy.signal import filtfilt
        import math
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * cutoff_hz / sample_rate
        cw, sw = math.cos(w0), math.sin(w0)
        alpha = sw / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / 1.0 - 1.0) + 2.0)
        two_sqrtA_alpha = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cw + two_sqrtA_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - two_sqrtA_alpha)
        a0 = (A + 1) - (A - 1) * cw + two_sqrtA_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - two_sqrtA_alpha
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0
        return filtfilt(b, a, audio).astype(np.float32)
    except Exception:  # noqa: BLE001
        return audio


def deesser(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 6000.0,
            threshold: float = 0.08, ratio: float = 4.0) -> np.ndarray:
    """Split-band de-esser — compress ONLY the HF band when it spikes.

    This is the precise tool for "highs that stab the ears": the low band passes
    untouched (voice stays present), while transient HF peaks (sibilants, vocoder
    metal) above *cutoff_hz* get gain-reduced above *threshold*. Gain is smoothed
    to avoid zipper noise. No-op on failure.
    """
    if len(audio) == 0:
        return audio
    try:
        from scipy.signal import butter, sosfilt
        sos_hp = butter(4, cutoff_hz / (sample_rate / 2.0), btype="high", output="sos")
        hf = sosfilt(sos_hp, audio).astype(np.float32)
        low = audio - hf
        env = np.abs(hf) + 1e-9
        gain = np.ones_like(env)
        over = env > threshold
        gain[over] = (threshold + (env[over] - threshold) / ratio) / env[over]
        win = max(1, int(sample_rate * 0.003))  # ~3 ms smoothing
        kernel = np.ones(win, dtype=np.float32) / win
        gain = np.convolve(gain, kernel, mode="same")
        return (low + hf * gain).astype(np.float32)
    except Exception:  # noqa: BLE001
        return audio


def soft_limit(audio: np.ndarray, gain_db: float = 3.0, ceiling: float = 0.95) -> np.ndarray:
    """Loudness "cap": boost by gain_db, then tanh soft-limit to *ceiling*.

    Raises quiet passages and smoothly caps peaks so the level sits uniformly near
    the top — the "just cap it at max volume and it's comfortable" instinct. NOTE:
    tanh adds mild harmonics, so pair it with a de-esser/shelf if the top gets edgy.
    """
    if len(audio) == 0:
        return audio
    g = 10.0 ** (gain_db / 20.0)
    return (ceiling * np.tanh(audio * g / ceiling)).astype(np.float32)


def soxr_smooth(audio: np.ndarray, sample_rate: int, via_hz: int = 20000) -> np.ndarray:
    """Anti-alias smoothing via a soxr round-trip through *via_hz*.

    Downsampling to `via_hz` with soxr's steep anti-alias filter then back up acts
    as a very clean brick-wall low-pass at via_hz/2 — smoother-sounding than a
    Butterworth for some ears. Needs librosa+soxr; falls back to no-op.
    """
    if len(audio) == 0 or via_hz >= sample_rate:
        return audio
    down = _resample(audio, sample_rate, via_hz)
    return _resample(down, via_hz, sample_rate)


def _resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """High-quality resample (soxr via librosa) with a linear fallback."""
    if sr_from == sr_to:
        return audio
    try:
        import librosa
        return librosa.resample(
            audio, orig_sr=sr_from, target_sr=sr_to, res_type="soxr_hq"
        ).astype(np.float32)
    except Exception:  # noqa: BLE001
        n_out = int(round(len(audio) * sr_to / sr_from))
        x_old = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
        return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Diagnostics — is the "digital noise" clipping/quantisation?
# ---------------------------------------------------------------------------

@dataclass
class ClipStats:
    peak: float          # max |sample| before int16 conversion
    clipped_pct: float   # fraction of samples at/over full scale
    dc_offset: float     # mean (should be ~0)
    rms: float


def clipping_stats(audio: np.ndarray) -> ClipStats:
    """Cheap diagnostics to tell a clipping/quantisation artefact from a vocoder one.

    peak ≥ 1.0 or clipped_pct > 0 → the shimmer is a conversion problem (normalise
    before int16), not the vocoder — fix that before adding a denoiser.
    """
    if len(audio) == 0:
        return ClipStats(0.0, 0.0, 0.0, 0.0)
    peak = float(np.max(np.abs(audio)))
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    return ClipStats(
        peak=peak,
        clipped_pct=clipped * 100.0,
        dc_offset=float(np.mean(audio)),
        rms=float(np.sqrt(np.mean(audio**2))),
    )
