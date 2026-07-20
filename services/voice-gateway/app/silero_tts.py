"""Local Silero TTS client for the voice gateway (replaces Yandex SpeechKit).

Drop-in for SpeechKitTTSClient: exposes `synthesize(text) -> (bytes, suffix, mime)`
returning a 16-bit PCM WAV. Encapsulates the production chain chosen by the TTS
experiment (see experiment-tts/production_voice.py and
19-In-Work/stt-tts-research-2026-07.md):

    Silero v5_5_ru / baya / 48 kHz
    → RUAccent `+` stress (optional, graceful if unavailable)
    → apply_tts
    → soxr anti-alias (via 20 kHz) → de-esser (5.5 kHz / 0.06 / x5) → peak-norm -1 dBFS

Runs on CPU by default (RTF ~0.04; the gateway flow is file-based, not hard-realtime),
so no GPU is required in production. Set VOICE_TTS_DEVICE=cuda if a GPU is present.

Self-contained apart from numpy/scipy/soxr/torch + optional ruaccent — no import from
the experiment tree, so the service stays independently deployable.
"""

from __future__ import annotations

import io
import re
import wave
from xml.sax.saxutils import escape

import numpy as np

from .config import settings
from .text_normalize import normalize_for_tts

# ---- production chain constants (single source of truth for the gateway) ----
_SOXR_VIA_HZ = 20000
_DEESS = (5500.0, 0.06, 5.0)   # (cutoff_hz, threshold, ratio)

# ---- emotion markers → Silero SSML (mirrors experiment-tts/intonation_ssml.py) ----
_MARKER_MAP = {
    "[q]": {"pitch": "high"},
    "[q_strong]": {"pitch": "x-high"},
    "[exc]": {"pitch": "x-high", "rate": "fast"},
    "[calm]": {"pitch": "low", "rate": "slow"},
    "[fast]": {"rate": "fast"},
    "[slow]": {"rate": "x-slow"},
    "[emp]": {"pitch": "high", "rate": "slow"},
    "[pause:short]": {"break_ms": 150},
    "[pause]": {"break_ms": 400},
    "[pause:long]": {"break_ms": 800},
}
_MARKER_RE = re.compile(r"\[(?:q_strong|q|exc|calm|fast|slow|emp|pause(?::\w+)?)\]")


def _wrap_prosody(text: str, attrs: dict) -> str:
    pa = {k: v for k, v in attrs.items() if k in ("pitch", "rate")}
    if not pa:
        return escape(text)
    attr_str = " ".join(f'{k}="{v}"' for k, v in pa.items())
    return f"<prosody {attr_str}>{escape(text)}</prosody>"


def _markers_to_ssml_body(text: str) -> str:
    """Convert marker-annotated text to an SSML body (no <speak> wrapper)."""
    parts = _MARKER_RE.split(text)
    marks = _MARKER_RE.findall(text)
    out = []
    if parts and parts[0].strip():
        out.append(escape(parts[0]))
    for mark, following in zip(marks, parts[1:]):
        attrs = _MARKER_MAP.get(mark, {})
        if "break_ms" in attrs:
            out.append(f'<break time="{attrs["break_ms"]}ms"/>')
            if following.strip():
                out.append(escape(following))
        elif following.strip():
            out.append(_wrap_prosody(following, attrs))
    return "".join(out)
_WARMUP = [
    "Прогрев.",
    "Короткий прогрев модели синтеза речи.",
    "Более длинное предложение для прогрева всех веток компиляции модели.",
]


# --------------------------------------------------------------------------- #
# DSP (inline; float32 mono in [-1, 1])
# --------------------------------------------------------------------------- #

def _peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    return (audio * (10.0 ** (target_dbfs / 20.0) / peak)).astype(np.float32)


def _soxr_smooth(audio: np.ndarray, sr: int, via_hz: int = _SOXR_VIA_HZ) -> np.ndarray:
    """Anti-alias smoothing: soxr round-trip through via_hz (brick-wall LP at via/2)."""
    if len(audio) == 0 or via_hz >= sr:
        return audio
    try:
        import soxr
        down = soxr.resample(audio, sr, via_hz)
        return soxr.resample(down, via_hz, sr).astype(np.float32)
    except Exception:  # noqa: BLE001
        return audio


def _deesser(audio: np.ndarray, sr: int, cutoff_hz: float, threshold: float,
             ratio: float) -> np.ndarray:
    """Split-band de-esser: compress only HF transients above threshold."""
    if len(audio) == 0:
        return audio
    try:
        from scipy.signal import butter, sosfilt
        sos_hp = butter(4, cutoff_hz / (sr / 2.0), btype="high", output="sos")
        hf = sosfilt(sos_hp, audio).astype(np.float32)
        low = audio - hf
        env = np.abs(hf) + 1e-9
        gain = np.ones_like(env)
        over = env > threshold
        gain[over] = (threshold + (env[over] - threshold) / ratio) / env[over]
        win = max(1, int(sr * 0.003))
        gain = np.convolve(gain, np.ones(win, dtype=np.float32) / win, mode="same")
        return (low + hf * gain).astype(np.float32)
    except Exception:  # noqa: BLE001
        return audio


def _f32_to_wav(audio: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class SileroTTSClient:
    """Local Silero synthesis with the chosen production post-chain."""

    def __init__(self) -> None:
        self.version = settings.silero_version
        self.speaker = settings.silero_speaker
        self.sample_rate = int(settings.silero_sample_rate_hz)
        self.device = settings.silero_device
        self.use_accent = settings.silero_use_accent
        self._model = None
        self._torch = None
        self._accentor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        self._torch = torch
        if settings.silero_cache_dir:
            torch.hub.set_dir(settings.silero_cache_dir)
        model, _ = torch.hub.load(
            "snakers4/silero-models", "silero_tts",
            language="ru", speaker=self.version, trust_repo=True,
        )
        model.to(self.device)
        self._model = model

        if self.use_accent:
            try:
                from ruaccent import RUAccent
                acc = RUAccent()
                acc.load(omograph_model_size="turbo3.1", use_dictionary=True)
                self._accentor = acc
            except Exception:  # noqa: BLE001 — accenting is optional
                self._accentor = None

        for w in _WARMUP:
            try:
                self._raw(w)
            except Exception:  # noqa: BLE001
                pass

    def _raw(self, text: str) -> np.ndarray:
        with self._torch.no_grad():
            wav = self._model.apply_tts(
                text=text, speaker=self.speaker, sample_rate=self.sample_rate,
                put_accent=True, put_yo=True,
            )
        return wav.squeeze().cpu().numpy().astype(np.float32)

    def _raw_ssml(self, ssml_text: str) -> np.ndarray:
        with self._torch.no_grad():
            wav = self._model.apply_tts(
                ssml_text=ssml_text, speaker=self.speaker, sample_rate=self.sample_rate,
            )
        return wav.squeeze().cpu().numpy().astype(np.float32)

    def _post(self, audio: np.ndarray) -> np.ndarray:
        audio = _soxr_smooth(audio, self.sample_rate, _SOXR_VIA_HZ)
        audio = _deesser(audio, self.sample_rate, *_DEESS)
        return _peak_normalize(audio, -1.0)

    def synthesize(self, text: str) -> tuple[bytes, str, str]:
        """text → (wav_bytes, ".wav", "audio/wav"). Drop-in for SpeechKitTTSClient.

        If the text carries emotion markers ([q]/[emp]/[pause]…), route through the
        SSML path so Silero applies prosody; otherwise use the plain (RUAccent) path.
        """
        text = (text or "").strip()
        if not text:
            raise RuntimeError("Empty TTS text")
        text = normalize_for_tts(text)
        self._ensure_loaded()
        if _MARKER_RE.search(text):
            ssml = f"<speak><p><s>{_markers_to_ssml_body(text)}</s></p></speak>"
            audio = self._post(self._raw_ssml(ssml))
        else:
            audio = self._post(self._raw(self._accent(text)))
        return _f32_to_wav(audio, self.sample_rate), ".wav", "audio/wav"

    def _accent(self, text: str) -> str:
        """Best-effort RUAccent stress marking; fall back to raw text on any error.

        RUAccent's omograph ONNX path can raise (e.g. missing token_type_ids) on
        certain inputs — accenting is an enhancement, not a hard dependency, so we
        never let it break synthesis. Silero still speaks, just without the extra
        dictionary stress on that utterance.
        """
        if self._accentor is None:
            return text
        try:
            return self._accentor.process_all(text)
        except Exception:  # noqa: BLE001
            return text
