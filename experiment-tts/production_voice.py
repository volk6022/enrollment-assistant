"""Canonical production voice — the chain chosen by the TTS experiment.

Locks in the decision reached by auditioning the full grid:

    engine    Silero TTS, bundle v5_5_ru  (v4 had a metallic vocoder buzz)
    voice     baya
    rate      48 kHz  (24 kHz aliased → more metal)
    accent    RUAccent `+` stress preprocessing
    post      soxr anti-alias (via 20 kHz) → de-esser (6→5.5 kHz, thr 0.06, ×5)
              → peak-normalize -1 dBFS      [grid id: soxr_deess_s2_20k]
    stitch    raised-cosine crossfade 20 ms between sentences (streaming)

Rationale in results/ (grid_*.md) and 19-In-Work/stt-tts-research-2026-07.md.

This module is self-contained apart from enhance.py / intonation_ssml.py and is
meant to be the single source of truth for synthesis config — drop it behind the
voice-gateway TTS client to go local, or import directly.
"""

from __future__ import annotations

import time

import numpy as np

from enhance import (
    crossfade,
    deesser,
    f32_to_pcm_bytes,
    peak_normalize,
    soxr_smooth,
)
from intonation_ssml import markers_to_ssml

# ---- Final chosen configuration (single source of truth) -------------------
VERSION = "v5_5_ru"
SPEAKER = "baya"
SAMPLE_RATE = 48000
SOXR_VIA_HZ = 20000              # anti-alias cutoff
DEESS = (5500.0, 0.06, 5.0)     # (cutoff_hz, threshold, ratio) — grid "s2"
CROSSFADE_MS = 20.0
_WARMUP = ["Прогрев.", "Короткий прогрев модели синтеза речи.",
           "Более длинное предложение для прогрева всех веток компиляции модели."]


class EnrollmentVoice:
    """Production Silero voice for the enrollment assistant (final config baked in)."""

    def __init__(
        self,
        device: str = "cuda",
        use_accent: bool = True,
        version: str = VERSION,
        speaker: str = SPEAKER,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.device = device
        self.use_accent = use_accent
        self.version = version
        self.speaker = speaker
        self.sample_rate = sample_rate
        self._model = None
        self._torch = None
        self._accentor = None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        """Load the bundle, RUAccent (optional), and warm up the JIT."""
        import torch
        self._torch = torch
        self._model, _ = torch.hub.load(
            "snakers4/silero-models", "silero_tts",
            language="ru", speaker=self.version, trust_repo=True,
        )
        self._model.to(self.device)

        if self.use_accent:
            from enhance import AccentProcessor
            acc = AccentProcessor()
            if acc.available:
                self._accentor = acc

        # Warm up short/medium/long so the first real call isn't hit by JIT.
        for w in _WARMUP:
            try:
                self._raw_tts(text=w)
            except Exception:  # noqa: BLE001
                pass

    # -- internals ----------------------------------------------------------
    def _raw_tts(self, text: str | None = None, ssml_text: str | None = None) -> np.ndarray:
        """Call Silero, return float32 mono at self.sample_rate (pre-post)."""
        with self._torch.no_grad():
            if ssml_text is not None:
                wav = self._model.apply_tts(
                    ssml_text=ssml_text, speaker=self.speaker, sample_rate=self.sample_rate
                )
            else:
                wav = self._model.apply_tts(
                    text=text, speaker=self.speaker, sample_rate=self.sample_rate,
                    put_accent=True, put_yo=True,
                )
        return wav.squeeze().cpu().numpy().astype(np.float32)

    def _post(self, audio: np.ndarray) -> np.ndarray:
        """The chosen post chain: soxr anti-alias → de-esser → peak-normalize."""
        audio = soxr_smooth(audio, self.sample_rate, via_hz=SOXR_VIA_HZ)
        audio = deesser(audio, self.sample_rate, *DEESS)
        return peak_normalize(audio, -1.0)

    def _accent(self, text: str) -> str:
        return self._accentor.process(text) if self._accentor is not None else text

    # -- public API ---------------------------------------------------------
    def synthesize_f32(self, text: str) -> np.ndarray:
        """text → post-processed float32 mono (accent applied if enabled)."""
        if self._model is None:
            raise RuntimeError("Call load() first.")
        return self._post(self._raw_tts(text=self._accent(text)))

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """text → (int16 PCM bytes, sample_rate). The main production entry point."""
        return f32_to_pcm_bytes(self.synthesize_f32(text)), self.sample_rate

    def synthesize_ssml(self, marked_text: str) -> tuple[bytes, int]:
        """Marker-annotated text ([q], [emp], [pause]…) → (int16 PCM, sample_rate).

        Uses intonation_ssml to build Silero SSML (verified working on v5_5_ru).
        NOTE: RUAccent is not applied in SSML mode — Silero's own accenting runs;
        pre-accent the source text if you need dictionary overrides here.
        """
        if self._model is None:
            raise RuntimeError("Call load() first.")
        body = markers_to_ssml(marked_text)
        ssml = f"<speak><p><s>{body}</s></p></speak>"
        return f32_to_pcm_bytes(self._post(self._raw_tts(ssml_text=ssml))), self.sample_rate

    def synthesize_stream(self, sentences: list[str]) -> tuple[bytes, int]:
        """Synthesize each sentence, crossfade the seams → (int16 PCM, sample_rate).

        For the LLM∥TTS path: feed sentences as the LLM emits them; each is
        post-processed independently, then joined with a raised-cosine crossfade
        so seams don't click.
        """
        if self._model is None:
            raise RuntimeError("Call load() first.")
        chunks = [self.synthesize_f32(s) for s in sentences if s.strip()]
        stitched = crossfade(chunks, self.sample_rate, ms=CROSSFADE_MS)
        return f32_to_pcm_bytes(stitched), self.sample_rate


if __name__ == "__main__":
    v = EnrollmentVoice()
    t0 = time.perf_counter()
    v.load()
    print(f"loaded+warmed in {time.perf_counter() - t0:.1f}s")
    pcm, sr = v.synthesize("Добрый день! Чем могу помочь с поступлением?")
    from audio_utils import save_wav_from_bytes
    save_wav_from_bytes(pcm, "grid_audio/production_check.wav", sr)
    print(f"wrote grid_audio/production_check.wav ({len(pcm)} bytes @ {sr} Hz)")
