"""Local faster-whisper STT client for the voice gateway (replaces Yandex STT).

Drop-in for SpeechKitShortAudioSTTClient: exposes
`recognize_bytes(audio_bytes, *, lang, sample_rate_hertz, audio_format) -> str`.

Model chosen by the STT experiment (see 19-In-Work/stt-tts-research-2026-07.md):
faster-whisper **large-v3-turbo**, CT2 **int8** on **CUDA** — CER 6.7% / fair WER
12.8% on the RU eval, RTF ~0.10. Stable, well-understood architecture with a small
decoder (no resident general-purpose LLM in VRAM).

Input handling:
  - "lpcm"      raw 16-bit mono PCM at sample_rate_hertz → float32, resample to 16 kHz.
  - "mp3"/"oggopus"  compressed bytes → decoded by faster-whisper (PyAV/ffmpeg).

Self-contained apart from numpy/soxr/faster-whisper.
"""

from __future__ import annotations

import io

import numpy as np

from .config import settings

_WHISPER_SR = 16000  # Whisper always operates at 16 kHz mono


class FasterWhisperSTTClient:
    """Local Whisper transcription via CTranslate2 (faster-whisper)."""

    def __init__(self) -> None:
        self.model_size = settings.whisper_model
        self.device = settings.whisper_device
        self.compute_type = settings.whisper_compute_type
        self.default_lang = settings.whisper_lang
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        kwargs = {"device": self.device, "compute_type": self.compute_type}
        if settings.whisper_download_root:
            kwargs["download_root"] = settings.whisper_download_root
        self._model = WhisperModel(self.model_size, **kwargs)
        # Warm up CUDA kernels so the first real call isn't hit by compilation.
        try:
            warm = np.zeros(_WHISPER_SR, dtype=np.float32)  # 1 s of silence
            list(self._model.transcribe(warm, language=self.default_lang[:2], beam_size=1)[0])
        except Exception:  # noqa: BLE001
            pass

    def _lpcm_to_f32_16k(self, audio_bytes: bytes, sample_rate_hertz: int) -> np.ndarray:
        """Raw int16 mono PCM → float32 [-1, 1] mono @ 16 kHz."""
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate_hertz != _WHISPER_SR and len(pcm) > 0:
            import soxr
            pcm = soxr.resample(pcm, sample_rate_hertz, _WHISPER_SR).astype(np.float32)
        return pcm

    def recognize_bytes(
        self,
        audio_bytes: bytes,
        *,
        lang: str | None = None,
        sample_rate_hertz: int = 8000,
        audio_format: str = "lpcm",
    ) -> str:
        """Transcribe audio bytes → text. Signature matches the Yandex STT client."""
        self._ensure_loaded()
        fmt = (audio_format or "lpcm").lower()
        language = (lang or self.default_lang or "ru")[:2]

        if fmt == "lpcm":
            audio = self._lpcm_to_f32_16k(audio_bytes, sample_rate_hertz)
            if len(audio) == 0:
                return ""
            source = audio
        else:
            # mp3/oggopus/etc. — hand the container to faster-whisper to decode.
            source = io.BytesIO(audio_bytes)

        segments, _info = self._model.transcribe(
            source,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
