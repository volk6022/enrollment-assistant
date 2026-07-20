from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    backend_url: str = os.getenv("BACKEND_URL", "http://backend:8000")
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    speechkit_api_key: str = os.getenv("YC_SPEECHKIT_API_KEY", "")
    folder_id: str = os.getenv("YC_FOLDER_ID", "")
    force_tts_folder_id: bool = os.getenv("YC_FORCE_TTS_FOLDER_ID", "false").lower() == "true"
    stt_lang: str = os.getenv("YC_SPEECHKIT_LANG", "ru-RU")
    stt_model: str = os.getenv("YC_SPEECHKIT_STT_MODEL", "general")
    # STT backend: "faster-whisper" (local, default) or "yandex" (SpeechKit fallback).
    stt_backend: str = os.getenv("VOICE_STT_BACKEND", "faster-whisper").lower()
    whisper_model: str = os.getenv("VOICE_WHISPER_MODEL", "large-v3-turbo")
    whisper_device: str = os.getenv("VOICE_WHISPER_DEVICE", "cuda")
    whisper_compute_type: str = os.getenv("VOICE_WHISPER_COMPUTE_TYPE", "int8")
    whisper_lang: str = os.getenv("VOICE_WHISPER_LANG", "ru")
    whisper_download_root: str = os.getenv("VOICE_WHISPER_DOWNLOAD_ROOT", "")
    tts_voice: str = os.getenv("YC_TTS_VOICE", "alena")
    tts_role: str = os.getenv("YC_TTS_ROLE", "neutral")
    tts_speed: str = os.getenv("YC_TTS_SPEED", "1.0")
    tts_format: str = os.getenv("YC_TTS_FORMAT", "mp3")
    tts_sample_rate_hz: str = os.getenv("YC_TTS_SAMPLE_RATE_HZ", "48000")
    # TTS backend: "silero" (local, default) or "yandex" (SpeechKit cloud fallback).
    tts_backend: str = os.getenv("VOICE_TTS_BACKEND", "silero").lower()
    # Local Silero config (chosen by the TTS experiment; see stt-tts-research-2026-07).
    silero_version: str = os.getenv("VOICE_SILERO_VERSION", "v5_5_ru")
    silero_speaker: str = os.getenv("VOICE_SILERO_SPEAKER", "baya")
    silero_sample_rate_hz: str = os.getenv("VOICE_SILERO_SAMPLE_RATE_HZ", "48000")
    silero_device: str = os.getenv("VOICE_TTS_DEVICE", "cpu")  # "cuda" if a GPU is present
    silero_use_accent: bool = os.getenv("VOICE_SILERO_USE_ACCENT", "true").lower() == "true"
    silero_cache_dir: str = os.getenv("VOICE_SILERO_CACHE_DIR", "")  # torch.hub dir; "" = default
    cache_dir: str = os.getenv("VOICE_CACHE_DIR", "/voice-cache")
    mode: str = os.getenv("VOICE_GATEWAY_MODE", "demo")
    public_base_url: str = os.getenv("VOICE_PUBLIC_BASE_URL", "http://localhost:8010")
    backend_retries: int = int(os.getenv("BACKEND_RETRIES", "12"))
    backend_retry_delay_sec: float = float(os.getenv("BACKEND_RETRY_DELAY_SEC", "1.0"))


settings = Settings()
