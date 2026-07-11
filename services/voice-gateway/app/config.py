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
    tts_voice: str = os.getenv("YC_TTS_VOICE", "alena")
    tts_role: str = os.getenv("YC_TTS_ROLE", "neutral")
    tts_speed: str = os.getenv("YC_TTS_SPEED", "1.0")
    tts_format: str = os.getenv("YC_TTS_FORMAT", "mp3")
    tts_sample_rate_hz: str = os.getenv("YC_TTS_SAMPLE_RATE_HZ", "48000")
    cache_dir: str = os.getenv("VOICE_CACHE_DIR", "/voice-cache")
    mode: str = os.getenv("VOICE_GATEWAY_MODE", "demo")
    public_base_url: str = os.getenv("VOICE_PUBLIC_BASE_URL", "http://localhost:8010")
    backend_retries: int = int(os.getenv("BACKEND_RETRIES", "12"))
    backend_retry_delay_sec: float = float(os.getenv("BACKEND_RETRY_DELAY_SEC", "1.0"))


settings = Settings()
