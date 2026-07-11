from __future__ import annotations

import requests

from .config import settings


class SpeechKitShortAudioSTTClient:
    """Synchronous STT for short demo/test audio.

    Real phone calls should use SpeechKit STT API v3 streaming via gRPC. The
    project includes a stub-generation script for that path, but this client is
    useful for local testing and non-live demos.
    """

    endpoint = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"

    def __init__(self) -> None:
        self.api_key = settings.speechkit_api_key
        self.folder_id = settings.folder_id
        self.lang = settings.stt_lang
        self.model = settings.stt_model

    def recognize_bytes(self, audio_bytes: bytes, *, lang: str | None = None, sample_rate_hertz: int = 8000, audio_format: str = "lpcm") -> str:
        if not self.api_key or not self.folder_id:
            raise RuntimeError("SpeechKit API key or folder ID is not configured")
        params = {
            "folderId": self.folder_id,
            "lang": lang or self.lang,
            "format": audio_format,
            "sampleRateHertz": sample_rate_hertz,
            "model": self.model,
        }
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/octet-stream",
        }
        resp = requests.post(self.endpoint, params=params, headers=headers, data=audio_bytes, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        return (payload.get("result") or "").strip()
