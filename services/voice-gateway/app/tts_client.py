from __future__ import annotations

import requests

from .audio_codec import pcm16_to_wav_bytes
from .config import settings


class SpeechKitTTSClient:
    endpoint = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"

    def __init__(self) -> None:
        self.api_key = settings.speechkit_api_key
        self.folder_id = settings.folder_id
        self.voice = settings.tts_voice
        self.emotion = settings.tts_role
        self.speed = settings.tts_speed
        self.audio_format = settings.tts_format
        self.sample_rate_hertz = settings.tts_sample_rate_hz

    def _request(self, text: str, fmt: str, *, with_emotion: bool = True) -> bytes:
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        data = {
            "text": text,
            "voice": self.voice,
            "speed": self.speed,
            "format": fmt,
        }
        # For SpeechKit API v1 with service-account API key, folderId is not required.
        # Keep backward compatibility if someone explicitly wants to send it.
        if self.folder_id and settings.force_tts_folder_id:
            data["folderId"] = self.folder_id
        if with_emotion and self.emotion:
            data["emotion"] = self.emotion
        if fmt == "lpcm" and self.sample_rate_hertz:
            data["sampleRateHertz"] = self.sample_rate_hertz
        resp = requests.post(self.endpoint, headers=headers, data=data, timeout=120)
        if resp.ok:
            return resp.content
        raise RuntimeError(f"TTS {resp.status_code}: {resp.text}")

    def synthesize(self, text: str) -> tuple[bytes, str, str]:
        if not self.api_key:
            raise RuntimeError("SpeechKit API key is not configured")
        text = (text or "").strip()
        if not text:
            raise RuntimeError("Empty TTS text")

        fmt = (self.audio_format or "mp3").lower()
        if fmt not in {"mp3", "oggopus", "lpcm"}:
            fmt = "mp3"

        try:
            raw = self._request(text, fmt, with_emotion=True)
        except Exception as exc:
            # Some voices/configs reject emotion. Retry without it before failing.
            if self.emotion:
                raw = self._request(text, fmt, with_emotion=False)
            else:
                raise exc

        if fmt == "mp3":
            return raw, ".mp3", "audio/mpeg"
        if fmt == "oggopus":
            return raw, ".ogg", "audio/ogg"
        wav = pcm16_to_wav_bytes(raw, sample_rate_hz=int(self.sample_rate_hertz or 48000))
        return wav, ".wav", "audio/wav"
