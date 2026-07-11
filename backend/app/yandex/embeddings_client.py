from __future__ import annotations

from typing import Any

import httpx

from ..config import settings


class YandexEmbeddingsClient:
    def __init__(self) -> None:
        self.endpoint = settings.yandex_embeddings_endpoint
        self.api_key = settings.yandex_ai_api_key
        self.folder_id = settings.yandex_folder_id
        self.model = settings.yandex_embedding_model
        self.timeout = settings.yandex_timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Api-Key {self.api_key}", "x-folder-id": self.folder_id, "Content-Type": "application/json"}

    def embed(self, text: str) -> list[float]:
        body: dict[str, Any] = {"model": self.model, "text": text}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.endpoint, headers=self._headers(), json=body)
            resp.raise_for_status()
            data = resp.json()
        return data.get("embedding", data.get("vector", []))
