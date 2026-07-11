from __future__ import annotations

from typing import Any

import httpx

from ..config import settings


class YandexVectorStoreAdminClient:
    def __init__(self) -> None:
        self.base_url = settings.yandex_vector_store_api_base.rstrip("/")
        self.api_key = settings.yandex_ai_api_key
        self.folder_id = settings.yandex_folder_id
        self.timeout = settings.yandex_timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Api-Key {self.api_key}", "x-folder-id": self.folder_id, "Content-Type": "application/json"}

    def create(self, name: str, file_ids: list[str] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name}
        if file_ids:
            body["fileIds"] = file_ids
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.base_url, headers=self._headers(), json=body)
            resp.raise_for_status()
            return resp.json()

    def attach_files(self, vector_store_id: str, file_ids: list[str]) -> dict[str, Any]:
        url = f"{self.base_url}/{vector_store_id}/files:batchAdd"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._headers(), json={"fileIds": file_ids})
            resp.raise_for_status()
            return resp.json()
