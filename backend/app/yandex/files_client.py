from __future__ import annotations

from pathlib import Path

import httpx

from ..config import settings


class YandexFilesClient:
    def __init__(self) -> None:
        self.base_url = settings.yandex_files_api_base.rstrip("/")
        self.api_key = settings.yandex_ai_api_key
        self.folder_id = settings.yandex_folder_id
        self.timeout = settings.yandex_timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Api-Key {self.api_key}", "x-folder-id": self.folder_id}

    def upload(self, path: str | Path, purpose: str = "assistants") -> dict:
        p = Path(path)
        with p.open("rb") as fh, httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.base_url,
                headers=self._headers(),
                data={"purpose": purpose},
                files={"file": (p.name, fh, "application/octet-stream")},
            )
            resp.raise_for_status()
            return resp.json()
