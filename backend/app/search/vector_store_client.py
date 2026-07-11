from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ..config import settings


@dataclass
class SearchHit:
    text: str
    score: float
    payload: dict[str, Any]


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            chunk = _to_text(item)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "body", "chunk"):
            if key in value:
                return _to_text(value.get(key))
        parts: list[str] = []
        for v in value.values():
            chunk = _to_text(v)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts)
    return str(value)


class YandexVectorStoreClient:
    def __init__(self) -> None:
        self.api_key = settings.yandex_ai_api_key
        self.base_url = settings.yandex_vector_store_api_base.rstrip("/")
        self.vector_store_id = settings.yandex_vector_store_id
        self.timeout = settings.yandex_timeout_seconds
        self.folder_id = settings.yandex_folder_id

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Api-Key {self.api_key}", "Content-Type": "application/json"}
        if self.folder_id:
            headers["x-folder-id"] = self.folder_id
        return headers

    def search(self, query: str, *, top_k: int = 6, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        url = f"{self.base_url}/vector_stores/{self.vector_store_id}/search"
        body: dict[str, Any] = {"query": query, "max_num_results": top_k}
        if filters:
            body["filter"] = filters
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._headers(), json=body)
            resp.raise_for_status()
            data = resp.json()
        out: list[SearchHit] = []
        for item in data.get("chunks", data.get("data", data.get("results", []))):
            payload = item.get("attributes", {}) or item.get("payload", {}) or {}
            raw_text = item.get("text")
            if raw_text is None:
                raw_text = item.get("content", "")
            out.append(
                SearchHit(
                    text=_to_text(raw_text),
                    score=float(item.get("score", 0.0)),
                    payload=payload,
                )
            )
        return out
