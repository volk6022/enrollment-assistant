from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import settings


class YandexResponsesClient:
    def __init__(self, *, model: str | None = None, timeout: int | None = None) -> None:
        self.model = model or settings.yandex_model_pro
        self.endpoint = settings.yandex_responses_endpoint
        self.api_key = settings.yandex_ai_api_key
        self.folder_id = settings.yandex_folder_id
        self.timeout = timeout or settings.yandex_timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id,
            "Content-Type": "application/json",
        }

    def create(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None = None, temperature: float = 0.1, max_tokens: int = 1200) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if schema:
            body["text"] = {"format": {"type": "json_schema", "schema": schema}}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.endpoint, headers=self._headers(), json=body)
            resp.raise_for_status()
            return resp.json()

    def extract_text(self, payload: dict[str, Any]) -> str:
        output = payload.get("output", [])
        for item in output:
            content = item.get("content", [])
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                    return part.get("text", "")
        if isinstance(payload.get("result"), dict):
            alternatives = payload["result"].get("alternatives", [])
            if alternatives:
                return alternatives[0].get("message", {}).get("text", "")
        return payload.get("text", "")

    def simple_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        payload = self.create(
            system_prompt="Верни только JSON по заданной схеме без markdown и пояснений.",
            user_prompt=prompt,
            schema=schema,
            temperature=0.0,
        )
        text = self.extract_text(payload).strip()
        return json.loads(text)
