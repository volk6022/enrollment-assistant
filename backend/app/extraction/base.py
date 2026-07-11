from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..intents.schemas import IntentContext
from ..search.vector_store_client import SearchHit
from ..yandex.responses_client import YandexResponsesClient


@dataclass
class RetrievalPacket:
    context: IntentContext
    queries: list[str]
    hits: list[SearchHit]
    sections: list[str]


class BaseExtractor:
    intent: str = "general"
    schema: dict[str, Any] = {"type": "object"}

    def __init__(self) -> None:
        self._client = YandexResponsesClient(model=settings.yandex_model_pro, timeout=settings.yandex_extract_timeout_seconds) if settings.yandex_enabled else None

    def _joined_sections(self, packet: RetrievalPacket) -> str:
        return "\n\n".join(packet.sections[:6])

    def extract(self, packet: RetrievalPacket) -> dict[str, Any]:
        if self._client is None:
            return self._fallback(packet)
        prompt = self._build_prompt(packet)
        try:
            data = self._client.simple_json(prompt, self.schema)
            if isinstance(data, dict):
                data.setdefault("source_points", self._source_points(packet))
                return data
        except Exception:
            pass
        return self._fallback(packet)

    def _source_points(self, packet: RetrievalPacket) -> list[str]:
        out = []
        for hit in packet.hits[:5]:
            point = hit.payload.get("point")
            source = hit.payload.get("source")
            if source and point:
                out.append(f"{source} (п. {point})")
        return out

    def _build_prompt(self, packet: RetrievalPacket) -> str:
        return (
            f"Извлеки структурированные данные для intent={self.intent}. "
            "Опирайся только на приведенные фрагменты документов. "
            f"\nВопрос: {packet.context.raw_question}\n\nФрагменты:\n{self._joined_sections(packet)}"
        )

    def _fallback(self, packet: RetrievalPacket) -> dict[str, Any]:
        return {"summary": self._joined_sections(packet)[:1200], "source_points": self._source_points(packet)}


def flatten_text(packet: RetrievalPacket) -> str:
    return "\n".join(packet.sections)


def extract_number_pairs(text: str) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for line in text.splitlines():
        m = re.search(r"([А-ЯA-ZЁ][^\n:]{2,80})[:\-]\s*(\d{2,3})\b", line)
        if m:
            pairs.append((m.group(1).strip(), int(m.group(2))))
    return pairs
