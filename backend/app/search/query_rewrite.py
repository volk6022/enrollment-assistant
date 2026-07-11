from __future__ import annotations

from ..config import settings
from ..intents.schemas import IntentContext
from ..yandex.responses_client import YandexResponsesClient


class QueryRewriter:
    def __init__(self) -> None:
        self._client = YandexResponsesClient(model=settings.yandex_model_lite) if settings.yandex_enabled else None

    def expand(self, ctx: IntentContext, profile_expansions: list[str]) -> list[str]:
        queries = [ctx.raw_question]
        queries.extend(profile_expansions)
        if self._client is not None:
            prompt = (
                "Перефразируй вопрос для поиска по нормативным документам. Верни JSON-массив из 2 коротких запросов без пояснений."
                f"\nВопрос: {ctx.raw_question}"
            )
            try:
                payload = self._client.simple_json(prompt, schema={"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2})
                if isinstance(payload, list):
                    queries.extend([q for q in payload if isinstance(q, str) and q.strip()])
                    ctx.notes.append("rewrite:yandex")
            except Exception as exc:
                ctx.notes.append(f"rewrite_fallback:{type(exc).__name__}")
        uniq: list[str] = []
        seen = set()
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                uniq.append(q.strip())
        return uniq[:5]
