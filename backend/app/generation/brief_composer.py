from __future__ import annotations

import json
from typing import Any

from ..config import settings
from ..intents.schemas import IntentContext
from ..yandex.responses_client import YandexResponsesClient


class BriefAnswerComposer:
    def __init__(self) -> None:
        self._client = YandexResponsesClient(model=settings.yandex_model_lite, timeout=settings.yandex_timeout_seconds) if settings.yandex_enabled else None

    def compose(self, ctx: IntentContext, extracted: dict[str, Any], fallback_answer: str) -> str:
        if self._client is None:
            return fallback_answer
        prompt = (
            "Ты помощник приемной комиссии. Сформулируй ответ кратко, по делу и человеческим языком. "
            "Не говори 'по найденным фрагментам', 'основание', 'смотрите пункты'. "
            "Не цитируй документы. Отвечай в 1-4 предложениях. Если вопрос Да/Нет, начни с 'Да' или 'Нет'. "
            "Если в данных есть перечень, перечисли его компактно через двоеточие или точки с запятой. "
            f"\nIntent: {ctx.intent}\nВопрос: {ctx.raw_question}\nИзвлеченные данные JSON:\n{json.dumps(extracted, ensure_ascii=False)}\n\n"
            f"Черновой ответ: {fallback_answer}"
        )
        try:
            payload = self._client.create(
                system_prompt="Сделай только финальный короткий ответ без markdown.",
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=320,
            )
            text = self._client.extract_text(payload).strip()
            if text:
                return text
        except Exception:
            pass
        return fallback_answer
