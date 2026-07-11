from __future__ import annotations

from typing import Final

from ..config import settings
from .responses_client import YandexResponsesClient


INTENTS: Final[list[str]] = [
    "contacts",
    "programs_by_form",
    "min_scores",
    "without_ege",
    "benefits",
    "documents",
    "deadlines",
    "apply",
    "exams",
    "eligibility",
    "citizenship",
    "after_9th_grade",
    "paid_education",
    "age_limits",
    "second_degree",
    "gender",
    "dormitory",
    "pass_score",
    "relatives_record",
    "where_apply",
    "transfer",
    "accelerated",
    "ege_validity",
    "normative",
    "general",
]


class YandexClassifierClient:
    def __init__(self) -> None:
        self._client = YandexResponsesClient(model=settings.yandex_model_lite)

    def classify(self, question: str) -> str | None:
        schema = {"type": "object", "properties": {"intent": {"type": "string", "enum": INTENTS}}, "required": ["intent"], "additionalProperties": False}
        data = self._client.simple_json(
            "Определи intent запроса приемной комиссии. Допустимые классы: " + ", ".join(INTENTS) + f".\nЗапрос: {question}",
            schema=schema,
        )
        if isinstance(data, dict):
            intent = data.get("intent")
            if intent in INTENTS:
                return intent
        return None
