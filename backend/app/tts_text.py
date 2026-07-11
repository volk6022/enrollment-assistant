from __future__ import annotations

import re

_PREFIXES = [
    r"Основание:\s*",
    r"Практически:\s*",
    r"Краткий ответ:\s*",
    r"Подтверждение:\s*",
]


def _clean(text: str) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    for p in _PREFIXES:
        value = re.sub(p, "", value, flags=re.IGNORECASE)
    value = re.sub(r"\([^)]*п\.\s*\d[^)]*\)", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def build_tts_text(answer: str, fallback: str = "") -> str:
    text = _clean(answer or fallback)
    if not text:
        return "Не удалось подготовить ответ для озвучивания."
    parts = re.split(r"(?<=[.!?])\s+", text)
    short = " ".join(parts[:2]).strip()
    if len(short) > 320:
        short = short[:317].rsplit(" ", 1)[0] + "..."
    return short
