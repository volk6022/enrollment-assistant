from __future__ import annotations

from typing import Any, Dict


RISKY_INTENTS = {"without_ege", "benefits", "eligibility", "min_scores"}


def should_recommend_handoff(result: Dict[str, Any]) -> bool:
    meta = result.get("meta") or {}
    if result.get("need_clarification") and meta.get("question_intent") in RISKY_INTENTS:
        return True
    if float(meta.get("grounding_confidence") or 0.0) < 0.35 and meta.get("engine") in {"grounded", "structured"}:
        return True
    text = (result.get("answer") or "").lower()
    if any(marker in text for marker in ["не удалось", "не найден", "недостаточно данных", "нужно уточнить категорию"]):
        return True
    return False
