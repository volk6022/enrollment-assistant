from __future__ import annotations

import re

from .schemas import IntentContext
from ..config import settings
from ..knowledge import infer_question_intent
from ..legal_hierarchy import question_profile
from ..main_helpers import parse_form, parse_level, parse_status
from ..yandex.classifier_client import YandexClassifierClient


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _heuristic_overrides(question: str, current_intent: str, study_form: str | None) -> tuple[str, list[str]]:
    q = _norm(question)
    notes: list[str] = []

    def has(*parts: str) -> bool:
        return all(p in q for p in parts)

    if any(k in q for k in ["очная", "заочная", "очн", "заочн", "дневн"]) and any(k in q for k in ["специальност", "направлен", "программ", "набор", "поступить", "есть у вас", "что на ней", "что есть", "осуществляется ли набор", "ведется ли набор"]):
        notes.append("override:programs_by_form")
        return "programs_by_form", notes

    if ("егэ" in q and any(k in q for k in ["обязатель", "дополнитель", "испытан", "экзам", "физподготов"])):
        notes.append("override:exams")
        return "exams", notes

    if any(k in q for k in ["гражданств", "двойн", "паспорт рф", "не гражданин"]) :
        notes.append("override:citizenship")
        return "citizenship", notes

    if any(k in q for k in ["здоров", "ввк", "медкомис", "военно врачеб", "болезн"]) :
        notes.append("override:health")
        return "health", notes

    if has("после", "9") or "9 класс" in q or "девять клас" in q:
        notes.append("override:after_9th_grade")
        return "after_9th_grade", notes

    if any(k in q for k in ["платн", "договорн", "коммерц", "внебюджет"]) :
        notes.append("override:paid_education")
        return "paid_education", notes

    if any(k in q for k in ["возраст", "до скольки", "до какого возраста", "сколько лет"]) :
        notes.append("override:age_limits")
        return "age_limits", notes

    if any(k in q for k in ["второе высшее", "второго высшего", "еще одно высшее"]) :
        notes.append("override:second_degree")
        return "second_degree", notes

    if any(k in q for k in ["девуш", "женщин", "девоч", "девушкам"]) and any(k in q for k in ["принима", "поступ", "берут", "общежит", "казарм"]):
        if any(k in q for k in ["общежит", "казарм", "проживан"]):
            notes.append("override:dormitory")
            return "dormitory", notes
        notes.append("override:gender")
        return "gender", notes

    if any(k in q for k in ["общежит", "казарм", "проживан"]) :
        notes.append("override:dormitory")
        return "dormitory", notes

    if any(k in q for k in ["проходн", "шанс", "конкурс", "реально поступить"]) :
        notes.append("override:pass_score")
        return "pass_score", notes

    if any(k in q for k in ["судим", "родствен", "штраф", "административк"]) and any(k in q for k in ["поступить", "могу ли"]):
        notes.append("override:relatives_record")
        return "relatives_record", notes

    if any(k in q for k in ["куда подав", "куда подать", "где подав", "заявление", "прямой набор", "кадровое подразделение"]) :
        notes.append("override:where_apply")
        return "where_apply", notes

    if any(k in q for k in ["перевест", "перевод из другого вуза", "из другого вуза"]) :
        notes.append("override:transfer")
        return "transfer", notes

    if any(k in q for k in ["сокращенн", "ускоренн"]) :
        notes.append("override:accelerated")
        return "accelerated", notes

    if "егэ" in q and any(k in q for k in ["действ", "каких годов", "какие результаты", "срок действ"]):
        notes.append("override:ege_validity")
        return "ege_validity", notes

    if current_intent == "eligibility" and study_form in {"очная", "заочная"} and any(k in q for k in ["специальност", "направлен", "программ"]):
        notes.append("override:programs_by_form")
        return "programs_by_form", notes

    return current_intent, notes


class IntentClassifier:
    def __init__(self) -> None:
        self._client = YandexClassifierClient() if settings.yandex_enabled else None

    def classify(self, question: str) -> IntentContext:
        level = parse_level(question)
        status = parse_status(question)
        study_form = parse_form(question)
        intent = infer_question_intent(question)
        profile = question_profile(question)
        notes: list[str] = []
        if self._client is not None:
            try:
                predicted = self._client.classify(question)
                if predicted:
                    intent = predicted
                    notes.append("classifier:yandex")
            except Exception as exc:
                notes.append(f"classifier_fallback:{type(exc).__name__}")
        intent, override_notes = _heuristic_overrides(question, intent, study_form)
        notes.extend(override_notes)
        if intent == "where_apply":
            profile = "organizational"
        needs_optional_clarification = intent in {"without_ege", "benefits", "eligibility", "exams", "min_scores", "programs_by_form"} and not level and intent not in {"programs_by_form"}
        return IntentContext(
            raw_question=question,
            intent=intent,
            profile=profile,
            level=level,
            status=status,
            study_form=study_form,
            needs_optional_clarification=needs_optional_clarification,
            rewritten_query=question,
            notes=notes,
        )
