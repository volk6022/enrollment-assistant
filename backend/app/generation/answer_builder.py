from __future__ import annotations

from typing import Any

from ..intents.schemas import IntentContext
from .brief_composer import BriefAnswerComposer


_POLICY_INTENTS = {
    "citizenship", "health", "ege_validity", "after_9th_grade", "paid_education", "age_limits",
    "second_degree", "gender", "dormitory", "pass_score", "relatives_record", "where_apply",
    "transfer", "accelerated",
}


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _brief_policy_answer(extracted: dict[str, Any]) -> str:
    direct = (extracted.get("direct_answer") or "").strip()
    key_points = _unique([x for x in (extracted.get("key_points") or []) if isinstance(x, str)])
    conditions = _unique([x for x in (extracted.get("conditions") or []) if isinstance(x, str)])
    parts: list[str] = []
    if direct:
        parts.append(direct)
    if key_points:
        parts.append("; ".join(key_points[:3]))
    if conditions:
        parts.append("Условия: " + "; ".join(conditions[:2]))
    return " ".join(parts).strip() or "Не удалось собрать краткий ответ по найденным данным."


def _heuristic_answer(ctx: IntentContext, extracted: dict[str, Any]) -> str:
    intent = ctx.intent
    if intent == "programs_by_form":
        programs = extracted.get("programs") or []
        form = extracted.get("form") or ctx.study_form or "этой форме обучения"
        restrictions = extracted.get("restrictions") or []
        if programs:
            parts = []
            for item in programs[:6]:
                code = item.get("code", "").strip()
                name = item.get("name", "").strip()
                profile = item.get("profile", "").strip()
                duration = item.get("duration", "").strip()
                chunk = " ".join(x for x in [code, name] if x)
                if profile:
                    chunk += f" — {profile}"
                if duration:
                    chunk += f" ({duration})"
                if chunk:
                    parts.append(chunk)
            answer = f"На {form} набор ведется по следующим программам: " + "; ".join(parts) + "."
            if restrictions:
                answer += " Важно: " + "; ".join(restrictions[:2]) + "."
            return answer
        return "Не удалось извлечь полный список программ по указанной форме обучения."

    if intent == "min_scores":
        scores = extracted.get("scores") or []
        if scores:
            parts = [f"{item['subject']} — {item['min_score']}" for item in scores if item.get('subject') and item.get('min_score') is not None]
            if parts:
                return "Минимальные баллы для поступления: " + "; ".join(parts[:6]) + "."
        return "Не удалось надежно извлечь полный список минимальных баллов по предметам."

    if intent == "exams":
        req = extracted.get("required_ege") or []
        extra = extracted.get("extra_exams") or []
        parts = []
        if req:
            parts.append("обязательные ЕГЭ: " + "; ".join([f"{x.get('subject')}" + (f" — {x.get('min_score')}" if x.get('min_score') is not None else "") for x in req[:5] if x.get('subject')]))
        if extra:
            parts.append("дополнительные испытания: " + "; ".join([f"{x.get('subject')}" + (f" — {x.get('format')}" if x.get('format') else "") + (f", минимум {x.get('min_score')}" if x.get('min_score') is not None else "") for x in extra[:5] if x.get('subject')]))
        if parts:
            return "Для поступления нужны следующие испытания: " + ". ".join(parts) + "."
        return "Не удалось извлечь полный перечень ЕГЭ и дополнительных испытаний."

    if intent == "documents":
        docs = extracted.get("documents") or []
        if docs:
            parts = [item.get("name", "") for item in docs if item.get("name")]
            return "Для поступления обычно требуются следующие документы: " + "; ".join(parts[:10]) + "."
        return "Не удалось извлечь полный перечень документов для поступления."

    if intent == "apply":
        methods = extracted.get("methods") or []
        if methods:
            parts = [item.get("name", "") for item in methods if item.get("allowed") and item.get("name")]
            if parts:
                return "Подать документы можно следующими способами: " + "; ".join(parts) + "."
        return "Не удалось извлечь полный перечень способов подачи документов."

    if intent == "without_ege":
        cases = extracted.get("cases") or []
        if cases:
            labels = [c.get("category", "") for c in cases if c.get("category")]
            labels = _unique(labels)
            if labels:
                return "Поступление без ЕГЭ возможно не для всех. Оно допускается в таких случаях: " + "; ".join(labels[:4]) + "."
        return "Поступление без ЕГЭ возможно только в случаях, прямо указанных в правилах приема и порядке приема."

    if intent == "benefits":
        items = extracted.get("benefits") or []
        if items:
            names = [x.get("name", "") for x in items if x.get("name")]
            names = _unique(names)
            if names:
                return "Льготы и особые права определяются правилами приема и профильными нормами. В найденных данных упоминаются: " + "; ".join(names[:5]) + "."
        return "Льготы и особые права определяются правилами приема и профильными нормативными актами."

    if intent in _POLICY_INTENTS:
        return _brief_policy_answer(extracted)

    return (extracted.get("summary") or "Не удалось собрать ответ по найденным данным.").strip()


def build_answer(ctx: IntentContext, extracted: dict[str, Any], citations: list[dict[str, Any]]) -> dict[str, Any]:
    heuristic = _heuristic_answer(ctx, extracted)
    composer = BriefAnswerComposer()
    answer = composer.compose(ctx, extracted, heuristic)
    tts_text = build_tts_text(ctx, extracted, answer)
    meta = {
        "engine": "yandex_v11",
        "question_intent": ctx.intent,
        "question_profile": ctx.profile,
        "notes": ctx.notes,
        "backend_engine": "yandex_v11",
        "answer_style": "brief_direct",
    }
    if ctx.needs_optional_clarification:
        meta["needs_optional_clarification"] = True
    return {
        "answer": answer,
        "voice_answer": tts_text,
        "tts_text": tts_text,
        "citations": citations,
        "meta": meta,
        "need_clarification": False,
        "clarification_type": None,
    }


def build_tts_text(ctx: IntentContext, extracted: dict[str, Any], answer: str) -> str:
    if ctx.intent == "programs_by_form" and extracted.get("programs"):
        names = []
        for item in extracted["programs"][:4]:
            code = item.get("code")
            name = item.get("name")
            if code and name:
                names.append(f"{code} {name}")
            elif name:
                names.append(name)
        if names:
            return "Есть следующие программы: " + "; ".join(names) + "."
    if ctx.intent == "min_scores" and extracted.get("scores"):
        parts = [f"{item['subject']} {item['min_score']}" for item in extracted['scores'][:4] if item.get('subject') and item.get('min_score') is not None]
        if parts:
            return "Минимальные баллы: " + "; ".join(parts) + "."
    if ctx.intent == "documents" and extracted.get("documents"):
        parts = [item.get("name", "") for item in extracted["documents"][:5] if item.get("name")]
        if parts:
            return "Обычно нужны: " + "; ".join(parts) + "."
    if ctx.intent == "apply" and extracted.get("methods"):
        parts = [item.get("name", "") for item in extracted["methods"] if item.get("allowed") and item.get("name")]
        if parts:
            return "Документы можно подать так: " + "; ".join(parts) + "."
    return answer[:500]
