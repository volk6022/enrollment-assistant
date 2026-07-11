import os
import re
from typing import Any, Dict, List, Tuple

import requests

from .knowledge import infer_question_intent, intent_keywords, question_requires_strict_grounding
from .legal_hierarchy import question_profile


SYSTEM_PROMPT = """Ты — официальный помощник абитуриента юридического вуза.
Отвечай только на основе переданных фрагментов и не додумывай факты.

Правила:
- сначала дай прямой ответ обычным человеческим языком;
- используй только то, что подтверждено фрагментами;
- если фрагментов недостаточно, прямо скажи, что точный ответ не подтвержден текущей выдачей;
- не подмешивай нормы для адъюнктуры и аспирантуры в общий вопрос, если пользователь их не спрашивал;
- не путай минимальные баллы ЕГЭ с дополнительными баллами за достижения;
- не подменяй порядок приема общими законами и побочными документами.

Формат строго:
VOICE: <1 короткая фраза>
USED_IDS: <id через запятую>
ANSWER: <нормальный краткий ответ без заголовков>
"""

SUBJECT_NAMES = [
    "русский язык",
    "обществознание",
    "история",
    "математика",
    "информатика",
    "иностранный язык",
    "физика",
    "химия",
    "биология",
]

DOC_TERMS = {
    "заявление о приеме": [r"заявлен[иея]\s+о\s+прием", r"заявлен[иея]"],
    "документ, удостоверяющий личность": [r"документ[^.\n]{0,50}удостоверяющ[^.\n]{0,20}личност", r"паспорт"],
    "документ об образовании": [r"документ[^.\n]{0,50}об\s+образован", r"аттестат", r"диплом"],
    "фотографии": [r"фотограф"],
    "согласие на обработку персональных данных": [r"согласи[^.\n]{0,30}обработк[^.\n]{0,20}персональн"],
    "СНИЛС": [r"снилс"],
    "медицинские документы": [r"медицин", r"ввк", r"врачеб", r"медицинск[^.\n]{0,25}заключен"],
    "документы, подтверждающие особые права или индивидуальные достижения": [r"индивидуальн(?:ые|ых)\s+достижен", r"особ[а-я]*\s+прав", r"льгот", r"квот"],
}

APPLY_TERMS = {
    "лично": [r"лично", r"при личном посещении"],
    "почтой": [r"почт", r"оператор[а-я\s]+почтовой\s+связи"],
    "в электронной форме": [r"электрон", r"личн(?:ый|ом)\s+кабинет", r"через\s+госуслуг", r"единый\s+портал", r"информационн(?:ой|ом)\s+систем"],
}

BENEFIT_TERMS = [r"льгот", r"квот", r"особ[а-я]*\s+прав", r"преимуществен", r"целев"]
WITHOUT_EGE_TERMS = [r"без\s+егэ", r"внутренн", r"вступительн", r"после\s+спо", r"после\s+колледжа"]


def _normalize_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url.endswith("/api"):
        url = url[:-4]
    return url


def _post_ollama_chat(base: str, payload: dict, timeout_sec: int) -> dict:
    r = requests.post(f"{base}/api/chat", json=payload, timeout=(10, timeout_sec))
    r.raise_for_status()
    return r.json()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zа-я0-9]+", (text or "").lower(), flags=re.IGNORECASE)


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?;])\s+|\n+", text or "")
    return [re.sub(r"\s+", " ", x).strip() for x in raw if x and x.strip()]


def _ctx_haystack(c: Dict[str, Any]) -> str:
    return " \n".join(str(x) for x in [c.get("doc_title"), c.get("source"), c.get("text")] if x)


def _build_context(hits: List[Dict[str, Any]], limit: int = 12, max_chars: int = 1800) -> List[Dict[str, Any]]:
    ctx: List[Dict[str, Any]] = []
    for i, h in enumerate(hits[:limit]):
        payload = h.get("payload", {}) or {}
        text = (h.get("text") or "").strip()
        if not text:
            continue
        ctx.append(
            {
                "id": i,
                "point": payload.get("point"),
                "pages": payload.get("pages"),
                "source": payload.get("source"),
                "doc_title": payload.get("doc_title"),
                "doc_number": payload.get("doc_number"),
                "issued_date": payload.get("issued_date"),
                "revision_date": payload.get("revision_date"),
                "legal_force_name": payload.get("legal_force_name"),
                "legal_force_level": payload.get("legal_force_level"),
                "doc_kind": payload.get("doc_kind"),
                "program_scope": payload.get("program_scope") or ["all"],
                "topic_scope": payload.get("topic_scope") or [],
                "section_path": payload.get("section_path") or [],
                "is_contact_source": payload.get("is_contact_source"),
                "is_local_admission_rule": payload.get("is_local_admission_rule"),
                "score": h.get("score"),
                "text": text[:max_chars],
            }
        )
    return ctx


def _parse_marked(text: str) -> Tuple[str, str, List[int]]:
    t = (text or "").lstrip()
    m_voice = re.search(r"^VOICE:\s*(.+)\s*$", t, flags=re.MULTILINE)
    m_used = re.search(r"^USED_IDS:\s*(.+)\s*$", t, flags=re.MULTILINE)
    voice = m_voice.group(1).strip() if m_voice else ""
    used_raw = m_used.group(1).strip() if m_used else ""
    used_ids = [int(x) for x in re.findall(r"\d+", used_raw)] if used_raw else []
    pos = t.find("ANSWER:")
    answer = t[pos + len("ANSWER:") :].strip() if pos != -1 else t.strip()
    return voice, answer, used_ids


def _voice_from_answer(answer: str) -> str:
    sentence = _split_sentences(answer)[0] if answer else ""
    return sentence[:180].strip() or "Нашёл информацию в документах."


def _make_citations(ctx: List[Dict[str, Any]], used_ids: List[int]) -> List[Dict[str, Any]]:
    used = set(used_ids or [])
    if not used:
        used = {c["id"] for c in ctx[:3]}
    citations: List[Dict[str, Any]] = []
    for c in ctx:
        if c["id"] not in used:
            continue
        citations.append(
            {
                "point": c.get("point"),
                "pages": c.get("pages"),
                "source": c.get("source"),
                "score": round(float(c.get("score")), 4) if c.get("score") is not None else None,
                "doc_title": c.get("doc_title"),
                "doc_number": c.get("doc_number"),
                "issued_date": c.get("issued_date"),
                "revision_date": c.get("revision_date"),
                "legal_force_name": c.get("legal_force_name"),
                "legal_force_level": c.get("legal_force_level"),
                "doc_kind": c.get("doc_kind"),
                "program_scope": c.get("program_scope"),
                "topic_scope": c.get("topic_scope"),
            }
        )
    return citations


def _question_mentions_adjunct(question: str) -> bool:
    q = (question or "").lower()
    return any(x in q for x in ["адъюнкт", "аспиран", "научн"])


def _filter_ctx(question: str, ctx: List[Dict[str, Any]], intent: str) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    mentions_adjunct = _question_mentions_adjunct(question)
    for c in ctx:
        programs = set(c.get("program_scope") or [])
        hay = _ctx_haystack(c).lower()
        doc_kind = str(c.get("doc_kind") or "")
        if programs == {"adjunct"} and not mentions_adjunct:
            continue
        if intent in {"documents", "apply", "deadlines", "min_scores", "exams"} and doc_kind in {"charter", "records_retention", "contacts"}:
            continue
        if intent in {"documents", "apply"} and doc_kind in {"education_law", "service_law", "federal_law"}:
            continue
        if intent == "min_scores":
            if any(x in hay for x in ["индивидуальн", "достижен", "гто", "дополнительн", "портфолио", "контрольн", "норматив", "упражнен", "физическ"]):
                continue
            if not any(x in hay for x in ["балл", "егэ", "миним", "русский язык", "обществозн", "истори", "рособрнадзор"]):
                continue
        if intent == "without_ege" and not any(re.search(p, hay, flags=re.IGNORECASE) for p in WITHOUT_EGE_TERMS):
            continue
        if intent == "benefits" and not any(re.search(p, hay, flags=re.IGNORECASE) for p in BENEFIT_TERMS):
            continue
        if intent == "documents":
            if not any(x in hay for x in ["документ", "заявлен", "паспорт", "образован", "копи", "оригинал", "фотограф", "снилс", "медицин", "справк"]):
                continue
            if any(x in hay for x in ["чернобыль", "радиаци", "по убыванию суммарного количества", "ранжирован", "зачислен"]):
                continue
        if intent == "apply":
            if not any(x in hay for x in ["подать", "подач", "направить", "лично", "почт", "электрон", "госуслуг", "портал", "кабинет"]):
                continue
        filtered.append(c)
    return filtered or ctx


def _sentence_score(question: str, intent: str, c: Dict[str, Any], sent: str) -> float:
    q_tokens = set(_tokenize(question))
    s_tokens = set(_tokenize(sent))
    overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
    score = overlap
    lower = sent.lower()
    for kw in intent_keywords(intent):
        if kw in lower:
            score += 0.28
    if c.get("is_local_admission_rule"):
        score += 0.18
    if str(c.get("doc_kind") or "") in {"rules_of_admission", "admission_order", "admission_attachment", "exam_rules"}:
        score += 0.12
    if intent == "min_scores" and any(x in lower for x in ["миним", "егэ", "балл"]):
        score += 0.25
    if intent == "documents" and any(x in lower for x in ["документ", "заявлен", "паспорт", "оригинал", "копи", "образован"]):
        score += 0.22
    if intent == "apply" and any(x in lower for x in ["подать", "подач", "направить", "почт", "лично", "госуслуг", "электрон"]):
        score += 0.22
    if intent == "deadlines" and re.search(r"\b\d{1,2}[. ](?:0?\d|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b", lower, flags=re.IGNORECASE):
        score += 0.28
    return score


def _collect_evidence(question: str, ctx: List[Dict[str, Any]], intent: str, limit: int = 8) -> List[Tuple[float, str, Dict[str, Any]]]:
    ranked: List[Tuple[float, str, Dict[str, Any]]] = []
    seen = set()
    for c in _filter_ctx(question, ctx, intent):
        for sent in _split_sentences(c.get("text") or ""):
            if len(sent) < 20:
                continue
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            score = _sentence_score(question, intent, c, sent)
            if score <= 0.1:
                continue
            ranked.append((score, sent, c))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:limit]


def _unique_ctxs(evidence: List[Tuple[float, str, Dict[str, Any]]], max_items: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for _, _, c in evidence:
        key = (c.get("source"), c.get("point"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_items:
            break
    return out


def _format_points(ctx_items: List[Dict[str, Any]], max_items: int = 3) -> str:
    parts = []
    seen = set()
    for c in ctx_items:
        key = (c.get("source"), c.get("point"))
        if key in seen:
            continue
        seen.add(key)
        label = c.get("source") or "документ"
        if c.get("point"):
            label += f" (п. {c['point']})"
        parts.append(label)
        if len(parts) >= max_items:
            break
    return "; ".join(parts)


def _extract_subject_scores(evidence: List[Tuple[float, str, Dict[str, Any]]]) -> List[str]:
    found: List[str] = []
    seen = set()
    for _, sent, _ in evidence:
        lower = sent.lower()
        for subject in SUBJECT_NAMES:
            m = re.search(rf"{re.escape(subject)}[^0-9]{{0,40}}(\d{{2,3}})", lower)
            if m:
                item = f"{subject} — {m.group(1)}"
                if item not in seen:
                    seen.add(item)
                    found.append(item)
        if not found:
            # only allow a generic number if it explicitly appears next to 'минимальный балл' or 'ЕГЭ'
            for m in re.finditer(r"(миним[^.\n]{0,30}бал[^.\n]{0,10}|егэ[^.\n]{0,20})(\d{2,3})", lower):
                item = f"минимальный балл — {m.group(2)}"
                if item not in seen:
                    seen.add(item)
                    found.append(item)
    return found[:6]


def _extract_document_list(evidence: List[Tuple[float, str, Dict[str, Any]]]) -> List[str]:
    found: List[str] = []
    seen = set()
    for _, sent, _ in evidence:
        lower = sent.lower()
        for label, patterns in DOC_TERMS.items():
            if any(re.search(p, lower, flags=re.IGNORECASE) for p in patterns) and label not in seen:
                seen.add(label)
                found.append(label)
    return found[:6]


def _extract_apply_methods(evidence: List[Tuple[float, str, Dict[str, Any]]]) -> List[str]:
    methods: List[str] = []
    seen = set()
    for _, sent, _ in evidence:
        lower = sent.lower()
        for label, patterns in APPLY_TERMS.items():
            if any(re.search(p, lower, flags=re.IGNORECASE) for p in patterns) and label not in seen:
                seen.add(label)
                methods.append(label)
    return methods[:4]


def _extract_dates(evidence: List[Tuple[float, str, Dict[str, Any]]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for _, sent, _ in evidence:
        for m in re.findall(r"\b\d{1,2}[. ](?:0?\d|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:[. ]20\d{2})?\b", sent, flags=re.IGNORECASE):
            key = m.lower().strip()
            if key not in seen:
                seen.add(key)
                out.append(m.strip())
    return out[:4]


def _pt(point: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(point or "")))


def _point_distance(a: Any, b: Any) -> int | None:
    pa = _pt(a)
    pb = _pt(b)
    if not pa or not pb:
        return None
    if len(pa) != len(pb) or pa[:-1] != pb[:-1]:
        return None
    return abs(pa[-1] - pb[-1])


def _same_section(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    sa = tuple((a.get("section_path") or [])[:2])
    sb = tuple((b.get("section_path") or [])[:2])
    return bool(sa) and sa == sb


def _source_blobs(ctx: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(ctx, key=lambda x: float(x.get("score") or 0.0), reverse=True)
    out: List[Dict[str, Any]] = []
    seen = set()
    for seed in ordered[:6]:
        group: List[Dict[str, Any]] = []
        for cand in ctx:
            if cand.get("source") != seed.get("source"):
                continue
            dist = _point_distance(seed.get("point"), cand.get("point"))
            if _same_section(seed, cand) or (dist is not None and dist <= 2):
                group.append(cand)
        if not group:
            group = [seed]
        group = sorted(group, key=lambda x: (_pt(x.get("point")) or (99999,), -float(x.get("score") or 0.0)))
        key = tuple((g.get("source"), g.get("point")) for g in group)
        if key in seen:
            continue
        seen.add(key)
        block = "\n".join(dict.fromkeys((x.get("text") or "").strip() for x in group if (x.get("text") or "").strip()))
        out.append({"ctx": group, "text": block})
    return out


def _extract_list_items(text: str) -> List[str]:
    items: List[str] = []
    for line in re.split(r"[\n;]+", text or ""):
        s = re.sub(r"^[-•·●▪◦*]+\s*", "", line).strip()
        s = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", s)
        if len(s) >= 6:
            items.append(s)
    return items


def _best_sentences(question: str, evidence: List[Tuple[float, str, Dict[str, Any]]], intent: str, max_items: int = 3) -> List[str]:
    ranked: List[Tuple[float, str]] = []
    for score, sent, _ in evidence:
        lower = sent.lower()
        bonus = 0.0
        if intent == "min_scores" and any(x in lower for x in ["миним", "егэ", "балл"]):
            bonus += 0.35
        if intent == "documents" and any(x in lower for x in ["документ", "заявлен", "паспорт", "образован", "копи", "оригинал"]):
            bonus += 0.32
        if intent == "apply" and any(x in lower for x in ["лично", "почт", "электрон", "госуслуг", "направить", "подать"]):
            bonus += 0.32
        if intent == "without_ege" and any(x in lower for x in ["без егэ", "внутренн", "вступительн", "результаты егэ"]):
            bonus += 0.35
        if len(sent) > 260:
            bonus -= 0.08
        ranked.append((score + bonus, re.sub(r"\s+", " ", sent).strip()))
    ranked.sort(key=lambda x: x[0], reverse=True)
    out: List[str] = []
    seen = set()
    for _, sent in ranked:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sent)
        if len(out) >= max_items:
            break
    return out


def _extract_subject_scores_from_blobs(blobs: List[Dict[str, Any]]) -> List[str]:
    found: List[str] = []
    seen = set()
    for blob in blobs:
        text = (blob.get("text") or "").lower()
        lines = _extract_list_items(text)
        for line in lines or [text]:
            lower = line.lower()
            for subject in SUBJECT_NAMES:
                if subject not in lower:
                    continue
                m = re.search(r"(?:не\s+менее\s+|миним[^0-9]{0,25}|балл[^0-9]{0,15}|егэ[^0-9]{0,15})?(\d{2,3})", lower)
                if m:
                    item = f"{subject} — {m.group(1)}"
                    if item not in seen:
                        seen.add(item)
                        found.append(item)
    return found[:8]


def _extract_document_list_from_blobs(blobs: List[Dict[str, Any]]) -> List[str]:
    found: List[str] = []
    seen = set()
    for blob in blobs:
        text = blob.get("text") or ""
        for line in _extract_list_items(text):
            lower = line.lower()
            if any(x in lower for x in ["чернобыль", "радиаци", "по убыванию", "зачислен"]):
                continue
            for label, patterns in DOC_TERMS.items():
                if any(re.search(p, lower, flags=re.IGNORECASE) for p in patterns):
                    if label not in seen:
                        seen.add(label)
                        found.append(label)
    return found[:10]


def _extract_apply_methods_from_blobs(blobs: List[Dict[str, Any]]) -> List[str]:
    methods: List[str] = []
    seen = set()
    for blob in blobs:
        text = blob.get("text") or ""
        for line in _extract_list_items(text) + [text]:
            lower = line.lower().strip()
            if not lower:
                continue
            for label, patterns in APPLY_TERMS.items():
                if any(re.search(p, lower, flags=re.IGNORECASE) for p in patterns) and label not in seen:
                    seen.add(label)
                    methods.append(label)
    return methods[:5]


def _extract_without_ege_categories(blobs: List[Dict[str, Any]]) -> List[str]:
    cats: List[str] = []
    seen = set()
    patterns = [
        (r"поступающ[^.\n]{0,120}после\s+спо[^.\n]{0,120}", "лица после СПО"),
        (r"поступающ[^.\n]{0,120}после\s+колледжа[^.\n]{0,120}", "лица после колледжа"),
        (r"внутренн(?:ие|их)\s+вступительн(?:ые|ых)\s+испытан", "лица, которым разрешены внутренние вступительные испытания"),
        (r"без\s+результат[^.\n]{0,80}егэ", "лица без результатов ЕГЭ в случаях, предусмотренных правилами приема"),
    ]
    for blob in blobs:
        text = blob.get("text") or ""
        for sent in _split_sentences(text):
            lower = sent.lower()
            if not any(re.search(p, lower, flags=re.IGNORECASE) for p in WITHOUT_EGE_TERMS):
                continue
            for pat, label in patterns:
                if re.search(pat, lower, flags=re.IGNORECASE) and label not in seen:
                    seen.add(label)
                    cats.append(label)
    return cats[:4]


def _format_support(sentences: List[str]) -> str:
    cleaned = [re.sub(r"\s+", " ", s).strip().rstrip(";:") for s in sentences if s]
    if not cleaned:
        return ""
    return " ".join(cleaned[:2])


def _structured_answer(question: str, ctx: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    intent = infer_question_intent(question)
    filtered_ctx = _filter_ctx(question, ctx, intent)
    evidence = _collect_evidence(question, filtered_ctx, intent)
    if not evidence:
        return None
    used_ctx = _unique_ctxs(evidence)
    profile = question_profile(question)
    blobs = _source_blobs(filtered_ctx)
    best_sentences = _best_sentences(question, evidence, intent, max_items=3)
    support = _format_support(best_sentences)

    if intent == "min_scores":
        subject_scores = _extract_subject_scores_from_blobs(blobs) or _extract_subject_scores(evidence)
        if len(subject_scores) >= 2:
            answer = f"По найденным фрагментам минимальные баллы ЕГЭ такие: {'; '.join(subject_scores[:6])}. Основание: {_format_points(used_ctx)}."
            confidence = 0.84
        elif subject_scores:
            answer = f"По текущей выдаче надежно подтверждено значение: {subject_scores[0]}. Полный список по всем предметам лучше сверить по пунктам {_format_points(used_ctx)}."
            confidence = 0.70
        else:
            answer = f"В найденных фрагментах есть упоминания минимальных баллов ЕГЭ, но без надежно подтвержденного полного списка по предметам. Проверьте пункты {_format_points(used_ctx)}."
            confidence = 0.45
        return {"voice_answer": _voice_from_answer(answer), "answer": answer, "citations": _make_citations(used_ctx, []), "meta": {"engine": "grounded", "question_profile": profile, "question_intent": intent, "grounding_confidence": confidence}}

    if intent == "documents":
        docs = _extract_document_list_from_blobs(blobs) or _extract_document_list(evidence)
        if len(docs) >= 2:
            answer = f"По найденным фрагментам для поступления обычно нужны: {', '.join(docs[:8])}. Точный состав может зависеть от вашей категории и программы. Основание: {_format_points(used_ctx)}."
            confidence = 0.82
        elif docs:
            answer = f"По текущей выдаче подтвержден как минимум такой документ: {docs[0]}. Остальной перечень лучше сверить по пунктам {_format_points(used_ctx)}."
            confidence = 0.63
        else:
            answer = f"Точный перечень документов нужно брать из правил приема и порядка приема по вашей программе. В текущей выдаче подтверждено, что ориентироваться нужно на пункты {_format_points(used_ctx)}."
            confidence = 0.42
        return {"voice_answer": _voice_from_answer(answer), "answer": answer, "citations": _make_citations(used_ctx, []), "meta": {"engine": "grounded", "question_profile": profile, "question_intent": intent, "grounding_confidence": confidence}}

    if intent == "apply":
        methods = _extract_apply_methods_from_blobs(blobs) or _extract_apply_methods(evidence)
        if methods:
            answer = f"По найденным фрагментам документы можно подать следующими способами: {', '.join(methods[:4])}. Точный порядок и ограничения смотрите в пунктах {_format_points(used_ctx)}."
            confidence = 0.80 if len(methods) >= 2 else 0.72
        else:
            answer = f"В текущей выдаче подтвержден общий порядок подачи заявления и документов, но без полного перечня способов. Ориентируйтесь на пункты {_format_points(used_ctx)}."
            confidence = 0.43
        return {"voice_answer": _voice_from_answer(answer), "answer": answer, "citations": _make_citations(used_ctx, []), "meta": {"engine": "grounded", "question_profile": profile, "question_intent": intent, "grounding_confidence": confidence}}

    if intent == "deadlines":
        dates = _extract_dates(evidence)
        if dates:
            answer = f"По найденным фрагментам ключевые даты приема такие: {', '.join(dates)}. Проверьте формулировку и этап приема в пунктах {_format_points(used_ctx)}."
            confidence = 0.75
        elif support:
            answer = f"По найденным пунктам о сроках подтверждается следующее: {support} Основание: {_format_points(used_ctx)}."
            confidence = 0.56
        else:
            answer = f"В текущей выдаче нет надежно выделенной даты, хотя вопрос относится к срокам приема. Проверьте пункты {_format_points(used_ctx)}."
            confidence = 0.38
        return {"voice_answer": _voice_from_answer(answer), "answer": answer, "citations": _make_citations(used_ctx, []), "meta": {"engine": "grounded", "question_profile": profile, "question_intent": intent, "grounding_confidence": confidence}}

    if intent == "without_ege":
        categories = _extract_without_ege_categories(blobs)
        if categories:
            answer = f"По найденным фрагментам поступление без ЕГЭ возможно не для всех, а только для прямо указанных категорий поступающих. В текущей выдаче подтверждаются такие ориентиры: {'; '.join(categories[:3])}. Основание: {_format_points(used_ctx)}."
            confidence = 0.74
        elif support:
            answer = f"По найденным фрагментам поступление без ЕГЭ возможно не для всех, а только в случаях, прямо указанных в правилах приема или порядке приема. Основание: {_format_points(used_ctx)}."
            confidence = 0.61
        else:
            answer = f"По текущей выдаче нельзя надежно подтвердить общий ответ о поступлении без ЕГЭ для всех категорий. Нужен конкретный пункт правил приема по вашей категории поступления. Проверьте {_format_points(used_ctx)}."
            confidence = 0.35
        return {"voice_answer": _voice_from_answer(answer), "answer": answer, "citations": _make_citations(used_ctx, []), "meta": {"engine": "grounded", "question_profile": profile, "question_intent": intent, "grounding_confidence": confidence, "needs_optional_clarification": True}}

    if intent == "benefits":
        if support:
            answer = f"По найденным фрагментам льготы и особые права нужно определять по пунктам, где они прямо перечислены. Подтверждение: {support} Основание: {_format_points(used_ctx)}."
            confidence = 0.55
        else:
            answer = f"Льготы и особые права нужно определять только по пунктам, где прямо указаны льготы, квоты, целевое обучение или преимущественное право. По текущей выдаче ориентируйтесь на {_format_points(used_ctx)}."
            confidence = 0.4
        return {"voice_answer": _voice_from_answer(answer), "answer": answer, "citations": _make_citations(used_ctx, []), "meta": {"engine": "grounded", "question_profile": profile, "question_intent": intent, "grounding_confidence": confidence, "needs_optional_clarification": True}}

    if intent == "exams":
        subjects = []
        seen = set()
        for _, sent, _ in evidence:
            lower = sent.lower()
            for s in SUBJECT_NAMES:
                if s in lower and s not in seen:
                    seen.add(s)
                    subjects.append(s)
        if subjects:
            answer = f"По найденным фрагментам для поступления фигурируют следующие предметы или испытания: {', '.join(subjects[:5])}. Точный набор зависит от программы и категории поступающего. Основание: {_format_points(used_ctx)}."
            confidence = 0.67
        elif support:
            answer = f"По найденным пунктам о вступительных испытаниях подтверждается следующее: {support} Основание: {_format_points(used_ctx)}."
            confidence = 0.52
        else:
            answer = f"Точный перечень вступительных испытаний нужно смотреть в правилах приема и порядке приема по вашей программе. По текущей выдаче ориентируйтесь на {_format_points(used_ctx)}."
            confidence = 0.37
        return {"voice_answer": _voice_from_answer(answer), "answer": answer, "citations": _make_citations(used_ctx, []), "meta": {"engine": "grounded", "question_profile": profile, "question_intent": intent, "grounding_confidence": confidence}}

    return None


def _llm_answer(question: str, ctx: List[Dict[str, Any]], model: str, base: str, timeout_sec: int, temperature: float, num_predict: int) -> Dict[str, Any]:
    intent = infer_question_intent(question)
    evidence = _collect_evidence(question, ctx, intent, limit=6)
    if not evidence:
        msg = "В найденных фрагментах нет достаточно сильного подтверждения для точного ответа."
        return {"voice_answer": msg, "answer": msg, "citations": [], "meta": {"engine": "grounded", "provider": "ollama", "model": model, "question_profile": question_profile(question), "question_intent": intent, "grounding_confidence": 0.0}}

    evidence_ctx = _unique_ctxs(evidence, max_items=4)
    context_block = "\n\n".join(
        f"[id={c['id']}] ({c.get('source')}; п. {c.get('point')}; kind={c.get('doc_kind')})\n{c.get('text')}" for c in evidence_ctx
    )
    prompt = f"Вопрос: {question}\nПрофиль: {question_profile(question)}\nIntent: {intent}\n\nФрагменты:\n{context_block}\n\nСформулируй короткий ответ без фантазий. Если точный ответ не подтвержден, так и скажи."
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    data = _post_ollama_chat(base, payload, timeout_sec=timeout_sec)
    content = (((data or {}).get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("Empty response from Ollama")
    voice, answer, used_ids = _parse_marked(content)
    if not answer:
        answer = "В найденных фрагментах нет достаточно сильного подтверждения для точного ответа."
    if not voice:
        voice = _voice_from_answer(answer)
    citations = _make_citations(evidence_ctx, used_ids)
    return {"voice_answer": voice, "answer": answer, "citations": citations, "meta": {"engine": "llm_grounded", "provider": "ollama", "model": model, "question_profile": question_profile(question), "question_intent": intent, "grounding_confidence": 0.62}}


def generate_llm_answer(question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider != "ollama":
        raise RuntimeError(f"LLM_PROVIDER must be 'ollama', got: {provider!r}")
    base = _normalize_base_url(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    timeout_sec = int(os.getenv("OLLAMA_TIMEOUT_SEC", "600"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "220"))
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

    ctx = _build_context(hits)
    if not ctx:
        msg = "Не нашёл релевантных фрагментов в документах."
        return {"voice_answer": msg, "answer": msg, "citations": [], "meta": {"engine": "grounded", "provider": provider, "model": model, "grounding_confidence": 0.0}}

    structured = _structured_answer(question, ctx)
    if structured is not None:
        structured["meta"] = {**(structured.get("meta") or {}), "provider": provider, "model": model}
        return structured

    if question_requires_strict_grounding(question):
        msg = "Точный ответ не подтвержден текущей выдачей документов. Нужен более точный процедурный фрагмент или одно уточнение по вашей категории поступления."
        return {"voice_answer": msg, "answer": msg, "citations": _make_citations(ctx[:2], []), "meta": {"engine": "grounded", "provider": provider, "model": model, "question_profile": question_profile(question), "question_intent": infer_question_intent(question), "grounding_confidence": 0.2, "handoff_recommended": False}}

    return _llm_answer(question, ctx, model=model, base=base, timeout_sec=timeout_sec, temperature=temperature, num_predict=num_predict)
