from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .knowledge import infer_question_intent, INTENT_TO_PROFILE


@dataclass(frozen=True)
class LegalForce:
    code: str
    label: str
    level: int


LEGAL_FORCE_MAP = {
    "constitution": LegalForce("constitution", "Конституция РФ", 700),
    "international_treaty": LegalForce("international_treaty", "Международный договор РФ", 650),
    "fkz": LegalForce("fkz", "Федеральный конституционный закон", 600),
    "federal_law": LegalForce("federal_law", "Федеральный закон / кодекс", 500),
    "regional_law": LegalForce("regional_law", "Закон субъекта РФ", 400),
    "presidential_decree": LegalForce("presidential_decree", "Указ Президента РФ", 300),
    "government_resolution": LegalForce("government_resolution", "Постановление Правительства РФ", 250),
    "ministry_order": LegalForce("ministry_order", "Ведомственный акт / приказ", 200),
    "local_rules": LegalForce("local_rules", "Локальный акт вуза / правила приема", 100),
    "other": LegalForce("other", "Иной документ", 50),
}

_ENCODED_UNICODE_RE = re.compile(r"#U([0-9A-Fa-f]{4})")

_CLASSIFICATION_RULES = [
    ("constitution", [r"\bконституц(?:ия|ии|ией)\b"]),
    ("international_treaty", [r"международн(?:ый|ого|ые|ых)?\s+договор", r"\bконвенц(?:ия|ии)\b", r"\bсоглашени[ея]\b"]),
    ("fkz", [r"федеральн(?:ый|ого)\s+конституционн(?:ый|ого)\s+закон", r"\bфкз\b"]),
    ("federal_law", [r"федеральн(?:ый|ого)\s+закон", r"\b\d+\s*[-–]?\s*фз\b", r"\bкодекс\b"]),
    ("regional_law", [r"закон\s+(?:субъекта|республики|края|области|города\s+федерального\s+значения)"]),
    ("presidential_decree", [r"указ\s+президент[а]?"]),
    ("government_resolution", [r"постановлени[ея]\s+правительств[ао]"]),
    ("local_rules", [r"правил[ао]\s+прием[аыу]", r"локальн(?:ый|ого)\s+акт", r"регламент", r"положение\s+о\s+приеме"]),
    ("ministry_order", [r"\bприказ\b", r"\bмвд\b", r"\bминобрнауки\b", r"\bрособрнадзор\b", r"министерств[ао]"]),
]

GENERAL_LAW_PATTERNS = [
    r"\bустав\b",
    r"о\s+службе\s+в\s+органах\s+внутренних\s+дел",
    r"об\s+образовании\s+в\s+российской\s+федерации",
    r"об\s+общих\s+принципах",
    r"конституц(?:ия|ии)",
]
PROCEDURAL_DOCUMENT_PATTERNS = [
    r"правил[ао]\s+прием[аыу]",
    r"порядок\s+прием[аыу]",
    r"положение\s+о\s+приеме",
    r"переч(?:ень|ни)\s+документ",
    r"приемн(?:ая|ой)\s+комисси",
    r"зачислен",
    r"вступительн(?:ые|ых)\s+испытан",
    r"минимальн(?:ые|ых)?\s+балл",
    r"подач[аи]\s+документ",
]
DOCUMENTS_QUESTION_PATTERNS = [
    r"каки(?:е|х)\s+документ",
    r"список\s+документ",
    r"переч(?:ень|ни)\s+документ",
    r"что\s+нужно\s+для\s+поступлен",
]
STRICT_STATUS_KEYWORDS = [
    "без егэ", "льгот", "особое право", "преимущественное право", "квот", "целев", "могу ли", "имею ли право", "после колледжа", "после спо", "сотрудник", "служу", "прямой набор",
]
STRICT_LEVEL_KEYWORDS = [
    "какие экзамен", "вступительн", "испытан", "конкурс", "проходн", "минимальн", "балл", "зачисл", "по результатам", "без егэ",
]
STRICT_FORM_KEYWORDS = [
    "очная или заочная", "очная либо заочная", "по очной", "по заочной", "дистанционн", "форма обучения",
]
ANSWER_FIRST_KEYWORDS = [
    "какие документы", "список документов", "перечень документов", "что нужно", "до какого", "когда", "срок", "куда подавать", "как подать", "адрес", "телефон", "часы работы", "где находится", "контакты", "минимальные баллы",
]
NARROW_CATEGORY_PATTERNS = [
    r"специальн(?:ое|ого)\s+звани",
    r"действующ(?:ий|его)\s+сотрудник",
    r"служб[аы]\s+в\s+органах\s+внутренних\s+дел",
    r"прям(?:ой|ого)\s+набор",
]


def decode_escaped_unicode(text: str) -> str:
    if not text:
        return ""
    return _ENCODED_UNICODE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _normalize(text: str) -> str:
    text = decode_escaped_unicode(text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def classify_legal_force(*, title: Optional[str] = None, doc_type: Optional[str] = None, source_name: Optional[str] = None, preview_text: Optional[str] = None) -> LegalForce:
    head = _normalize(" \n".join(x for x in [title, doc_type, source_name] if x))
    preview = _normalize(preview_text or "")

    if not head and not preview:
        return LEGAL_FORCE_MAP["other"]

    for code, patterns in _CLASSIFICATION_RULES:
        for pattern in patterns:
            if head and re.search(pattern, head, flags=re.IGNORECASE):
                return LEGAL_FORCE_MAP[code]

    if head:
        if re.search(r"правил[ао]\s+прием[аыу]", head, flags=re.IGNORECASE):
            return LEGAL_FORCE_MAP["local_rules"]
        if re.search(r"\bприказ\b", head, flags=re.IGNORECASE):
            return LEGAL_FORCE_MAP["ministry_order"]
        if re.search(r"федеральн(?:ый|ого)\s+закон|\b\d+\s*[-–]?\s*фз\b", head, flags=re.IGNORECASE):
            return LEGAL_FORCE_MAP["federal_law"]

    for code, patterns in _CLASSIFICATION_RULES:
        for pattern in patterns:
            if preview and re.search(pattern, preview, flags=re.IGNORECASE):
                return LEGAL_FORCE_MAP[code]

    return LEGAL_FORCE_MAP["other"]


def question_profile(question: str) -> str:
    intent = infer_question_intent(question)
    return INTENT_TO_PROFILE.get(intent, "mixed")


def is_general_law_document(text: str) -> bool:
    t = _normalize(text)
    return any(re.search(pattern, t, flags=re.IGNORECASE) for pattern in GENERAL_LAW_PATTERNS)


def is_procedural_document(text: str) -> bool:
    t = _normalize(text)
    return any(re.search(pattern, t, flags=re.IGNORECASE) for pattern in PROCEDURAL_DOCUMENT_PATTERNS)


def is_documents_question(question: str) -> bool:
    q = _normalize(question)
    return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in DOCUMENTS_QUESTION_PATTERNS)


def mentions_narrow_category(text: str) -> bool:
    t = _normalize(text)
    return any(re.search(pattern, t, flags=re.IGNORECASE) for pattern in NARROW_CATEGORY_PATTERNS)


def legal_force_bonus(question: str, payload: dict) -> float:
    profile = question_profile(question)
    intent = infer_question_intent(question)
    code = (payload or {}).get("legal_force_code") or "other"
    level = int((payload or {}).get("legal_force_level") or 0)
    if profile == "normative":
        return min(0.22, level / 3200)
    if profile == "procedural":
        if intent == "min_scores":
            if code == "local_rules":
                return 0.16
            if code == "ministry_order":
                return 0.12
            if code in {"federal_law", "fkz", "constitution", "international_treaty"}:
                return -0.06
            return 0.0
        if code == "local_rules":
            return 0.14
        if code in {"ministry_order", "government_resolution"}:
            return 0.10
        if code in {"federal_law", "fkz", "constitution", "international_treaty"}:
            return -0.04
        return 0.0
    if profile == "organizational":
        if code in {"local_rules", "ministry_order"}:
            return 0.06
        if code in {"constitution", "international_treaty", "fkz", "federal_law"}:
            return -0.04
        return 0.0
    if profile == "eligibility":
        if code in {"federal_law", "fkz", "government_resolution", "ministry_order", "local_rules"}:
            return 0.06
        return 0.02 if level else 0.0
    return 0.04 if level else 0.0


def requires_precise_status(question: str) -> bool:
    q = _normalize(question)
    return bool(q) and any(kw in q for kw in STRICT_STATUS_KEYWORDS)


def requires_precise_level(question: str) -> bool:
    q = _normalize(question)
    return bool(q) and any(kw in q for kw in STRICT_LEVEL_KEYWORDS)


def requires_precise_form(question: str) -> bool:
    q = _normalize(question)
    return bool(q) and any(kw in q for kw in STRICT_FORM_KEYWORDS)


def prefers_answer_first(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False
    if question_profile(question) in {"procedural", "organizational"} and infer_question_intent(question) not in {"without_ege", "benefits"}:
        return True
    return any(kw in q for kw in ANSWER_FIRST_KEYWORDS)
