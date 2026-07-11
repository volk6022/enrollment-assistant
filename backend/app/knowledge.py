from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


APP_DIR = Path(__file__).resolve().parent
CONFIG_DIR = APP_DIR.parent / "config"
_ENCODED_UNICODE_RE = re.compile(r"#U([0-9A-Fa-f]{4})")


def _decode_escaped_unicode(text: Optional[str]) -> str:
    if not text:
        return ""
    return _ENCODED_UNICODE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", _decode_escaped_unicode(text or "")).strip().lower()


def _norm_key(text: Optional[str]) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", _norm(text), flags=re.IGNORECASE)


def _has_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


INTENT_RULES: List[tuple[str, List[str]]] = [
    ("contacts", [
        r"где\s+наход", r"адрес", r"телефон", r"email", r"e-mail", r"почт", r"контакт", r"часы\s+работы", r"график\s+работы", r"кабинет", r"как\s+проехать", r"где\s+приемн",
    ]),
    ("min_scores", [
        r"минимальн[а-я\s]+балл", r"минимальн[а-я\s]+количеств[а-я\s]+балл", r"скольк[о]?\s+балл[а-я\s]+нужн", r"порог\s+по\s+егэ", r"проходн[а-я\s]+минимум", r"ниже\s+какого\s+балл", r"балл[а-я\s]+егэ", r"порог", r"минимум\s+по\s+русскому", r"минимум\s+по\s+обществознанию",
    ]),
    ("without_ege", [
        r"без\s+егэ", r"без\s+результат[а-я\s]+егэ", r"по\s+внутренн(?:им|ие)\s+испытан", r"внутренн(?:ие|их)\s+экзамен", r"после\s+колледжа.*без\s+егэ", r"после\s+спо.*без\s+егэ",
    ]),
    ("benefits", [
        r"льгот", r"квот", r"особ[а-я]*\s+прав", r"преимуществен", r"без\s+вступительных", r"целев", r"индивидуальн(?:ые|ых)\s+достижен", r"сирот", r"инвалид", r"ветеран",
    ]),
    ("documents", [
        r"каки(?:е|х)\s+документ", r"переч(?:ень|ни)\s+документ", r"список\s+документ", r"что\s+нужно\s+для\s+поступлен", r"какие\s+справк", r"какие\s+бумаг",
    ]),
    ("deadlines", [
        r"до\s+какого", r"когда\s+подав", r"срок", r"дедлайн", r"дата\s+подач", r"завершени[ея]\s+прием", r"когда\s+заканчивается\s+прием", r"до\s+какого\s+числа",
    ]),
    ("apply", [
        r"как\s+подать", r"куда\s+подав", r"способ[а-я\s]+подач", r"через\s+госуслуг", r"личн(?:о|ый\s+кабинет)", r"почт(?:ой|ой\s+можно)", r"отправить\s+документ", r"подач[аи]\s+документ",
    ]),
    ("exams", [
        r"каки[ея]\s+экзамен", r"что\s+сдават", r"вступительн(?:ые|ых)\s+испытан", r"какие\s+предмет", r"физическ[а-я\s]+подготов", r"экзам", r"испытан",
    ]),
    ("eligibility", [
        r"могу\s+ли", r"можно\s+ли", r"имею\s+ли\s+право", r"кто\s+может", r"кто\s+имеет\s+право", r"подхожу\s+ли", r"после\s+колледжа", r"после\s+спо", r"можно\s+поступить",
    ]),
    ("normative", [
        r"каким\s+законом", r"каким\s+актом", r"на\s+основании\s+чего", r"правовое\s+основание", r"какая\s+норма", r"чем\s+регулируется", r"каким\s+нпа", r"каким\s+приказом",
    ]),
]

INTENT_TO_TOPICS: Dict[str, Set[str]] = {
    "contacts": {"contacts"},
    "min_scores": {"exams", "eligibility"},
    "without_ege": {"eligibility", "exams"},
    "benefits": {"benefits", "eligibility"},
    "documents": {"documents"},
    "deadlines": {"deadlines"},
    "apply": {"apply", "documents"},
    "exams": {"exams"},
    "eligibility": {"eligibility"},
    "normative": {"eligibility", "benefits"},
}

INTENT_TO_PROFILE = {
    "contacts": "organizational",
    "min_scores": "procedural",
    "without_ege": "eligibility",
    "benefits": "eligibility",
    "documents": "procedural",
    "deadlines": "procedural",
    "apply": "procedural",
    "exams": "procedural",
    "eligibility": "eligibility",
    "normative": "normative",
    "general": "mixed",
}

PROGRAM_PATTERNS = {
    "adjunct": [r"адъюнкт", r"аспирант", r"научн[а-я\-\s]+кадр"],
    "master": [r"магистр", r"магистрат"],
    "bachelor": [r"бакалавр"],
    "specialist": [r"специалитет", r"специалист"],
    "spo": [r"спо", r"средн(?:ее|его)\s+профессиональн", r"колледж"],
}

DOC_KIND_RULES = [
    ("rules_of_admission", [r"правил[ао]\s+прием[аыу]", r"правила\s+приема"]),
    ("admission_attachment", [r"приложени[ея].*прием", r"переч(?:ень|ни)\s+документ", r"образец\s+заявлен", r"график\s+прием", r"расписани[ея]\s+вступительн", r"минимальн[а-я\s]+балл"]),
    ("admission_order", [r"порядок\s+и\s+условия\s+прием", r"порядок\s+прием", r"условия\s+прием"]),
    ("exam_rules", [r"вступительн(?:ые|ых)\s+испытан", r"экзамен", r"физическ[а-я\s]+подготов", r"минимальн[а-я\s]+балл", r"егэ"]),
    ("benefits", [r"особ[а-я]*\s+прав", r"льгот", r"квот", r"целев", r"индивидуальн(?:ые|ых)\s+достижен"]),
    ("contacts", [r"контакт", r"адрес", r"телефон", r"приемн(?:ая|ой)\s+комисси"]),
    ("charter", [r"устав"]),
    ("education_law", [r"об\s+образовании\s+в\s+российской\s+федерации"]),
    ("service_law", [r"о\s+службе\s+в\s+органах\s+внутренних\s+дел"]),
    ("health_requirements", [r"военно\-?врачебн", r"врачебн(?:ых|ые)\s+комисси", r"медицинск[а-я\s]+осмотр"]),
    ("records_retention", [r"сроков\s+их\s+хранения", r"сроков\s+хранения", r"образующихся\s+в\s+процессе\s+деятельности"]),
]

NEGATIVE_MARKERS = {
    "min_scores": [r"индивидуальн(?:ые|ых)\s+достижен", r"диплом\s+с\s+отличием", r"гто", r"дополнительн(?:ые|ых)\s+балл", r"портфолио", r"олимпиад"],
    "apply": [r"сроков\s+их\s+хранения", r"военно\-?врачебн", r"ввк"],
    "documents": [r"сроков\s+их\s+хранения"],
    "benefits": [r"договор\s+об\s+образовании", r"платн(?:ое|ых)\s+мест"],
}

INTENT_QUERY_EXPANSIONS = {
    "min_scores": [
        "минимальные баллы егэ по предметам",
        "минимальное количество баллов егэ русский язык обществознание история",
        "минимальный порог результатов егэ для поступления",
    ],
    "without_ege": [
        "поступление без егэ внутренние вступительные испытания",
        "кто может сдавать внутренние вступительные испытания вместо егэ",
        "после спо внутренние экзамены вместо егэ",
    ],
    "documents": [
        "перечень документов для поступления заявление паспорт документ об образовании",
        "какие документы представляются при приеме",
    ],
    "apply": [
        "способы подачи документов лично почтой электронно",
        "как направить заявление о приеме и документы",
    ],
    "deadlines": [
        "сроки приема документов даты завершения приема",
        "до какого числа принимают документы",
    ],
    "exams": [
        "вступительные испытания перечень предметов егэ внутренние экзамены",
        "какие предметы нужны для поступления",
    ],
    "benefits": [
        "льготы квоты особые права преимущественное право при поступлении",
    ],
    "eligibility": [
        "кто может поступать условия допуска к приему",
    ],
}

RISKY_INTENTS = {"min_scores", "without_ege", "benefits", "eligibility", "documents", "apply", "exams"}

INTENT_KEYWORDS = {
    "min_scores": ["миним", "балл", "егэ", "предмет"],
    "without_ege": ["без егэ", "внутренн", "испытан", "после спо", "после колледжа"],
    "documents": ["документ", "заявлен", "паспорт", "оригинал", "копи", "документ об образовании"],
    "apply": ["подать", "подач", "направить", "лично", "почт", "госуслуг", "электрон"],
    "deadlines": ["срок", "до какого", "дата", "прием документов"],
    "exams": ["экзам", "вступительн", "егэ", "предмет", "физическ"],
    "benefits": ["льгот", "квот", "особое право", "преимуществен", "целев"],
    "eligibility": ["можно", "имеет право", "допуска", "после спо"],
    "contacts": ["адрес", "телефон", "почта", "график", "кабинет"],
}


def _json_candidates(env_name: str, defaults: List[Path]) -> List[Path]:
    out: List[Path] = []
    env_val = os.getenv(env_name, "").strip()
    if env_val:
        out.append(Path(env_val))
    out.extend(defaults)
    return out


@lru_cache(maxsize=1)
def load_document_registry() -> Dict[str, Dict[str, Any]]:
    defaults = [
        Path("/data/npa/2025/document_registry.json"),
        Path("/data/npa/document_registry.json"),
        CONFIG_DIR / "document_registry.sample.json",
    ]
    for path in _json_candidates("LEGAL_DOC_REGISTRY_PATH", defaults):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        docs = raw.get("documents", raw) if isinstance(raw, dict) else raw
        if not isinstance(docs, list):
            continue
        out: Dict[str, Dict[str, Any]] = {}
        for item in docs:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            for key in [item.get("source"), item.get("source_file"), item.get("file_name"), item.get("title"), item.get("match")]:
                nkey = _norm_key(key)
                if nkey:
                    out[nkey] = normalized
        return out
    return {}


@lru_cache(maxsize=1)
def load_contacts_data() -> Dict[str, Any]:
    defaults = [
        Path("/data/contacts.json"),
        Path("/data/npa/2025/contacts.json"),
        CONFIG_DIR / "contacts.sample.json",
    ]
    for path in _json_candidates("CONTACTS_FILE_PATH", defaults):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def get_registry_entry(*, source: Optional[str] = None, source_file: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    registry = load_document_registry()
    for cand in [source, source_file, title, _decode_escaped_unicode(source or ""), _decode_escaped_unicode(source_file or "")]:
        key = _norm_key(cand)
        if key and key in registry:
            return dict(registry[key])
    return {}


def infer_question_intent(question: str) -> str:
    q = _norm(question)
    if not q:
        return "general"
    scores: Dict[str, int] = {}
    for intent, patterns in INTENT_RULES:
        scores[intent] = sum(2 if re.search(p, q, flags=re.IGNORECASE) else 0 for p in patterns)
    if "егэ" in q and ("балл" in q or "порог" in q or "миним" in q):
        scores["min_scores"] = scores.get("min_scores", 0) + 3
    if "без егэ" in q:
        scores["without_ege"] = scores.get("without_ege", 0) + 3
    if ("льгот" in q or "квот" in q) and "платн" not in q:
        scores["benefits"] = scores.get("benefits", 0) + 2
    if ("контакт" in q or "телефон" in q or "адрес" in q):
        scores["contacts"] = scores.get("contacts", 0) + 2
    best_intent, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else ("general", 0)
    if best_score <= 0:
        return "general"
    return best_intent


def infer_question_topics(question: str) -> Set[str]:
    intent = infer_question_intent(question)
    topics = set(INTENT_TO_TOPICS.get(intent, set()))
    q = _norm(question)
    if "документ" in q:
        topics.add("documents")
    if "срок" in q or "до какого" in q:
        topics.add("deadlines")
    if "егэ" in q or "экзам" in q:
        topics.add("exams")
    return topics or {"general"}


def infer_question_programs(question: str) -> Set[str]:
    q = _norm(question)
    out = {program for program, patterns in PROGRAM_PATTERNS.items() if _has_any_pattern(q, patterns)}
    return out or {"all"}


def infer_doc_kind(text: str) -> str:
    t = _norm(text)
    for kind, patterns in DOC_KIND_RULES:
        if _has_any_pattern(t, patterns):
            return kind
    if "приказ" in t:
        return "ministry_order"
    if "федеральный закон" in t or re.search(r"\b\d+\s*\-?фз\b", t, flags=re.IGNORECASE):
        return "federal_law"
    return "other"


def infer_program_scope(text: str) -> List[str]:
    t = _norm(text)
    out = [program for program, patterns in PROGRAM_PATTERNS.items() if _has_any_pattern(t, patterns)]
    return sorted(set(out)) if out else ["all"]


def infer_topic_scope(text: str, doc_kind: str) -> List[str]:
    t = _norm(text)
    out: Set[str] = set()
    if any(x in t for x in ["документ", "заявлен", "оригинал", "копи"]):
        out.add("documents")
    if any(x in t for x in ["срок", "до ", "завершени", "дата"]):
        out.add("deadlines")
    if any(x in t for x in ["подать", "подач", "направить", "госуслуг", "почт"]):
        out.add("apply")
    if any(x in t for x in ["егэ", "экзам", "испытан", "балл"]):
        out.add("exams")
    if any(x in t for x in ["льгот", "квот", "целев", "особое право", "преимуществен"]):
        out.add("benefits")
    if any(x in t for x in ["может", "имеет право", "допуска", "без егэ", "после спо"]):
        out.add("eligibility")
    if any(x in t for x in ["адрес", "телефон", "email", "почт", "кабинет", "контакт"]):
        out.add("contacts")

    out |= {
        "rules_of_admission": {"documents", "deadlines", "apply", "exams", "eligibility", "benefits", "min_scores"},
        "admission_attachment": {"documents", "deadlines", "apply", "exams", "min_scores", "benefits"},
        "admission_order": {"documents", "deadlines", "apply", "exams", "eligibility", "benefits", "min_scores"},
        "exam_rules": {"exams", "eligibility", "min_scores"},
        "benefits": {"benefits", "eligibility"},
        "contacts": {"contacts"},
        "education_law": {"eligibility", "benefits"},
        "service_law": {"eligibility"},
        "health_requirements": {"documents", "eligibility"},
    }.get(doc_kind, set())
    return sorted(out)


def infer_profile_flags(doc_kind: str) -> Dict[str, Any]:
    return {
        "is_general_law": doc_kind in {"education_law", "service_law", "federal_law", "charter"},
        "is_local_admission_rule": doc_kind in {"rules_of_admission", "admission_attachment"},
        "is_contact_source": doc_kind == "contacts",
    }


def resolve_document_profile(*, source: Optional[str], source_file: Optional[str], title: Optional[str], doc_type: Optional[str], preview_text: Optional[str], base_legal_force: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    combined = " \n".join(x for x in [title, source, doc_type, preview_text, source_file] if x)
    doc_kind = infer_doc_kind(combined)
    profile: Dict[str, Any] = {
        "doc_kind": doc_kind,
        "program_scope": infer_program_scope(combined),
        "topic_scope": infer_topic_scope(combined, doc_kind),
    }
    profile.update(infer_profile_flags(doc_kind))

    entry = get_registry_entry(source=source, source_file=source_file, title=title)
    if entry:
        for key, value in entry.items():
            if value is not None:
                profile[key] = value

    if base_legal_force:
        profile.setdefault("legal_force_code", base_legal_force.get("code"))
        profile.setdefault("legal_force_name", base_legal_force.get("label"))
        profile.setdefault("legal_force_level", base_legal_force.get("level"))

    if isinstance(profile.get("program_scope"), str):
        profile["program_scope"] = [profile["program_scope"]]
    if isinstance(profile.get("topic_scope"), str):
        profile["topic_scope"] = [profile["topic_scope"]]

    flags = infer_profile_flags(str(profile.get("doc_kind") or "other"))
    for key, value in flags.items():
        profile.setdefault(key, value)

    return profile


def _has_negative_marker(intent: str, haystack: str) -> bool:
    return any(re.search(p, haystack, flags=re.IGNORECASE) for p in NEGATIVE_MARKERS.get(intent, []))


def should_exclude_payload_for_question(question: str, payload: Dict[str, Any], chunk_text: str) -> bool:
    q = _norm(question)
    intent = infer_question_intent(question)
    q_topics = infer_question_topics(question)
    q_programs = infer_question_programs(question)
    p_topics = set(payload.get("topic_scope") or [])
    p_programs = set(payload.get("program_scope") or []) or {"all"}
    doc_kind = str(payload.get("doc_kind") or "")
    haystack = _norm(" \n".join(str(x) for x in [payload.get("doc_title"), payload.get("source"), payload.get("doc_type"), chunk_text] if x))

    if intent == "contacts":
        return not bool(payload.get("is_contact_source"))

    if "all" not in q_programs and "all" not in p_programs and not (q_programs & p_programs):
        return True

    if intent in {"documents", "deadlines", "apply"}:
        if "о порядке рассмотрения обращений граждан" in haystack:
            return True
        if doc_kind in {"charter", "education_law", "service_law", "records_retention", "contacts", "federal_law"}:
            return True
        if doc_kind == "health_requirements" and not any(x in q for x in ["медицин", "ввк", "врачеб", "осмотр"]):
            return True
        if p_topics and not (p_topics & {"documents", "deadlines", "apply", "exams"}):
            return True

    if intent in {"exams", "min_scores"}:
        if doc_kind in {"charter", "records_retention", "contacts", "federal_law"}:
            return True
        if doc_kind == "health_requirements" and not any(x in q for x in ["медицин", "ввк", "врачеб", "осмотр"]):
            return True
        if p_topics and not (p_topics & {"exams", "eligibility", "documents"}):
            return True
        if intent == "min_scores":
            if _has_negative_marker("min_scores", haystack):
                return True
            if re.search(r"(контрольн(?:ые|ых)\s+норматив|физическ(?:ая|ой)\s+подготов|упражнен)", haystack, flags=re.IGNORECASE):
                return True
            if not re.search(r"(минимальн|егэ|балл|общеобразовательн|русск(?:ий|ого)\s+язык|обществознани|истори|вступительн(?:ые|ых)\s+испытан)", haystack, flags=re.IGNORECASE):
                return True

    if intent == "benefits":
        if doc_kind in {"records_retention", "health_requirements", "contacts", "federal_law"}:
            return True
        if _has_negative_marker("benefits", haystack):
            return True
        if p_topics and not (p_topics & {"benefits", "eligibility"}):
            return True

    if intent in {"eligibility", "without_ege"}:
        if doc_kind in {"records_retention", "contacts", "federal_law"}:
            return True
        if doc_kind == "health_requirements" and not any(x in q for x in ["медицин", "ввк", "врачеб", "осмотр"]):
            return True
        if p_topics and not (p_topics & {"eligibility", "benefits", "exams"}):
            return True
        if intent == "without_ege" and not re.search(r"(без\s+егэ|внутренн(?:ие|их)\s+(?:испытан|экзамен)|результат[а-я\s]+егэ|после\s+спо|после\s+колледжа)", haystack, flags=re.IGNORECASE):
            return True

    if intent == "normative":
        if doc_kind in {"records_retention", "contacts"}:
            return True

    if "как подать" in q and ("военно-врачеб" in haystack or "ввк" in haystack):
        return True
    if ("документ" in q or "подать" in q) and "сроков их хранения" in haystack:
        return True
    if "специалитет" in q and p_programs == {"adjunct"}:
        return True

    return False


def intent_keywords(intent: str) -> List[str]:
    return list(INTENT_KEYWORDS.get(intent, []))


def build_query_variants(question: str) -> List[str]:
    q = _norm(question)
    if not q:
        return []
    intent = infer_question_intent(question)
    variants: List[str] = [re.sub(r"\s+", " ", question).strip()]
    for extra in INTENT_QUERY_EXPANSIONS.get(intent, []):
        variants.append(f"{question} {extra}".strip())
    # compact lexical variant similar to the keyword-focused rewrites used in production search stacks
    terms = []
    seen = set()
    for token in re.findall(r"[a-zа-я0-9]+", q, flags=re.IGNORECASE):
        if len(token) < 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= 8:
            break
    if terms:
        variants.append(" ".join(terms))
    # intent-first synthetic query
    kw = intent_keywords(intent)
    if kw:
        variants.append(" ".join(dict.fromkeys([question, *kw])))
    out: List[str] = []
    seenv = set()
    for v in variants:
        normv = _norm(v)
        if not normv or normv in seenv:
            continue
        seenv.add(normv)
        out.append(v.strip())
    return out[:4]


def question_requires_strict_grounding(question: str) -> bool:
    return infer_question_intent(question) in RISKY_INTENTS


def wants_combined_clarification(question: str) -> bool:
    return infer_question_intent(question) in {"exams", "min_scores", "benefits", "eligibility", "without_ege"}


def build_combined_clarification(question: str, missing: List[str]) -> str:
    labels: List[str] = []
    if "level" in missing:
        labels.append("уровень обучения")
    if "status" in missing:
        labels.append("категорию поступающего")
    if "form" in missing:
        labels.append("форму обучения")
    if not labels:
        return "Уточните, пожалуйста, несколько деталей, чтобы я ответил точнее."
    hints: List[str] = []
    if "level" in missing:
        hints.append("специалитет, бакалавриат, магистратура или адъюнктура")
    if "status" in missing:
        hints.append("поступаете впервые, после СПО или как действующий сотрудник")
    if "form" in missing:
        hints.append("очная, заочная или дистанционная")
    return f"Чтобы ответить точнее, уточните {', '.join(labels)}. Например: {'; '.join(hints)}."


def answer_from_contacts(question: str) -> Optional[Dict[str, Any]]:
    if infer_question_intent(question) != "contacts":
        return None

    data = load_contacts_data()
    contacts = data.get("contacts") if isinstance(data.get("contacts"), dict) else data
    if not isinstance(contacts, dict) or not contacts:
        msg = "В текущей базе контактов нет подтвержденных данных о приемной комиссии. Добавьте contacts.json с адресом, телефоном, email и часами работы."
        return {
            "voice_answer": msg,
            "answer": msg,
            "citations": [{"source": "contacts.json", "point": None, "pages": None, "score": 1.0}],
            "meta": {"engine": "structured", "question_profile": "organizational", "question_intent": "contacts", "handoff_recommended": True},
        }

    fields = {
        "address": contacts.get("address"),
        "office": contacts.get("office"),
        "hours": contacts.get("hours"),
        "phone": contacts.get("phone"),
        "email": contacts.get("email"),
        "website": contacts.get("website"),
    }
    pieces: List[str] = []
    if fields["address"]:
        pieces.append(f"Адрес: {fields['address']}.")
    if fields["office"]:
        pieces.append(f"Кабинет: {fields['office']}.")
    if fields["hours"]:
        pieces.append(f"Часы работы: {fields['hours']}.")
    if fields["phone"]:
        pieces.append(f"Телефон: {fields['phone']}.")
    if fields["email"]:
        pieces.append(f"Email: {fields['email']}.")
    if fields["website"]:
        pieces.append(f"Сайт: {fields['website']}.")

    if not pieces:
        msg = "В contacts.json нет заполненных полей для ответа на организационный вопрос."
        return {
            "voice_answer": msg,
            "answer": msg,
            "citations": [{"source": "contacts.json", "point": None, "pages": None, "score": 1.0}],
            "meta": {"engine": "structured", "question_profile": "organizational", "question_intent": "contacts", "handoff_recommended": True},
        }

    answer = " ".join(pieces)
    return {
        "voice_answer": pieces[0],
        "answer": answer,
        "citations": [{"source": "contacts.json", "point": None, "pages": None, "score": 1.0}],
        "meta": {"engine": "structured", "question_profile": "organizational", "question_intent": "contacts"},
    }
