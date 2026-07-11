import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from .knowledge import build_query_variants, infer_question_intent, infer_question_programs, infer_question_topics, should_exclude_payload_for_question
from .legal_hierarchy import (
    is_documents_question,
    is_general_law_document,
    is_procedural_document,
    legal_force_bonus,
    mentions_narrow_category,
    question_profile,
)


@dataclass
class Hit:
    text: str
    payload: Dict[str, Any]
    score: float
    collection: str


def _tokenize_ru(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-zа-я0-9]+", " ", text, flags=re.IGNORECASE)
    return [t for t in text.split() if len(t) >= 2]


def _question_years(q: str) -> List[int]:
    return sorted({int(x) for x in re.findall(r"\b(20\d{2})\b", q or "")})


def _wants_revision(q: str) -> bool:
    ql = (q or "").lower()
    return any(x in ql for x in ["редакц", "в ред", "изменен", "изменени", "ред.", "ред "])


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [x.strip() for x in raw if x and x.strip()]


def _payload_haystack(payload: Dict[str, Any], text: str = "") -> str:
    parts = [payload.get("doc_title") or "", payload.get("source") or "", payload.get("doc_type") or "", payload.get("section_path") or "", text or payload.get("text") or ""]
    if isinstance(parts[3], list):
        parts[3] = " ".join(str(x) for x in parts[3])
    return " \n".join(str(x) for x in parts if x)


def _has_any(text: str, patterns: List[str]) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in patterns)


def _candidate_key(h: "Hit") -> str:
    pl = h.payload or {}
    return f"{h.collection}|{pl.get('source')}|{pl.get('point')}|{pl.get('pages')}|{hash(h.text)}"


def _point_tuple(point: Any) -> tuple[int, ...]:
    if point is None:
        return ()
    nums = re.findall(r"\d+", str(point))
    return tuple(int(x) for x in nums)


def _point_distance(a: Any, b: Any) -> Optional[int]:
    ta = _point_tuple(a)
    tb = _point_tuple(b)
    if not ta or not tb or len(ta) != len(tb):
        return None
    if ta[:-1] != tb[:-1]:
        return None
    return abs(ta[-1] - tb[-1])


def _section_prefix(section_path: Any) -> tuple[str, ...]:
    if not isinstance(section_path, list):
        return ()
    return tuple(str(x).strip().lower() for x in section_path[:3] if str(x).strip())


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _topic_overlap(question: str, payload: Dict[str, Any]) -> float:
    intent = infer_question_intent(question)
    q_topics = infer_question_topics(question)
    p_topics = set(payload.get("topic_scope") or [])
    if intent == "min_scores" and "min_scores" in p_topics:
        return 0.24
    if not q_topics or not p_topics or "general" in q_topics:
        return 0.0
    inter = q_topics & p_topics
    if not inter:
        return -0.14
    return 0.14 + 0.04 * max(0, len(inter) - 1)


def _program_overlap(question: str, payload: Dict[str, Any]) -> float:
    q_programs = infer_question_programs(question)
    p_programs = set(payload.get("program_scope") or []) or {"all"}
    if p_programs == {"adjunct"} and "adjunct" not in q_programs and "all" in q_programs:
        return -0.30
    if "all" in q_programs or "all" in p_programs:
        return 0.0
    if q_programs & p_programs:
        return 0.14
    return -0.18


def is_irrelevant_for_question(question: str, chunk_text: str, payload: Optional[Dict[str, Any]] = None) -> bool:
    payload = payload or {}
    q = (question or "").lower()
    t = (chunk_text or "").lower()
    haystack = _payload_haystack(payload, chunk_text).lower()
    profile = question_profile(question)

    if should_exclude_payload_for_question(question, payload, chunk_text):
        return True
    if ("физ" in q or "упраж" in q) and not ("физ" in t or "упраж" in t):
        return True
    p_programs = set(payload.get("program_scope") or [])
    if p_programs == {"adjunct"} and not any(x in q for x in ["адъюнкт", "аспиран", "научн"]):
        return True
    if profile == "procedural":
        if is_general_law_document(haystack) and not is_procedural_document(haystack) and not payload.get("is_local_admission_rule"):
            return True
        if is_documents_question(question) and mentions_narrow_category(haystack) and not mentions_narrow_category(q):
            return True
    if profile == "organizational" and not payload.get("is_contact_source"):
        return True
    return False


class RAG:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embedder = TextEmbedding(self.embed_model)
        self._bm25: Dict[str, BM25Okapi] = {}
        self._bm25_texts: Dict[str, List[str]] = {}
        self._bm25_payloads: Dict[str, List[Dict[str, Any]]] = {}

    def _embed(self, text: str) -> List[float]:
        vec = next(self.embedder.embed([text]))
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def refresh_cache(self, collection: str, limit: int = 20000) -> int:
        texts: List[str] = []
        payloads: List[Dict[str, Any]] = []
        offset = None
        total = 0
        while True:
            points, offset = self.client.scroll(collection_name=collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
            if not points:
                break
            for p in points:
                pl = p.payload or {}
                txt = (pl.get("text") or "").strip()
                if not txt:
                    continue
                texts.append(txt)
                payloads.append(pl)
                total += 1
                if total >= limit:
                    break
            if total >= limit or offset is None:
                break
        tokenized = [_tokenize_ru(t) for t in texts]
        self._bm25[collection] = BM25Okapi(tokenized) if texts else BM25Okapi([["empty"]])
        self._bm25_texts[collection] = texts
        self._bm25_payloads[collection] = payloads
        return len(texts)

    def _ensure_bm25(self, collection: str) -> None:
        if collection not in self._bm25:
            try:
                self.refresh_cache(collection)
            except Exception:
                pass

    def _bm25_candidates(self, query: str, collection: str, top_n: int = 25) -> List[Hit]:
        self._ensure_bm25(collection)
        if collection not in self._bm25:
            return []
        bm25 = self._bm25[collection]
        texts = self._bm25_texts.get(collection, [])
        payloads = self._bm25_payloads.get(collection, [])
        if not texts:
            return []
        q_tok = _tokenize_ru(query)
        scores = bm25.get_scores(q_tok)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [Hit(text=texts[i], payload=payloads[i], score=float(scores[i]), collection=collection) for i in idxs]

    def _edition_bonus(self, question: str, payload: Dict[str, Any]) -> float:
        years = _question_years(question)
        wants = _wants_revision(question)
        rev_year = payload.get("revision_year")
        if rev_year is None and payload.get("revision_date"):
            try:
                rev_year = int(str(payload["revision_date"])[:4])
            except Exception:
                rev_year = None
        if years or wants:
            if rev_year is None:
                return -0.02
            target = max(years) if years else 2025
            if rev_year >= target:
                return 0.12
            if rev_year == target - 1:
                return 0.05
            return -0.02
        if rev_year is None:
            return 0.0
        if rev_year >= 2025:
            return 0.04
        if rev_year >= 2023:
            return 0.02
        return 0.0

    def _source_specific_bonus(self, question: str, payload: Dict[str, Any], text: str) -> float:
        ql = (question or "").lower()
        force_code = (payload.get("legal_force_code") or "").lower()
        haystack = _payload_haystack(payload, text).lower()
        profile = question_profile(question)
        intent = infer_question_intent(question)
        doc_kind = str(payload.get("doc_kind") or "")
        bonus = _topic_overlap(question, payload) + _program_overlap(question, payload)

        if profile == "procedural":
            if payload.get("is_local_admission_rule") or doc_kind in {"admission_order", "admission_attachment"}:
                bonus += 0.28
            if is_procedural_document(haystack):
                bonus += 0.12
            if payload.get("is_general_law") and not payload.get("is_local_admission_rule"):
                bonus -= 0.20
            if intent == "documents":
                if doc_kind in {"rules_of_admission", "admission_attachment", "admission_order"}:
                    bonus += 0.16
                if _has_any(haystack, ["переч", "документ", "заявлен", "копи", "оригинал"]):
                    bonus += 0.22
                if mentions_narrow_category(haystack) and not mentions_narrow_category(question):
                    bonus -= 0.26
            if intent == "deadlines":
                if re.search(r"\b\d{1,2}[. ](?:0?\d|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b", haystack, flags=re.IGNORECASE):
                    bonus += 0.20
            if intent == "apply":
                if doc_kind in {"rules_of_admission", "admission_attachment", "admission_order"}:
                    bonus += 0.16
                if _has_any(haystack, ["подать", "подача", "направить", "почт", "госуслуг", "личный кабинет", "лично"]):
                    bonus += 0.24
            if intent == "exams":
                if _has_any(haystack, ["вступительн", "егэ", "русский язык", "обществозн", "физическ"]):
                    bonus += 0.16
            if intent == "min_scores":
                if doc_kind in {"rules_of_admission", "admission_attachment", "admission_order", "exam_rules"}:
                    bonus += 0.24
                if _has_any(haystack, ["минимальн", "балл", "егэ", "русский язык", "обществозн", "истори", "общеобразовательн"]):
                    bonus += 0.32
                if _has_any(haystack, ["индивидуальн", "достижен", "диплом с отличием", "гто", "портфолио", "дополнительн", "контрольн", "норматив", "упражнен", "физическ"]):
                    bonus -= 0.46
            if doc_kind in {"records_retention", "charter", "health_requirements"}:
                bonus -= 0.26
            if force_code in {"local_rules", "ministry_order", "government_resolution"}:
                bonus += 0.04
        elif profile == "normative":
            if force_code in {"constitution", "international_treaty", "fkz", "federal_law"}:
                bonus += 0.10
            if doc_kind in {"rules_of_admission", "admission_order", "admission_attachment"}:
                bonus += 0.03
        elif profile == "organizational":
            bonus += 0.40 if payload.get("is_contact_source") else -0.20
        elif profile == "eligibility":
            if doc_kind in {"rules_of_admission", "admission_order", "benefits", "education_law", "service_law", "exam_rules"}:
                bonus += 0.12
            if doc_kind in {"records_retention", "contacts"}:
                bonus -= 0.22
            if intent == "without_ege":
                if doc_kind in {"rules_of_admission", "admission_order", "exam_rules"}:
                    bonus += 0.12
                if _has_any(haystack, ["без егэ", "внутренн", "вступительн", "результаты егэ", "после спо", "после колледжа"]):
                    bonus += 0.22
                else:
                    bonus -= 0.20
            if intent == "benefits":
                if _has_any(haystack, ["льгот", "квот", "особое право", "преимуществен", "без вступительных", "целев"]):
                    bonus += 0.20
                if _has_any(haystack, ["платн", "договор об образовании"]):
                    bonus -= 0.20
            if _has_any(haystack, ["имеет право", "допуска", "льгот", "квот", "без вступительных", "особое право", "вступительн"]):
                bonus += 0.08
        if "общежит" in ql and "общежит" not in haystack:
            bonus -= 0.05
        return bonus

    def _score_hit(self, question: str, h: Hit) -> float:
        q_tok = set(_tokenize_ru(question))
        t_tok = set(_tokenize_ru(h.text))
        overlap = len(q_tok & t_tok) / max(1, len(q_tok))
        return float(h.score) + 0.45 * overlap + self._edition_bonus(question, h.payload) + legal_force_bonus(question, h.payload) + self._source_specific_bonus(question, h.payload, h.text)

    def _search_one_collection(self, question: str, collection: str, top_k: int) -> List[Hit]:
        query_variants = build_query_variants(question) or [question]
        aggregate: Dict[str, Dict[str, Any]] = {}

        def add_ranked(hits: List[Hit], variant_idx: int, modality_weight: float) -> None:
            for rank, h in enumerate(hits, start=1):
                key = _candidate_key(h)
                rec = aggregate.setdefault(key, {"hit": h, "rrf": 0.0, "variants": set(), "best_raw": -1e9})
                rec["rrf"] += modality_weight * _rrf(rank)
                rec["variants"].add(variant_idx)
                rec["best_raw"] = max(rec["best_raw"], float(h.score))

        for variant_idx, query in enumerate(query_variants):
            try:
                vec = self._embed(query)
                vhits = self.client.search(collection_name=collection, query_vector=vec, limit=max(24, top_k * 8), with_payload=True)
            except Exception:
                vhits = []
            vec_hits: List[Hit] = []
            for p in vhits:
                pl = p.payload or {}
                txt = (pl.get("text") or "").strip()
                if txt:
                    vec_hits.append(Hit(text=txt, payload=pl, score=float(p.score), collection=collection))
            bm25_hits = self._bm25_candidates(query, collection, top_n=max(30, top_k * 10))
            add_ranked(vec_hits, variant_idx, modality_weight=1.0)
            add_ranked(bm25_hits, variant_idx, modality_weight=1.0)

        candidates: List[Hit] = []
        for rec in aggregate.values():
            h = rec["hit"]
            if is_irrelevant_for_question(question, h.text, h.payload):
                continue
            temp = Hit(text=h.text, payload=h.payload, score=0.0, collection=h.collection)
            feature_score = self._score_hit(question, temp)
            coverage_bonus = min(0.18, 0.06 * max(0, len(rec["variants"]) - 1))
            h.score = 4.0 * rec["rrf"] + feature_score + coverage_bonus
            candidates.append(h)

        if not candidates:
            # fall back to unfiltered candidates if the filters were too aggressive
            for rec in aggregate.values():
                h = rec["hit"]
                temp = Hit(text=h.text, payload=h.payload, score=0.0, collection=h.collection)
                h.score = 4.0 * rec["rrf"] + self._score_hit(question, temp)
                candidates.append(h)

        candidates.sort(key=lambda h: h.score, reverse=True)
        return candidates[:top_k]

    def _collections_for_search(self, primary: Optional[str]) -> List[str]:
        collections: List[str] = []
        if primary:
            collections.append(primary)
        env_candidates = [
            os.getenv("QDRANT_COLLECTION_LEGAL", "").strip(),
            os.getenv("QDRANT_COLLECTION_NPA", "").strip(),
            os.getenv("QDRANT_COLLECTION", "").strip(),
        ]
        for c in env_candidates:
            if c and c not in collections:
                collections.append(c)
        return collections

    def _neighbor_hits(self, question: str, seeds: List[Hit], max_extra: int = 8) -> List[Hit]:
        extras: List[Hit] = []
        seen = {_candidate_key(h) for h in seeds}
        for seed in seeds[:3]:
            src = str((seed.payload or {}).get("source") or "")
            if not src:
                continue
            collection = seed.collection
            self._ensure_bm25(collection)
            texts = self._bm25_texts.get(collection, [])
            payloads = self._bm25_payloads.get(collection, [])
            seed_point = (seed.payload or {}).get("point")
            seed_section = _section_prefix((seed.payload or {}).get("section_path"))
            seed_kind = str((seed.payload or {}).get("doc_kind") or "")
            for txt, pl in zip(texts, payloads):
                if str(pl.get("source") or "") != src:
                    continue
                if should_exclude_payload_for_question(question, pl, txt):
                    continue
                dist = _point_distance(seed_point, pl.get("point"))
                section_bonus = 0.0
                if seed_section and _section_prefix(pl.get("section_path")) == seed_section:
                    section_bonus = 0.16
                elif seed_section and _section_prefix(pl.get("section_path"))[:1] == seed_section[:1]:
                    section_bonus = 0.08
                point_bonus = 0.0
                if dist is not None and dist <= 2:
                    point_bonus = 0.24 - 0.08 * dist
                elif seed_point and pl.get("point") == seed_point:
                    point_bonus = 0.18
                if max(section_bonus, point_bonus) <= 0.0:
                    continue
                hit = Hit(text=txt, payload=pl, score=seed.score + max(section_bonus, point_bonus) - 0.10, collection=collection)
                if seed_kind and seed_kind == str(pl.get("doc_kind") or ""):
                    hit.score += 0.04
                key = _candidate_key(hit)
                if key in seen:
                    continue
                seen.add(key)
                extras.append(hit)
                if len(extras) >= max_extra:
                    break
            if len(extras) >= max_extra:
                break
        extras.sort(key=lambda h: h.score, reverse=True)
        return extras[:max_extra]

    def _select_diverse_hits(self, question: str, hits: List[Hit], top_k: int) -> List[Hit]:
        profile = question_profile(question)
        out: List[Hit] = []
        seen_keys = set()
        seen_force = set()
        per_source: Dict[str, int] = {}

        def add_hit(h: Hit) -> bool:
            key = (h.collection, h.payload.get("source"), h.payload.get("point"), h.payload.get("pages"), h.text[:120])
            if key in seen_keys:
                return False
            src = str(h.payload.get("source") or "")
            limit_per_source = 3 if profile == "procedural" else 2
            if per_source.get(src, 0) >= limit_per_source:
                return False
            seen_keys.add(key)
            per_source[src] = per_source.get(src, 0) + 1
            out.append(h)
            return True

        if profile in {"normative", "mixed", "eligibility"}:
            for h in hits:
                force_code = h.payload.get("legal_force_code") or "other"
                if force_code not in seen_force and add_hit(h):
                    seen_force.add(force_code)
                if len(out) >= min(3, top_k):
                    break
        for h in hits:
            add_hit(h)
            if len(out) >= top_k:
                break
        return out[:top_k]

    def search(self, question: str, collection: Optional[str] = None, top_k: int = 5) -> List[Hit]:
        all_hits: List[Hit] = []
        per_col_k = max(10, top_k * 3)
        for col in self._collections_for_search(collection):
            try:
                all_hits.extend(self._search_one_collection(question, col, top_k=per_col_k))
            except Exception:
                continue
        if not all_hits:
            return []
        all_hits.sort(key=lambda h: h.score, reverse=True)
        seeds = self._select_diverse_hits(question, all_hits, top_k=max(top_k, 6))
        expanded = seeds + self._neighbor_hits(question, seeds, max_extra=max(6, top_k + 2))
        expanded.sort(key=lambda h: h.score, reverse=True)
        return self._select_diverse_hits(question, expanded, top_k=max(top_k, 8))

    def has_actionable_hits(self, question: str, hits: List[Hit]) -> bool:
        if not hits:
            return False
        profile = question_profile(question)
        intent = infer_question_intent(question)
        top_hits = hits[:4]
        top_score = float(top_hits[0].score) if top_hits else 0.0
        doc_kinds = {str((h.payload or {}).get("doc_kind") or "") for h in top_hits}
        if profile == "procedural":
            if intent == "min_scores":
                return sum(1 for h in top_hits if str((h.payload or {}).get("doc_kind") or "") in {"rules_of_admission", "admission_attachment", "admission_order", "exam_rules"} and re.search(r"(минимальн|балл|егэ)", (h.text or "").lower())) >= 1
            if is_documents_question(question):
                return any((h.payload or {}).get("is_local_admission_rule") or str((h.payload or {}).get("doc_kind") or "") in {"admission_order", "admission_attachment"} for h in top_hits)
            return any(str((h.payload or {}).get("doc_kind") or "") in {"rules_of_admission", "admission_order", "admission_attachment", "exam_rules"} for h in top_hits) or top_score >= 0.95
        if profile == "organizational":
            return any((h.payload or {}).get("is_contact_source") for h in top_hits)
        if profile == "eligibility":
            return any(k in {"rules_of_admission", "admission_order", "benefits", "education_law", "service_law", "exam_rules"} for k in doc_kinds) and top_score >= 0.35
        if profile == "normative":
            return any(int((h.payload or {}).get("legal_force_level") or 0) >= 200 for h in top_hits)
        return top_score >= 0.8

    def _best_voice_sentence(self, question: str, hits: List[Hit]) -> str:
        q_tok = set(_tokenize_ru(question))
        best_sentence = ""
        best_score = -1.0
        for h in hits[:4]:
            for sent in _split_sentences(h.text):
                if len(sent) < 30:
                    continue
                s_tok = set(_tokenize_ru(sent))
                overlap = len(q_tok & s_tok) / max(1, len(q_tok))
                score = overlap + min(len(sent), 180) / 1000
                if score > best_score:
                    best_sentence = re.sub(r"\s+", " ", sent).strip()
                    best_score = score
        if best_sentence:
            return best_sentence[:197].rstrip() + "…" if len(best_sentence) > 200 else best_sentence
        if not hits:
            return "Не нашёл подходящий фрагмент в документах."
        txt = re.sub(r"\s+", " ", (hits[0].text or "").strip())
        return txt[:197].rstrip() + "…" if len(txt) > 200 else txt

    def format_answer_without_llm(self, question: str, hits: List[Hit]) -> Dict[str, Any]:
        if not hits:
            msg = "Не нашёл подходящий фрагмент в документах. Переформулируйте вопрос или соединитесь с приёмной комиссией."
            return {"voice_answer": msg, "answer": msg, "citations": [], "meta": {"engine": "extractive", "handoff_recommended": True}}
        voice = self._best_voice_sentence(question, hits)
        citations = []
        excerpts = []
        for h in hits[:4]:
            pl = h.payload or {}
            snippet = re.sub(r"\s+", " ", (h.text or "").strip())
            if len(snippet) > 500:
                snippet = snippet[:500].rstrip() + "…"
            citations.append({
                "point": pl.get("point"),
                "pages": pl.get("pages"),
                "source": pl.get("source"),
                "score": round(float(h.score), 4),
                "doc_title": pl.get("doc_title"),
                "doc_number": pl.get("doc_number"),
                "issued_date": pl.get("issued_date"),
                "revision_date": pl.get("revision_date"),
                "legal_force_name": pl.get("legal_force_name"),
                "legal_force_level": pl.get("legal_force_level"),
                "doc_kind": pl.get("doc_kind"),
                "program_scope": pl.get("program_scope"),
                "topic_scope": pl.get("topic_scope"),
            })
            label_parts = []
            if pl.get("legal_force_name"):
                label_parts.append(pl.get("legal_force_name"))
            if pl.get("doc_kind"):
                label_parts.append(f"kind={pl.get('doc_kind')}")
            if pl.get("doc_number") or pl.get("issued_date"):
                label_parts.append(" ".join([x for x in [pl.get("doc_number"), pl.get("issued_date")] if x]))
            if pl.get("revision_date"):
                label_parts.append(f"ред. {pl.get('revision_date')}")
            if pl.get("point"):
                label_parts.append(f"п. {pl.get('point')}")
            if pl.get("pages"):
                label_parts.append(f"стр. {pl.get('pages')}")
            label = ", ".join([x for x in label_parts if x]) or "фрагмент"
            excerpts.append(f"— ({label}; {pl.get('source') or 'document'}) {snippet}")
        full = voice + "\n\nПодтверждение в документах:\n" + "\n".join(excerpts)
        return {"voice_answer": voice, "answer": full, "citations": citations, "meta": {"engine": "extractive", "question_profile": question_profile(question)}}
