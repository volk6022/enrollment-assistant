from __future__ import annotations

from collections import defaultdict
from typing import Any

from .extraction import EXTRACTOR_BY_INTENT
from .extraction.base import RetrievalPacket
from .generation.answer_builder import build_answer
from .generation.citations import build_citations
from .intents.classify import IntentClassifier
from .intents.schemas import IntentContext
from .knowledge import answer_from_contacts
from .search.query_rewrite import QueryRewriter
from .search.search_profiles import build_search_profile
from .search.vector_store_client import SearchHit, YandexVectorStoreClient


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for v in value:
            chunk = _safe_text(v)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "body", "chunk"):
            if key in value:
                return _safe_text(value.get(key))
        parts: list[str] = []
        for v in value.values():
            chunk = _safe_text(v)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts)
    return str(value)


class V11Orchestrator:
    def __init__(self) -> None:
        self.classifier = IntentClassifier()
        self.rewriter = QueryRewriter()
        self.search = YandexVectorStoreClient()

    def _expand_sections(self, hits: list[SearchHit]) -> list[str]:
        grouped: dict[tuple[str, str], list[SearchHit]] = defaultdict(list)
        for hit in hits:
            payload = hit.payload or {}
            key = (str(payload.get("source", "")), str(payload.get("section_path", payload.get("point", ""))))
            grouped[key].append(hit)
        sections: list[str] = []
        for _, group in grouped.items():
            ordered = sorted(group, key=lambda h: str((h.payload or {}).get("point", "")))
            section_blob = "\n".join(_safe_text(h.text) for h in ordered if _safe_text(h.text))
            if section_blob.strip():
                sections.append(section_blob)
        if not sections:
            sections = [_safe_text(h.text) for h in hits if _safe_text(h.text)]
        return sections[:8]

    def _run_search(self, ctx: IntentContext, *, top_k: int) -> tuple[list[str], list[SearchHit]]:
        profile = build_search_profile(ctx)
        queries = self.rewriter.expand(ctx, profile.expansions)
        all_hits: list[SearchHit] = []
        seen: set[tuple[str, str]] = set()
        for query in queries:
            for hit in self.search.search(query, top_k=profile.top_k, filters=profile.filters):
                payload = hit.payload or {}
                key = (str(payload.get("source", "")), str(payload.get("point", "")), _safe_text(hit.text)[:120])
                if key in seen:
                    continue
                seen.add(key)
                all_hits.append(hit)
        all_hits.sort(key=lambda h: h.score, reverse=True)
        return queries, all_hits[: max(top_k, 8)]

    def answer(self, question: str, *, top_k: int = 5) -> dict:
        ctx = self.classifier.classify(question)
        if ctx.intent == "contacts":
            contact = answer_from_contacts(question)
            if contact:
                contact.setdefault("tts_text", contact.get("voice_answer") or contact.get("answer"))
                contact.setdefault("meta", {})
                contact["meta"].update({"engine": "yandex_v11", "question_intent": "contacts", "backend_engine": "yandex_v11", "answer_style": "brief_direct"})
                return contact
        queries, hits = self._run_search(ctx, top_k=top_k)
        packet = RetrievalPacket(context=ctx, queries=queries, hits=hits, sections=self._expand_sections(hits))
        extractor_cls = EXTRACTOR_BY_INTENT.get(ctx.intent)
        if extractor_cls is None:
            extracted = {"summary": "\n\n".join(packet.sections[:4]), "source_points": []}
        else:
            extracted = extractor_cls().extract(packet)
        citations = build_citations(hits)
        return build_answer(ctx, extracted, citations)
