from __future__ import annotations

from ..search.vector_store_client import SearchHit


def build_citations(hits: list[SearchHit]) -> list[dict]:
    citations = []
    for hit in hits[:5]:
        payload = hit.payload or {}
        citations.append(
            {
                "point": payload.get("point"),
                "pages": payload.get("pages"),
                "source": payload.get("source"),
                "score": round(float(hit.score), 4),
                "doc_title": payload.get("doc_title") or payload.get("title"),
                "doc_number": payload.get("doc_number"),
                "issued_date": payload.get("issued_date"),
                "revision_date": payload.get("revision_date"),
                "legal_force_name": payload.get("legal_force_name"),
                "legal_force_level": payload.get("legal_force_level"),
                "doc_kind": payload.get("doc_kind"),
                "program_scope": payload.get("program_scope"),
                "topic_scope": payload.get("topic_scope"),
            }
        )
    return citations
