"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion (RRF).

Ported from `rag/retrieve.py` -- algorithm unchanged. RRF sums 1/(k+rank)
across each ranked list, which avoids comparing incomparable score scales
(BM25 vs cosine); this is what replaces the legacy hand-tuned metadata
score-bonus tower (see `rag/RAG_MAPPING.md`).
"""
from __future__ import annotations

from backend.rag.config import RagSettingsProtocol
from backend.rag.gguf_encoder_client import GgufEncoderClient
from backend.rag.index import Indexes


def rrf_fuse(*ranked_lists: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (idx, _) in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(
    query: str,
    idx: Indexes,
    encoder: GgufEncoderClient,
    settings: RagSettingsProtocol,
) -> list[dict]:
    qvec = encoder.embed_query(query)
    dense = idx.dense_search(qvec, settings.rag_dense_top)
    sparse = idx.bm25_search(query, settings.rag_bm25_top)
    fused = rrf_fuse(dense, sparse, k=settings.rag_rrf_k)[: settings.rag_fused_top]

    out = []
    for i, rrf_score in fused:
        c = dict(idx.chunks[i])
        c["rrf_score"] = rrf_score
        out.append(c)
    return out
