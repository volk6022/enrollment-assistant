"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion (RRF).

RRF avoids comparing incomparable score scales (BM25 vs cosine): it sums
1/(k+rank) across each ranked list. This replaces the legacy tower of hand-tuned
metadata score-bonuses — fusion here is pure rank arithmetic, and quality comes
from the downstream reranker.
"""
from __future__ import annotations

from rag.config import DEFAULT, RagConfig
from rag.embed import embed_query
from rag.index import Indexes


def rrf_fuse(*ranked_lists: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (idx, _) in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query: str, idx: Indexes, cfg: RagConfig = DEFAULT) -> list[dict]:
    qvec = embed_query(query, cfg)
    dense = idx.dense_search(qvec, cfg.dense_top)
    sparse = idx.bm25_search(query, cfg.bm25_top)
    fused = rrf_fuse(dense, sparse, k=cfg.rrf_k)[: cfg.fused_top]

    out = []
    for i, rrf_score in fused:
        c = dict(idx.chunks[i])
        c["rrf_score"] = rrf_score
        out.append(c)
    return out
