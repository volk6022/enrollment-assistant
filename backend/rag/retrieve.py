"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion (RRF).

Ported from `rag/retrieve.py` -- algorithm unchanged. RRF sums 1/(k+rank)
across each ranked list, which avoids comparing incomparable score scales
(BM25 vs cosine); this is what replaces the legacy hand-tuned metadata
score-bonus tower (see `rag/RAG_MAPPING.md`).
"""
from __future__ import annotations

import time

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
    timings: dict[str, float] | None = None,
) -> list[dict]:
    """Same algorithm as before T-12; the only addition is `timings` -- when
    the caller passes a dict, this fills in `embed_ms`/`dense_ms`/`bm25_ms`/
    `rrf_ms` (contracts/websocket.md §3.6's `meta.timings_ms` component
    breakdown, which `RagPipeline.search`'s old single `retrieve_ms` number
    couldn't answer). Optional and additive on purpose -- every existing
    caller that doesn't care about per-component timing keeps working
    unmodified.
    """
    t0 = time.perf_counter()
    qvec = encoder.embed_query(query)
    t1 = time.perf_counter()
    dense = idx.dense_search(qvec, settings.rag_dense_top)
    t2 = time.perf_counter()
    sparse = idx.bm25_search(query, settings.rag_bm25_top)
    t3 = time.perf_counter()
    fused = rrf_fuse(dense, sparse, k=settings.rag_rrf_k)[: settings.rag_fused_top]
    t4 = time.perf_counter()

    if timings is not None:
        timings["embed_ms"] = (t1 - t0) * 1000
        timings["dense_ms"] = (t2 - t1) * 1000
        timings["bm25_ms"] = (t3 - t2) * 1000
        timings["rrf_ms"] = (t4 - t3) * 1000

    out = []
    for i, rrf_score in fused:
        c = dict(idx.chunks[i])
        c["rrf_score"] = rrf_score
        out.append(c)
    return out
