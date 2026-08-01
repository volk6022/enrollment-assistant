"""Neural reranking via bge-reranker-v2-m3 GGUF Q8_0 over llama-server.

Ported from `rag/rerank.py`. The cross-encoder itself moved off-process (GGUF
weights served by llama-server, called over HTTP through `GgufEncoderClient`)
-- the fuse -> rerank -> top-k algorithm is unchanged.
"""
from __future__ import annotations

from backend.rag.config import RagSettingsProtocol
from backend.rag.gguf_encoder_client import GgufEncoderClient


def rerank(
    query: str,
    candidates: list[dict],
    encoder: GgufEncoderClient,
    settings: RagSettingsProtocol,
    top_k: int | None = None,
) -> list[dict]:
    if not candidates:
        return []
    docs = [c["text"] for c in candidates]
    scores = encoder.rerank(query, docs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[: (top_k or settings.rag_top_k)]
