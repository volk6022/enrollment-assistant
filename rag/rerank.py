"""Neural reranking with bge-reranker-v2-m3 (multilingual cross-encoder).

This is the biggest quality lever the legacy pipeline was missing. The
cross-encoder scores each (query, chunk) pair jointly and reorders the fused
candidates; only the final top-k go to the LLM.
"""
from __future__ import annotations

import os

from rag.config import DEFAULT, RagConfig

_RERANKER = None


def get_reranker(cfg: RagConfig = DEFAULT):
    global _RERANKER
    if _RERANKER is None:
        os.environ.setdefault("HF_HOME", cfg.hf_env()["HF_HOME"])
        from sentence_transformers import CrossEncoder

        m = CrossEncoder(cfg.reranker_model, device=cfg.device, max_length=cfg.rerank_max_length)
        if cfg.rerank_fp16 and cfg.device == "cuda":
            m.model.half()
        m.predict([("прогрев", "прогрев модели")] * 8, batch_size=cfg.rerank_batch_size)  # warm CUDA
        _RERANKER = m
    return _RERANKER


def rerank(query: str, candidates: list[dict], cfg: RagConfig = DEFAULT, top_k: int | None = None) -> list[dict]:
    if not candidates:
        return []
    model = get_reranker(cfg)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs, batch_size=cfg.rerank_batch_size, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[: (top_k or cfg.final_top)]
