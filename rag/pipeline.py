"""End-to-end RAG pipeline with per-step timing.

query → embed → (dense ∥ bm25) → RRF → rerank → generate
Each stage is timed separately so the experiment grid can see where latency goes
(the guideline budgets <1s for everything up to and excluding generation).
"""
from __future__ import annotations

import time

from rag.config import DEFAULT, RagConfig
from rag.embed import embed_query
from rag.generate import LlamaServer, build_messages
from rag.index import Indexes
from rag.rephrase import rephrase_canonical
from rag.rerank import rerank
from rag.retrieve import rrf_fuse


class Pipeline:
    def __init__(self, idx: Indexes, server: LlamaServer | None = None, cfg: RagConfig = DEFAULT):
        self.idx = idx
        self.server = server
        self.cfg = cfg

    def retrieve_timed(self, query: str) -> tuple[list[dict], dict]:
        cfg = self.cfg
        t = {}
        t0 = time.perf_counter()
        qvec = embed_query(query, cfg)
        t["embed_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dense = self.idx.dense_search(qvec, cfg.dense_top)
        t["dense_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sparse = self.idx.bm25_search(query, cfg.bm25_top)
        t["bm25_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf_fuse(dense, sparse, k=cfg.rrf_k)[: cfg.fused_top]
        t["rrf_ms"] = (time.perf_counter() - t0) * 1000
        candidates = []
        for i, s in fused:
            c = dict(self.idx.chunks[i])
            c["rrf_score"] = s
            candidates.append(c)
        return candidates, t

    def retrieve_union_timed(self, queries: list[str]) -> tuple[list[dict], dict]:
        """Multi-query retrieval: RRF-union the dense+sparse lists of several
        queries into one candidate pool. Used by conversational mode to combine
        the spoken query with its canonical rewrite."""
        cfg = self.cfg
        t = {"embed_ms": 0.0, "dense_ms": 0.0, "bm25_ms": 0.0}
        ranked = []
        for q in queries:
            if not q or not q.strip():
                continue
            t0 = time.perf_counter()
            qvec = embed_query(q, cfg)
            t["embed_ms"] += (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            ranked.append(self.idx.dense_search(qvec, cfg.dense_top))
            t["dense_ms"] += (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            ranked.append(self.idx.bm25_search(q, cfg.bm25_top))
            t["bm25_ms"] += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf_fuse(*ranked, k=cfg.rrf_k)[: cfg.fused_top]
        t["rrf_ms"] = (time.perf_counter() - t0) * 1000
        candidates = []
        for i, s in fused:
            c = dict(self.idx.chunks[i])
            c["rrf_score"] = s
            candidates.append(c)
        return candidates, t

    def search(self, query: str) -> tuple[list[dict], dict]:
        """Retrieval + rerank (no generation) — used for retrieval eval."""
        candidates, t = self.retrieve_timed(query)
        t0 = time.perf_counter()
        top = rerank(query, candidates, self.cfg)
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000
        t["search_ms"] = t["embed_ms"] + t["dense_ms"] + t["bm25_ms"] + t["rrf_ms"] + t["rerank_ms"]
        return top, t

    def search_conversational(self, query: str) -> tuple[list[dict], str, dict]:
        """Spoken-input retrieval: rephrase -> union(spoken, canonical) ->
        rerank with the CLEAN canonical query. Returns (top, canonical, timings)."""
        t = {}
        t0 = time.perf_counter()
        canonical = rephrase_canonical(self.server, query, self.cfg)
        t["rephrase_ms"] = (time.perf_counter() - t0) * 1000

        candidates, tr = self.retrieve_union_timed([query, canonical])
        t.update(tr)
        t0 = time.perf_counter()
        top = rerank(canonical, candidates, self.cfg)  # clean query orders best
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000
        t["search_ms"] = (t["rephrase_ms"] + t["embed_ms"] + t["dense_ms"]
                          + t["bm25_ms"] + t["rrf_ms"] + t["rerank_ms"])
        return top, canonical, t

    def answer(self, query: str) -> dict:
        canonical = None
        if self.cfg.conversational and self.server is not None:
            top, canonical, t = self.search_conversational(query)
        else:
            top, t = self.search(query)
        gen = {}
        answer = ""
        if self.server is not None:
            messages = build_messages(query, top)  # generate on the ORIGINAL query
            gen = self.server.complete(messages, self.cfg)
            answer = gen["answer"]
            t["gen_ms"] = gen["gen_sec"] * 1000
        return {
            "canonical_query": canonical,
            "question": query,
            "answer": answer,
            "citations": [
                {"source": c["source"], "point": c.get("point"), "rerank_score": c.get("rerank_score")}
                for c in top
            ],
            "top_chunks": top,
            "timings": t,
            "gen": gen,
        }
