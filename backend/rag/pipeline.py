"""End-to-end retrieval pipeline: query -> embed -> (dense || bm25) -> RRF ->
rerank -> top-k chunks.

Ported from `rag/pipeline.py`, minus answer generation -- LLM answer
generation is `backend/llm`'s job (T-03/T-09, contracts/llm.md); this package
only retrieves and reranks. `rag/generate.py` and `rag/rephrase.py` have no
equivalent here.

Call shape matches plan.md §2/§9: the "rag" pool is one synchronous worker.
`RagPipeline.search` is a plain blocking method -- call it directly from
inside that pool's worker thread. `RagPipeline.asearch` is the thin async
wrapper for callers on the event loop (T-09's session orchestration): it just
does `run_in_executor`, it contains no pipeline logic of its own.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import Executor
from pathlib import Path

from backend.rag.config import DEFAULT, RagSettingsProtocol
from backend.rag.gguf_encoder_client import GgufEncoderClient
from backend.rag.index import Indexes
from backend.rag.rerank import rerank as _rerank
from backend.rag.retrieve import retrieve as _retrieve


class RagPipeline:
    def __init__(
        self,
        artifacts_dir: Path,
        settings: RagSettingsProtocol = DEFAULT,
        encoder: GgufEncoderClient | None = None,
    ):
        self.settings = settings
        self.encoder = encoder or GgufEncoderClient(settings)
        self.idx = Indexes(artifacts_dir)

    def search(self, query: str) -> tuple[list[dict], dict]:
        """Sync: hybrid retrieval + rerank, no generation. Blocking -- call
        from the "rag" pool worker thread, never from the event loop.

        Returns (top_chunks, timings_ms). `top_chunks` carries `rerank_score`
        on every chunk so a downstream caller can apply `RAG_MIN_SCORE`
        (A-11 "honest not-found") without this module deciding that policy.
        """
        t: dict[str, float] = {}

        t0 = time.perf_counter()
        candidates = _retrieve(query, self.idx, self.encoder, self.settings, timings=t)
        t["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        top = _rerank(query, candidates, self.encoder, self.settings)
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000

        t["search_ms"] = t["retrieve_ms"] + t["rerank_ms"]
        return top, t

    async def asearch(
        self, query: str, executor: Executor | None = None
    ) -> tuple[list[dict], dict]:
        """Async wrapper for callers on the event loop. Pass the dedicated
        single-worker "rag" ThreadPoolExecutor (plan.md §2/§9, T-09 owns its
        construction) as `executor`; the default (`None`) falls back to
        asyncio's own default pool, which is fine for standalone use/tests
        but NOT the serialized single-worker pool the plan requires for
        production (concurrent GPU access from the encoders must be
        serialized, see plan.md §2 "сериализация, чтобы не рвать VRAM").
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.search, query)
