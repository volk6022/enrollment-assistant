"""Configuration surface for the backend RAG package.

FR-32 says `.env` is read in exactly one place: `backend/config.py` (T-01,
wave-1, a parallel task). That module doesn't exist yet at the time this
package was written. Rather than call `os.getenv` here too (which would
violate FR-32 the moment `backend/config.py` lands and becomes the *actual*
single source), this module does no environment reading at all:

- `RagSettingsProtocol` is the shape the pipeline needs.
- `RagSettings` is a plain, dependency-free dataclass whose defaults mirror
  `.env.example` field-for-field (checked against the file on 2026-08-01).

Wiring note for whoever lands `backend/config.py` (T-01) or the "rag" pool
(T-09): either make the pydantic-settings `Settings` class satisfy
`RagSettingsProtocol` directly and pass it straight into `RagPipeline`, or
construct a `RagSettings` from it, e.g.:

    rag_settings = RagSettings(
        embedding_endpoint=settings.EMBEDDING_ENDPOINT,
        reranker_endpoint=settings.RERANKER_ENDPOINT,
        rag_top_k=settings.RAG_TOP_K,
        ...
    )

Either way, nothing in `backend/rag/` should ever call `os.environ` /
`os.getenv` directly -- that's the FR-32 invariant this module is protecting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class RagSettingsProtocol(Protocol):
    """Structural type: anything with these attributes can drive RagPipeline."""

    embedding_endpoint: str
    reranker_endpoint: str

    rag_top_k: int
    rag_fused_top: int
    rag_dense_top: int
    rag_bm25_top: int
    rag_rrf_k: int
    rag_min_score: float
    rag_max_length: int
    rag_batch_size: int

    faiss_index_path: str
    kb_source_path: str


@dataclass(frozen=True)
class RagSettings:
    """Standalone default settings -- values copied verbatim from
    `.env.example` (RAG_* block + EMBEDDING_ENDPOINT/RERANKER_ENDPOINT), with
    `host.docker.internal` swapped for `127.0.0.1` since this dataclass is
    also what's used for direct (non-container) runs and tests. Ports 20100
    (embedding) / 20101 (reranker) match `contracts/llm.md` §8 and
    `scripts/serve-models.*` exactly.
    """

    embedding_endpoint: str = "http://127.0.0.1:20100"
    reranker_endpoint: str = "http://127.0.0.1:20101"

    # RAG_TOP_K=10 -- sweep of 1200 tests in experiments-rag-params/GRID_RESULTS.md:
    # significant gain over k=5 (paired t=2.12, delta=0.033+-0.016); k=15/20 don't
    # earn back their +1.8s/+3.6s prefill cost. Was 5 in the legacy rag/config.py.
    rag_top_k: int = 10
    rag_fused_top: int = 50        # RAG_FUSED_TOP -- candidates handed to the reranker
    rag_dense_top: int = 20        # RAG_DENSE_TOP
    rag_bm25_top: int = 20         # RAG_BM25_TOP
    rag_rrf_k: int = 60            # RAG_RRF_K
    rag_min_score: float = 0.3     # RAG_MIN_SCORE -- below this reranker score, treat
                                    # as "not in the knowledge base" (A-11); the actual
                                    # honest-refusal decision is downstream (dialogue/
                                    # generation), this package only carries the score.
    rag_max_length: int = 256      # RAG_MAX_LENGTH -- informational; enforced server-side
                                    # by the reranker's llama-server instance, not by us.
    rag_batch_size: int = 32       # RAG_BATCH_SIZE -- texts per /v1/embeddings request.

    faiss_index_path: str = "/data/faiss_index"    # FAISS_INDEX_PATH (container path)
    kb_source_path: str = "/data/knowledge_base"   # KB_SOURCE_PATH (container path)

    # Not in .env.example (no LLM_TIMEOUT_S equivalent existed for the encoders
    # before this task) -- generous but bounded so a wedged llama-server fails
    # loudly instead of hanging the "rag" pool worker forever.
    embed_timeout_s: float = 60.0
    rerank_timeout_s: float = 60.0


DEFAULT = RagSettings()
