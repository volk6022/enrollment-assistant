"""Backend RAG package (T-04): hybrid retrieval (dense FAISS + BM25, RRF) +
neural rerank, both encoders served as GGUF Q8_0 by llama-server.

Ported from the legacy `rag/` package (untouched by this task), swapping the
torch/sentence-transformers encoders for `GgufEncoderClient` HTTP calls to
llama-server on `EMBEDDING_ENDPOINT` / `RERANKER_ENDPOINT`. See plan.md §6
and `docs/streaming-research-findings.md` §5/§5.1/§5.2 for why.

Public surface:

    from backend.rag import RagPipeline, RagSettings, GgufEncoderClient

`RagPipeline.search` / `.asearch` is the interface T-09's "rag" pool calls.
"""
from __future__ import annotations

from backend.rag.config import DEFAULT, RagSettings, RagSettingsProtocol
from backend.rag.gguf_encoder_client import EncoderClientError, GgufEncoderClient
from backend.rag.index import Indexes
from backend.rag.pipeline import RagPipeline

__all__ = [
    "RagPipeline",
    "RagSettings",
    "RagSettingsProtocol",
    "DEFAULT",
    "GgufEncoderClient",
    "EncoderClientError",
    "Indexes",
]
