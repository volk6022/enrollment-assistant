"""Sync HTTP client for the two GGUF encoders (bge-m3 embedder, bge-reranker-v2-m3
reranker) served by llama-server.

Ported from `experiments-rag-params/gguf_encoder_client.py`, which is the
already-validated reference: same `/v1/embeddings` and `/v1/rerank` request
shapes, same L2-normalization of embeddings for FAISS inner-product = cosine.

Two things dropped versus the experiment version, both process-lifecycle, not
protocol:

1. No `GgufServer` subprocess management. In production the two llama-server
   instances are long-lived, started once by `scripts/serve-models.{ps1,sh}`
   (T-11) on `EMBEDDING_ENDPOINT` / `RERANKER_ENDPOINT` from `.env`, before the
   backend process starts. This client only ever speaks HTTP to whatever
   `RagSettings` points at -- it never starts, stops, or owns a server.
2. No `--no-cache-prompt -cram 0 -ctxcp 0 -cpent -1` flag handling here --
   those are server *launch* flags (contracts/llm.md §8), irrelevant to a pure
   HTTP client. They matter for T-04's acceptance test (5856 chunks without an
   HTTP 500) but live in the process that starts the server, not in this file.

Runs inside the "rag" thread pool (plan.md §2, §9): exactly one worker,
reached via `run_in_executor` from the event loop. Blocking `requests` calls
are intentional here, mirroring the stt/tts workers' synchronous design --
this whole module has no `async def` in it on purpose.
"""
from __future__ import annotations

import math

import numpy as np
import requests

from backend.rag.config import RagSettingsProtocol


class EncoderClientError(RuntimeError):
    """Raised when an encoder endpoint returns an HTTP error or a malformed body."""


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # never route 127.0.0.1/host.docker.internal through a SOCKS proxy
    return s


class GgufEncoderClient:
    """Talks to the embedding and reranking llama-server endpoints."""

    def __init__(self, settings: RagSettingsProtocol):
        self.settings = settings
        self.sess = _session()

    # --- embed ---------------------------------------------------------

    def embed(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed texts via POST /v1/embeddings, batched at `batch_size` (default:
        `settings.rag_batch_size`). Returns L2-normalized float32 vectors so a
        FAISS `IndexFlatIP` computes cosine similarity, matching the legacy
        `rag/embed.py`'s `normalize_embeddings=True`.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        bs = batch_size or self.settings.rag_batch_size
        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            r = self.sess.post(
                f"{self.settings.embedding_endpoint}/v1/embeddings",
                json={"input": batch, "model": "x"},
                timeout=self.settings.embed_timeout_s,
            )
            if r.status_code >= 400:
                # Surface which text index failed and whether the process is even
                # alive -- a bare 500 here is exactly the failure mode findings.md
                # §5 documents (server dies silently around chunk ~784/5856 without
                # the cache-disabling flags).
                raise EncoderClientError(
                    f"embed failed at text {i}/{len(texts)} "
                    f"(HTTP {r.status_code}): {r.text[:300]}"
                )
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        vecs = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    # --- rerank ----------------------------------------------------------

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Score (query, doc) pairs via POST /v1/rerank. Returns relevance
        scores in `docs` order, squashed through a sigmoid into [0, 1].

        llama-server's `/v1/rerank` for a GGUF cross-encoder returns the raw
        pre-activation logit, not a probability (unlike the legacy
        `sentence_transformers.CrossEncoder.predict`, which applies a sigmoid
        automatically for a single-label regression head such as
        bge-reranker-v2-m3 -- see `rag/rerank.py`, the un-ported legacy path).
        `RAG_MIN_SCORE` (`backend/config.py`'s `rag_min_score`, `Field(ge=0.0,
        le=1.0)`) was calibrated against that [0,1] sigmoid scale and never
        re-tuned for raw logits, so leaving the score unsquashed silently
        turned the threshold into "reject almost everything" (raw logits
        routinely sit well below 0, e.g. -2.8, while the field only accepts
        values in [0,1]) -- see the RAG_MIN_SCORE bug investigation. Applying
        sigmoid here, once, at the client boundary, restores the calibrated
        scale for every caller (`backend/rag/rerank.py`'s top-k selection,
        `backend/ws/session.py`'s `rag_max_score`/`RAG_MIN_SCORE` gate) without
        having to touch the threshold or any downstream comparison -- sigmoid
        is strictly monotonic, so relative ORDER (all the pipeline actually
        needs per plan.md §6) is unchanged, only the numbers become
        comparable to a [0,1] threshold again.
        """
        if not docs:
            return []
        r = self.sess.post(
            f"{self.settings.reranker_endpoint}/v1/rerank",
            json={"query": query, "documents": docs, "model": "x", "top_n": len(docs)},
            timeout=self.settings.rerank_timeout_s,
        )
        if r.status_code >= 400:
            raise EncoderClientError(f"rerank failed (HTTP {r.status_code}): {r.text[:300]}")
        payload = r.json()
        rows = payload.get("results") or payload.get("data") or []
        # -inf, not 0.0: a missing row means the server never scored that doc
        # at all, and 0.0 is a real, fairly confident *logit* (sigmoid(0.0) ==
        # 0.5, comfortably above RAG_MIN_SCORE=0.3) -- initializing with it
        # used to make an unscored doc silently outrank/clear the threshold
        # instead of being the least relevant thing in the batch. sigmoid(-inf)
        # == 0.0 below, which is the actual "no evidence" score.
        logits = [float("-inf")] * len(docs)
        for row in rows:
            logits[row["index"]] = float(row["relevance_score"])
        # 1/(1+e^-x) via expit's stable form -- avoids overflow in exp() for
        # very negative logits (float overflow on exp(710+) raises, expit-style
        # branching does not); exp(-inf) == 0.0 in Python, no special-casing
        # needed for the missing-row sentinel above.
        return [
            1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
            for x in logits
        ]
