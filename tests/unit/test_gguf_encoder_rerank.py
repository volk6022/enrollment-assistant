"""Regression coverage for the RAG_MIN_SCORE / raw-logit bug.

`llama-server`'s `/v1/rerank` for a GGUF cross-encoder (bge-reranker-v2-m3
Q8_0) returns a raw pre-activation logit, not a 0-1 probability -- the client
itself documents this (`backend/rag/gguf_encoder_client.py`). `RAG_MIN_SCORE`
(`backend/config.py`'s `rag_min_score`, `Field(ge=0.0, le=1.0)`) was
calibrated against the legacy `sentence_transformers.CrossEncoder.predict`
path, which applies a sigmoid automatically for a single-label regression
head. Without squashing the GGUF client's output the same way, real queries
against the live index scored things like [-0.354, -0.63, -0.838, -2.603,
-2.781] for a clearly relevant top-1 chunk -- all below any sane [0,1]
threshold -- and the honest-KB-match path (`answer_question`) silently lost
to `no_kb_match` (priority 90 in `Formulating`) on most questions the KB
actually answers.

This test is deliberately a fast, dependency-free unit test (fakes the
`requests.Session.post` call, no llama-server needed) -- it pins the
*transform*, not the live model's actual scores. The live-server, real-index
check (does the pipeline actually clear RAG_MIN_SCORE=0.3 on real questions)
is a separate, manual verification against `RagPipeline` per the delivery
report; there's no fast/offline way to reproduce "is this reranker.gguf
actually well-calibrated" without the real weights and index.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from backend.rag.config import DEFAULT
from backend.rag.gguf_encoder_client import GgufEncoderClient


class _FakeResponse:
    def __init__(self, status_code: int, body: dict, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text or str(body)

    def json(self) -> dict:
        return self._body


class _FakeSession:
    """Stands in for `requests.Session`: records the request, replays a
    canned `/v1/rerank`-shaped response built from raw logits.

    `logits_by_index` is deliberately a sparse mapping, not a plain list --
    that lets tests simulate a server response that omits rows for some
    doc indices (e.g. `top_n` trimming, or a partial-failure response),
    which is exactly the case the missing-row fallback has to handle.
    """

    def __init__(self, logits_by_index: dict[int, float]):
        self._logits_by_index = logits_by_index
        self.trust_env = False
        self.last_request: dict | None = None

    def post(self, url: str, json: dict, timeout: float) -> _FakeResponse:
        self.last_request = {"url": url, "json": json, "timeout": timeout}
        results = [
            {"index": i, "relevance_score": logit}
            for i, logit in self._logits_by_index.items()
        ]
        return _FakeResponse(200, {"results": results})


def _client_with_fake_logits(logits: list[float]) -> GgufEncoderClient:
    return _client_with_sparse_logits({i: v for i, v in enumerate(logits)}, n_docs=len(logits))


def _client_with_sparse_logits(logits_by_index: dict[int, float], n_docs: int) -> GgufEncoderClient:
    settings = replace(DEFAULT, reranker_endpoint="http://fake-reranker")
    client = GgufEncoderClient(settings)
    client.sess = _FakeSession(logits_by_index)
    return client


def test_rerank_squashes_raw_logits_into_unit_interval() -> None:
    """The exact numbers from the bug report's first failing query
    ("Какие документы нужны для поступления на бюджет?"): a relevant top-1
    chunk scored -0.354 as a raw logit -- comfortably below RAG_MIN_SCORE=0.3
    on ANY sane reading, but sigmoid(-0.354) ~= 0.412, which clears it.
    """
    raw_logits = [-0.354, -0.63, -0.838, -2.603, -2.781]
    docs = [f"chunk {i}" for i in range(len(raw_logits))]

    client = _client_with_fake_logits(raw_logits)
    scores = client.rerank("Какие документы нужны для поступления на бюджет?", docs)

    assert len(scores) == len(raw_logits)
    for score in scores:
        assert 0.0 <= score <= 1.0, f"score {score} escaped [0,1] -- looks like raw logits again"

    expected = [1.0 / (1.0 + math.exp(-x)) for x in raw_logits]
    for got, want in zip(scores, expected):
        assert got == pytest.approx(want, abs=1e-9)

    # RAG_MIN_SCORE=0.3 must now actually admit the top-1 chunk, matching the
    # live-stack finding that this query's top-1 result is relevant.
    assert scores[0] >= DEFAULT.rag_min_score


def test_rerank_preserves_relative_order() -> None:
    """Sigmoid is strictly monotonic -- squashing must never reshuffle the
    ranking the pipeline's top-k selection (`backend/rag/rerank.py`) relies
    on, only rescale it.
    """
    raw_logits = [3.141, -0.165, 2.126, -2.781, 0.0, -0.838]
    docs = [f"chunk {i}" for i in range(len(raw_logits))]

    client = _client_with_fake_logits(raw_logits)
    scores = client.rerank("q", docs)

    rank_by_logit = sorted(range(len(raw_logits)), key=lambda i: raw_logits[i], reverse=True)
    rank_by_score = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    assert rank_by_logit == rank_by_score


def test_rerank_handles_extreme_logits_without_overflow() -> None:
    """A very negative logit must not raise `OverflowError` in `math.exp`
    (naive `1/(1+exp(-x))` overflows around x <~ -710) and must still land
    close to 0/1 at the extremes.
    """
    raw_logits = [-1000.0, 1000.0, 0.0]
    docs = [f"chunk {i}" for i in range(len(raw_logits))]

    client = _client_with_fake_logits(raw_logits)
    scores = client.rerank("q", docs)

    assert scores[0] == pytest.approx(0.0, abs=1e-9)
    assert scores[1] == pytest.approx(1.0, abs=1e-9)
    assert scores[2] == pytest.approx(0.5, abs=1e-9)


def test_rerank_empty_docs_returns_empty_list() -> None:
    client = _client_with_fake_logits([])
    assert client.rerank("q", []) == []


def test_rerank_missing_row_scores_below_threshold_not_above_it() -> None:
    """Regression for a second bug introduced alongside the sigmoid fix: the
    per-doc score array used to be pre-filled with `0.0` for indices the
    server never returned a row for. `0.0` reads as a neutral *raw logit*,
    but after sigmoid, `0.0` becomes `0.5` -- comfortably ABOVE
    RAG_MIN_SCORE=0.3 -- so a doc the reranker never scored at all would
    silently be treated as more relevant than a real, poorly-matching logit
    like -0.838 (sigmoid ~= 0.30) and could even clear the threshold on its
    own. A missing row means "no evidence", which must score at the bottom
    (~0.0 after sigmoid), never in the middle.
    """
    docs = [f"chunk {i}" for i in range(3)]
    # Server only returns rows for index 0 and 2 -- index 1's row is missing,
    # simulating a partial `/v1/rerank` response.
    client = _client_with_sparse_logits({0: -0.838, 2: 1.5}, n_docs=len(docs))

    scores = client.rerank("q", docs)

    assert len(scores) == 3
    assert scores[1] == pytest.approx(0.0, abs=1e-9)
    assert scores[1] < DEFAULT.rag_min_score
    # and it must rank last, not somewhere in the middle of real scores
    assert scores[1] < scores[0] < scores[2]
