"""Contract tests: `GgufEncoderClient.rerank` against a LIVE reranker
llama-server (RAG_MIN_SCORE bug investigation, contracts/llm.md §8).

Two properties a unit test with a fake HTTP session cannot prove, because
they depend on the real model and the real index:

  * The live server's `/v1/rerank` genuinely returns raw logits (not
    already-squashed probabilities) for bge-reranker-v2-m3 GGUF -- if a
    future llama.cpp/llama-server version changes that, sigmoid-ing again
    client-side would silently double-squash every score towards 0.5 and
    quietly break `RAG_MIN_SCORE` a second time, in the opposite direction.
    A relevant top-1 result scoring far below 0.5 is exactly what the
    original bug report looked like.
  * The `-b`/`-ub` flags in `scripts/serve-models.{ps1,sh}` actually cover
    the real worst case: a `RAG_FUSED_TOP`-sized candidate batch that
    includes the most token-dense chunk in the real index
    (`backend/rag/artifacts/chunks.jsonl`, a numeric table that tokenizes to
    687 tokens despite being only 1200 characters) paired with a long query.
    Before the fix (default -ub 512), this was an HTTP 500 ("input (826
    tokens) is too large to process") that `backend/ws/session.py`'s broad
    `except Exception` turned into a silent fake-failure answer.

Point RERANKER_ENDPOINT_TEST (default http://127.0.0.1:20101) at a
llama-server started with the flags in contracts/llm.md §8 (or run
`scripts/serve-models.ps1`/`.sh`) before running. Tests skip with a clear
reason if no server answers /health -- same pattern as
tests/contract/test_llm_client.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from backend.rag.config import DEFAULT
from backend.rag.gguf_encoder_client import GgufEncoderClient

RERANKER_ENDPOINT = os.environ.get("RERANKER_ENDPOINT_TEST", "http://127.0.0.1:20101")

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = REPO_ROOT / "backend" / "rag" / "artifacts" / "chunks.jsonl"


def _server_reachable() -> bool:
    try:
        response = httpx.get(f"{RERANKER_ENDPOINT}/health", timeout=2.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=(
        f"reranker llama-server not reachable at {RERANKER_ENDPOINT} -- start it with "
        f"scripts/serve-models.ps1 / .sh (flags per "
        f"specs/001-streaming-dialogue/contracts/llm.md §8) first"
    ),
)


def _client() -> GgufEncoderClient:
    return GgufEncoderClient(replace(DEFAULT, reranker_endpoint=RERANKER_ENDPOINT))


def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        pytest.skip(f"no built RAG index at {CHUNKS_PATH} -- see README.md")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_rerank_scores_land_in_unit_interval_on_real_chunks() -> None:
    """The RAG_MIN_SCORE bug's core symptom: raw logits routinely sit well
    outside [0,1] (e.g. -2.8). Whatever the live server returns after this
    client's sigmoid must be in-range for EVERY doc, on real chunk text --
    not just the short synthetic strings a unit test would use.
    """
    chunks = _load_chunks()
    docs = [c["text"] for c in chunks[:20]]
    client = _client()

    scores = client.rerank("Какие документы нужны для поступления на бюджет?", docs)

    assert len(scores) == len(docs)
    for score in scores:
        assert 0.0 <= score <= 1.0, (
            f"score {score} outside [0,1] -- looks like the client is returning raw "
            f"logits again (RAG_MIN_SCORE regression)"
        )


def test_rerank_does_not_500_on_worst_case_candidate_batch() -> None:
    """Regression for the second bug: llama-server's default -ub (512) 500s
    once a single (query, doc) pair exceeds it. This reconstructs the
    measured worst case from the real index -- the most token-dense chunk
    (a numeric table that tokenizes far denser than its character count
    suggests) combined with a long, rambling query -- inside a full
    `RAG_FUSED_TOP`-sized (50) candidate batch, matching what
    `backend/rag/rerank.py` actually sends in production.
    """
    chunks = _load_chunks()
    # The most token-dense chunk found in this index during the investigation
    # (source: "Постановление Правительства РФ от 04_07_2013 N 565", a
    # height/weight table) -- found by tokenizing every chunk against the live
    # reranker and taking the max; pinned by content prefix (not list index)
    # so this test survives the index being rebuilt in a different order.
    pathological = next(
        (c["text"] for c in chunks if c["text"].startswith("| 71,6 - 82,9 | 83,0 - 94,6")),
        max(chunks, key=lambda c: len(c["text"]))["text"],
    )

    long_transcript = (
        "Здравствуйте да вот я хотел бы узнать в общем смотрите у меня такая ситуация "
        "я заканчиваю школу в этом году и мне нужно поступать и вот я не очень понимаю "
        "какие документы нужны для поступления на бюджетное отделение и до какого числа "
        "их нужно подавать и можно ли подавать документы через госуслуги или обязательно "
        "нужно приезжать лично в приемную комиссию и еще у меня вопрос про общежитие "
        "будет ли оно предоставлено на первом курсе и как оплачивается и что делать если "
        "я иногородний и еще хотел спросить про льготы у меня мама одна воспитывает и я "
        "не знаю положены ли какие-то льготы при поступлении в этом случае"
    )

    other_docs = [c["text"] for c in chunks[:49]]
    candidates = other_docs + [pathological]  # RAG_FUSED_TOP=50 shape
    client = _client()

    scores = client.rerank(long_transcript, candidates)

    assert len(scores) == len(candidates)
    for score in scores:
        assert 0.0 <= score <= 1.0
