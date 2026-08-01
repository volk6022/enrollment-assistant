"""Contract tests: LlamaClient against a LIVE llama-server (llm.md, plan.md §12).

Every property checked here was proven wrong by the "obvious" implementation
at some point during streaming-research-findings.md §4 — a mock would pass
all of these trivially and prove nothing:

  * `json_schema` + a pre-filled closed `<think></think>` on
    /v1/chat/completions returns HTTP 400 (§4.2) — only a live server
    surfaces that.
  * `chat_template_kwargs.enable_thinking=false` is a byte-for-byte no-op
    for this model's template (§4.3) — a mock would happily honor it.
  * KV-cache reuse is a server-side behaviour of `cache_prompt` — nothing
    to assert against without a real llama-server process.

Point LLM_ENDPOINT_TEST (default http://127.0.0.1:20099) at a llama-server
started with the flags in contracts/llm.md §8 before running. Tests skip
with a clear reason if no server answers /health.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest
import pytest_asyncio

from backend.llm.client import DecisionInFlightError, LlamaClient, _find_sentence_end
from backend.llm.prompts import (
    build_answer_messages,
    build_barge_in_prompt,
    build_intent_prompt,
    build_interject_prompt,
)
from backend.llm.schemas import BargeInDecision, IntentDecision, InterjectDecision

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT_TEST", "http://127.0.0.1:20099")


def _server_reachable() -> bool:
    try:
        response = httpx.get(f"{LLM_ENDPOINT}/health", timeout=2.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=(
        f"llama-server not reachable at {LLM_ENDPOINT} — start it with the flags "
        f"in specs/001-streaming-dialogue/contracts/llm.md §8 first"
    ),
)

ADMISSIONS_CONTEXT = "\n".join(
    f"[Документ {i}] Пункт {i}. Документы принимаются с 20 июня по 31 июля."
    for i in range(1, 11)
)


@pytest_asyncio.fixture
async def client():
    llama_client = LlamaClient(endpoint=LLM_ENDPOINT, timeout_s=60.0)
    yield llama_client
    await llama_client.aclose()


@pytest.mark.asyncio
async def test_stream_answer_suppresses_reasoning_and_is_fast(client: LlamaClient) -> None:
    """llm.md §2: TTFT < NFR-01's 0.3s budget, and no <think> anywhere in output.

    NFR-01 defines TTFT as time to the first *raw* token — that's
    `last_stream_timings.ttft_s`, captured before sentence assembly. The
    time to the first *yielded sentence* is a different, larger number
    whenever the first sentence itself is short (a one-sentence answer makes
    "first sentence" and "whole answer" the same latency) — that's expected
    streaming behaviour per FR-11, not a regression, so it isn't asserted
    against the NFR-01 budget here.

    NFR-01's budget (and streaming-research-findings.md §4.3's 0.075s) is a
    property of a *warm* prefix: the whole point of FR-10 is that RAG
    context + history + transcript are already prefilled via incremental
    updates while the interlocutor is still talking, so by the time
    stream_answer actually runs, only a short volatile tail needs
    evaluating. A genuinely cold call — first-ever request against this
    exact prefix — pays full prefill for the whole prompt instead (measured
    separately at ~0.5-0.65s for a RAG-context-sized prompt here, see the
    task report) and is not what NFR-01 is a claim about. This test warms
    the prefix with one throwaway call first, matching how T-09's
    orchestration will actually drive this client, then asserts the budget
    on the call that follows.
    """
    messages = build_answer_messages(
        system_prompt="Ты — ассистент приёмной комиссии. Отвечай кратко, 1-2 предложения.",
        rag_context=ADMISSIONS_CONTEXT,
        dialogue_history="",
        transcript="Какие документы нужны для поступления и до какого числа их принимают?",
    )
    async for _ in client.stream_answer(messages, max_tokens=150, temperature=0.0):
        pass  # warm-up: primes the KV cache for this exact prefix, per FR-10

    sentences: list[str] = [
        sentence
        async for sentence in client.stream_answer(messages, max_tokens=150, temperature=0.0)
    ]

    full_text = " ".join(sentences)
    assert sentences, "stream_answer produced no sentences at all"
    assert "<think>" not in full_text
    assert "</think>" not in full_text

    timings = client.last_stream_timings
    assert timings is not None
    assert timings.ttft_s is not None
    assert timings.ttft_s < 0.3, f"TTFT {timings.ttft_s:.3f}s exceeds NFR-01 budget of 0.3s"


@pytest.mark.asyncio
async def test_decide_returns_schema_valid_object_without_reasoning(client: LlamaClient) -> None:
    """llm.md §3: raw /completion + json_schema, no <think>, valid BargeInDecision."""
    prompt = build_barge_in_prompt(
        dialogue_history="",
        draft_answer="Для поступления нужны документ об образовании, заявление и медицинская справка.",
        voiced_seconds=4.1,
        total_seconds=9.7,
        interlocutor_transcript="нет, подождите, я про другое хотел спросить",
    )
    result = await client.decide(prompt, BargeInDecision.SCHEMA)

    decision = BargeInDecision.model_validate(result)
    assert decision.interrupt is True
    assert "<think>" not in result.get("reason", "")
    assert client.last_decide_timings is not None


@pytest.mark.asyncio
async def test_decide_distinguishes_backchannel_from_real_interrupt(client: LlamaClient) -> None:
    """The exact discrimination llm.md §5 and FR-13/FR-17 require: poddakivanie != interrupt."""
    prompt = build_barge_in_prompt(
        dialogue_history="",
        draft_answer="Для поступления нужны документ об образовании, заявление и медицинская справка.",
        voiced_seconds=4.1,
        total_seconds=9.7,
        interlocutor_transcript="ага... понятно... да",
    )
    result = await client.decide(prompt, BargeInDecision.SCHEMA)
    decision = BargeInDecision.model_validate(result)
    assert decision.interrupt is False


@pytest.mark.asyncio
async def test_decide_intent_and_interject_schemas_validate(client: LlamaClient) -> None:
    intent_prompt = build_intent_prompt(
        dialogue_history="",
        transcript="Спасибо большое, до свидания",
    )
    intent_result = await client.decide(intent_prompt, IntentDecision.SCHEMA)
    IntentDecision.model_validate(intent_result)  # raises if the shape drifted from §4.3

    interject_prompt = build_interject_prompt(
        dialogue_history="",
        transcript_so_far="и вот смотрите я хочу поступить у меня диплом и...",
    )
    interject_result = await client.decide(interject_prompt, InterjectDecision.SCHEMA)
    InterjectDecision.model_validate(interject_result)


@pytest.mark.asyncio
async def test_decide_reuses_kv_cache_on_repeat_prefix(client: LlamaClient) -> None:
    """llm.md §6 / streaming-research-findings.md §4.1: cache_prompt=true must
    make a repeat call against the same prefix re-evaluate an order of
    magnitude fewer tokens than the cold call.
    """
    prompt = build_intent_prompt(
        dialogue_history="Собеседник: Здравствуйте.\nАгент: Здравствуйте, чем могу помочь?",
        transcript="Какие документы нужны для поступления на очную форму?",
    )
    first = await client.decide(prompt, IntentDecision.SCHEMA)
    IntentDecision.model_validate(first)
    first_timings = client.last_decide_timings
    assert first_timings is not None

    second = await client.decide(prompt, IntentDecision.SCHEMA)
    IntentDecision.model_validate(second)
    second_timings = client.last_decide_timings
    assert second_timings is not None

    assert second_timings.cached_tokens > 0
    assert second_timings.prompt_tokens * 10 <= max(first_timings.prompt_tokens, 1), (
        f"expected an order-of-magnitude drop in prompt_tokens on cache reuse, "
        f"got {first_timings.prompt_tokens} -> {second_timings.prompt_tokens}"
    )


@pytest.mark.asyncio
async def test_decide_refuses_concurrent_second_call(client: LlamaClient) -> None:
    """llm.md §7: a contended decide() is refused immediately, never queued."""
    prompt = build_intent_prompt(dialogue_history="", transcript="Здравствуйте")

    first_task = asyncio.create_task(
        client.decide(prompt, IntentDecision.SCHEMA, max_tokens=200)
    )
    await asyncio.sleep(0)  # let the first call flip _decide_in_flight before we try the second

    with pytest.raises(DecisionInFlightError):
        await client.decide(prompt, IntentDecision.SCHEMA)

    await first_task  # don't leak the task


def test_find_sentence_end_splits_on_terminal_punctuation() -> None:
    for text in ("Привет. ", "Всё хорошо! ", "Правда? ", "Понятно… "):
        end = _find_sentence_end(text)
        assert end is not None, f"expected a boundary in {text!r}"
        assert text[:end] == text.strip()


def test_find_sentence_end_does_not_split_inside_a_date() -> None:
    assert _find_sentence_end("Документы принимаются до 31.07 включительно") is None


def test_find_sentence_end_does_not_split_after_abbreviation() -> None:
    assert _find_sentence_end("см. приложение") is None


def test_find_sentence_end_waits_for_lookahead_on_trailing_dot() -> None:
    assert _find_sentence_end("Документы принимаются до 31.") is None
