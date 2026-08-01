"""HTTP client for llama-server: exactly two call paths, no third.

Reasoning suppression and structured output are mutually incompatible on the
chat endpoint (contracts/llm.md §2-3; verified — `json_schema` plus a
pre-filled closed `<think></think>` assistant turn returns HTTP 400
"Unexpected empty grammar stack after accepting piece: <think>"). So free
text and decisions never share a code path:

  stream_answer  POST /v1/chat/completions, stream=True. `<think>` is
                 suppressed by appending a closed assistant turn
                 (`<think>\\n\\n</think>\\n\\n`) — the only method that works
                 for this model; `chat_template_kwargs.enable_thinking=false`
                 is a documented no-op for this template (streaming-research-
                 findings.md §4.3).

  decide         POST /completion (raw, no chat template applied). The
                 caller renders ChatML markup by hand (prompts.py) so the
                 prompt ends on an open `<|im_start|>assistant\\n` — the
                 json_schema grammar then constrains generation from the
                 very first token, and the model is physically unable to
                 start `<think>`.

Anything else needed from llama-server belongs in a different module.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypedDict

import httpx


class Message(TypedDict):
    """One chat turn — the role/content shape llama-server's chat endpoint expects."""

    role: str
    content: str


_THINK_PREFILL: Message = {"role": "assistant", "content": "<think>\n\n</think>\n\n"}

_SENTENCE_BOUNDARY_CHARS = ("!", "?", "…")

# Heuristic, not exhaustive (contracts/llm.md §2: "не внутри числа и не после
# общепринятого сокращения"). Multi-dot compounds like "т.е." are not
# special-cased — worst case a sentence splits one word early, which only
# costs TTS one extra short utterance, never a wrong transcript.
_ABBREVIATIONS = frozenset(
    {
        "др", "пр", "им", "рис", "см", "стр", "гл", "проф",
        "тыс", "млн", "млрд", "руб", "просп", "ул", "г", "гг",
    }
)


class LlamaClientError(RuntimeError):
    """Base class for errors raised by LlamaClient."""


class LlamaServerError(LlamaClientError):
    """llama-server answered with a non-2xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"llama-server HTTP {status_code}: {body[:500]}")


class DecisionInFlightError(LlamaClientError):
    """A second concurrent decide() call was attempted (llm.md §7).

    `decide()` is guarded by a semaphore of 1, and a contended call is
    refused rather than queued: a decision that lands after its window
    closed (the barge-in already resolved itself, the 20s timer already
    fired) is worse than no decision — queuing would let a stale one report
    anyway.
    """

    def __init__(self) -> None:
        super().__init__("decide() is already in flight; the second concurrent call is refused")


@dataclass(slots=True, frozen=True)
class CallTimings:
    """Latency/cache telemetry from one llama-server call (plan.md §12).

    `prompt_tokens` is the delta actually re-evaluated this call (llama-
    server's `timings.prompt_n`); `cached_tokens` is how much of the prefix
    was reused (`timings.cache_n`). A repeat call against an unchanged
    prefix should show `prompt_tokens` collapse and `cached_tokens` rise.
    """

    wall_s: float
    prompt_tokens: int
    cached_tokens: int
    predicted_tokens: int
    ttft_s: float | None = None
    """Time to first raw token, before sentence assembly (NFR-01). None for
    `decide()`, which does not stream — its whole call is the "first token"."""


def _preceding_word(buffer: str, dot_index: int) -> str:
    start = dot_index
    while start > 0 and buffer[start - 1].isalpha():
        start -= 1
    return buffer[start:dot_index]


def _find_sentence_end(buffer: str) -> int | None:
    """First index (exclusive) in `buffer` where a complete sentence ends.

    `!`, `?`, `…` and newline always end a sentence. `.` does not when it
    sits between two digits (dates, decimals: "31.07") or right after a
    common abbreviation. A trailing `.` with nothing after it yet is left
    unresolved (returns None) rather than guessed — the caller waits for the
    next token, or flushes the whole remainder unconditionally once the
    stream itself has ended.
    """
    for i, ch in enumerate(buffer):
        if ch == "\n" or ch in _SENTENCE_BOUNDARY_CHARS:
            return i + 1
        if ch != ".":
            continue
        if i + 1 >= len(buffer):
            return None
        prev_digit = i > 0 and buffer[i - 1].isdigit()
        next_digit = buffer[i + 1].isdigit()
        if prev_digit and next_digit:
            continue
        if _preceding_word(buffer, i).lower() in _ABBREVIATIONS:
            continue
        return i + 1
    return None


class LlamaClient:
    """Two call paths to llama-server. No third path — see module docstring."""

    def __init__(self, *, endpoint: str, timeout_s: float = 30.0) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))
        self._decide_in_flight = False
        self.last_stream_timings: CallTimings | None = None
        self.last_decide_timings: CallTimings | None = None

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> LlamaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def stream_answer(
        self,
        messages: Iterable[Message],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Free-form answer text, one complete sentence per yield (llm.md §2).

        Reasoning is suppressed by appending a closed think-block assistant
        turn to `messages` — callers pass only system/user turns; the
        prefill is protocol mechanics for this endpoint, not prompt content,
        so it lives here rather than in prompts.py.

        Cancellation is closing the SSE connection (FR-12): stop iterating
        (`break`) or call `.aclose()` on this generator and httpx tears the
        connection down via the `async with` block below. Nothing here
        commits to any store, so there is nothing to roll back — that's the
        caller's responsibility once it decides the partial text is usable.
        """
        body: dict[str, Any] = {
            "messages": [*messages, _THINK_PREFILL],
            "stream": True,
            "cache_prompt": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        buffer = ""
        t0 = time.perf_counter()
        ttft_s: float | None = None
        final_timings: dict[str, Any] = {}
        async with self._http.stream(
            "POST", f"{self._endpoint}/v1/chat/completions", json=body
        ) as response:
            if response.status_code >= 400:
                error_body = (await response.aread()).decode("utf-8", errors="replace")
                raise LlamaServerError(response.status_code, error_body)
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choice = (chunk.get("choices") or [{}])[0]
                timings = chunk.get("timings")
                if timings is not None:
                    final_timings = timings
                token = (choice.get("delta") or {}).get("content") or ""
                if not token:
                    continue
                if ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                buffer += token
                end = _find_sentence_end(buffer)
                while end is not None:
                    sentence, buffer = buffer[:end].strip(), buffer[end:]
                    if sentence:
                        yield sentence
                    end = _find_sentence_end(buffer)
        self.last_stream_timings = CallTimings(
            wall_s=time.perf_counter() - t0,
            prompt_tokens=final_timings.get("prompt_n", 0),
            cached_tokens=final_timings.get("cache_n", 0),
            predicted_tokens=final_timings.get("predicted_n", 0),
            ttft_s=ttft_s,
        )
        tail = buffer.strip()
        if tail:
            yield tail

    async def decide(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """One structured decision via raw /completion (llm.md §3).

        `temperature` is fixed at 0: decisions must be reproducible — the
        same situation has to reach the same verdict, or A-06/A-07 can't be
        debugged or accepted. Refuses a second concurrent call instead of
        queuing it — see DecisionInFlightError.
        """
        if self._decide_in_flight:
            raise DecisionInFlightError()
        self._decide_in_flight = True
        t0 = time.perf_counter()
        try:
            response = await self._http.post(
                f"{self._endpoint}/completion",
                json={
                    "prompt": prompt,
                    "json_schema": schema,
                    "cache_prompt": True,
                    "temperature": 0.0,
                    "n_predict": max_tokens,
                },
            )
            if response.status_code >= 400:
                raise LlamaServerError(response.status_code, response.text)
            body = response.json()
            timings = body.get("timings") or {}
            self.last_decide_timings = CallTimings(
                wall_s=time.perf_counter() - t0,
                prompt_tokens=timings.get("prompt_n", 0),
                cached_tokens=timings.get("cache_n", 0),
                predicted_tokens=timings.get("predicted_n", 0),
            )
            content: str = body["content"]
            return json.loads(content)
        finally:
            self._decide_in_flight = False
