"""Can Qwopus3.5-4B's reasoning preamble be suppressed? Streaming TTS depends on it.

llama_prefill_probe.py exp A measured TTFT 0.30 s / "first sentence" 0.39 s — but the
stream opened with `<think>`, so that first sentence was the model's reasoning, not
speakable answer text. With thinking on, the real time-to-first-speakable-token is
however long the whole reasoning block takes, which destroys the latency budget.

So: try each suppression route and report, for each, whether `<think>` appears and how
long until the first token of ACTUAL answer arrives.

  raw_completion_plain      /completion, no template (what exp A did)
  chat_default              /v1/chat/completions, template as-is
  chat_enable_thinking_off  /v1/chat/completions + chat_template_kwargs.enable_thinking=false
  chat_nothink_tag          /v1/chat/completions with a /no_think marker in the system prompt
  chat_prefill_closed_think assistant turn pre-filled with an already-closed <think></think>

Run with llama-server up on :20055.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:20055"
OUT = Path(__file__).resolve().parent / "runs" / "llama_nothink.json"

SYSTEM = ("Ты — помощница приёмной комиссии института. Отвечай кратко, 1–3 предложения, "
          "только по существу заданного вопроса.")
CONTEXT = "\n".join(
    f"[Документ {i}] Пункт {i}. Для поступления кандидат представляет заявление, "
    f"документ об образовании, медицинское заключение и результаты вступительных "
    f"испытаний. Документы принимаются с 20 июня по 31 июля."
    for i in range(1, 21)
)
QUESTION = "Какие документы нужны для поступления и до какого числа их принимают?"
USER = f"Контекст:\n{CONTEXT}\n\nВопрос: {QUESTION}"


async def stream_chat(client: httpx.AsyncClient, body: dict) -> dict:
    t0 = time.perf_counter()
    first_tok = None
    first_speakable = None
    acc = ""
    async with client.stream("POST", f"{BASE}/v1/chat/completions",
                            json={**body, "stream": True}, timeout=240.0) as r:
        if r.status_code >= 400:
            return {"error": f"HTTP {r.status_code}: {(await r.aread()).decode()[:300]}"}
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            js = json.loads(payload)
            delta = (js.get("choices") or [{}])[0].get("delta") or {}
            tok = delta.get("content") or ""
            # some builds surface reasoning separately
            if delta.get("reasoning_content") and first_tok is None:
                first_tok = time.perf_counter() - t0
            if not tok:
                continue
            if first_tok is None:
                first_tok = time.perf_counter() - t0
            acc += tok
            if first_speakable is None:
                # speakable = outside any <think> block
                after = acc.split("</think>")[-1] if "</think>" in acc else (
                    "" if "<think>" in acc else acc)
                if after.strip():
                    first_speakable = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {
        "ttft_s": round(first_tok, 3) if first_tok else None,
        "first_speakable_s": round(first_speakable, 3) if first_speakable else None,
        "total_s": round(total, 3),
        "has_think_tag": "<think>" in acc,
        "chars": len(acc),
        "text": acc.strip()[:400],
    }


async def stream_raw(client: httpx.AsyncClient, prompt: str) -> dict:
    t0 = time.perf_counter()
    first_tok = None
    acc = ""
    async with client.stream("POST", f"{BASE}/completion", json={
        "prompt": prompt, "n_predict": 200, "stream": True,
        "cache_prompt": True, "temperature": 0.0,
    }, timeout=240.0) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            js = json.loads(line[6:])
            tok = js.get("content") or ""
            if tok and first_tok is None:
                first_tok = time.perf_counter() - t0
            acc += tok
            if js.get("stop"):
                break
    return {
        "ttft_s": round(first_tok, 3) if first_tok else None,
        "first_speakable_s": None,
        "total_s": round(time.perf_counter() - t0, 3),
        "has_think_tag": "<think>" in acc,
        "chars": len(acc),
        "text": acc.strip()[:400],
    }


async def main() -> None:
    base_msgs = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": USER}]
    common = {"temperature": 0.0, "max_tokens": 220}

    cases: dict = {}
    async with httpx.AsyncClient() as client:
        cases["raw_completion_plain"] = await stream_raw(
            client, f"{SYSTEM}\n\n{USER}\nОтвет:")

        cases["chat_default"] = await stream_chat(
            client, {"messages": base_msgs, **common})

        cases["chat_enable_thinking_off"] = await stream_chat(
            client, {"messages": base_msgs,
                     "chat_template_kwargs": {"enable_thinking": False}, **common})

        cases["chat_nothink_tag"] = await stream_chat(
            client, {"messages": [{"role": "system", "content": SYSTEM + " /no_think"},
                                  {"role": "user", "content": USER}], **common})

        # pre-fill the assistant turn with an already-closed think block so the model
        # has nothing left to reason into and must start answering immediately
        cases["chat_prefill_closed_think"] = await stream_chat(
            client, {"messages": base_msgs + [
                {"role": "assistant", "content": "<think>\n\n</think>\n\n"}],
                     **common})

    for name, r in cases.items():
        print(f"=== {name} ===")
        if "error" in r:
            print(f"  ERROR {r['error']}")
        else:
            print(f"  ttft={r['ttft_s']}s  first_speakable={r['first_speakable_s']}s  "
                  f"total={r['total_s']}s  <think>={r['has_think_tag']}  chars={r['chars']}")
            print(f"  text: {r['text'][:220]!r}")
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
