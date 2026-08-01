"""llama-server experiments for the streaming rewrite.

Four questions the streaming design depends on:

  A. TTFT / throughput      How fast is the first token out, and at what rate do the rest
                            follow? Sets the floor on "time until TTS can start speaking".

  B. Incremental prefill    The plan is to keep feeding the LLM the transcript AS the user
                            speaks, so the prompt is already prefilled when they stop. That
                            only pays off if llama-server reuses the KV cache for the
                            unchanged prefix. Measure prefill cost of a prompt that GROWS
                            by appending, with cache_prompt on.

  C. Prompt layout          The cache only survives if everything volatile sits at the END.
                            Compare append-at-end (transcript last) vs append-in-middle
                            (instructions after the transcript). If layout B re-prefills
                            everything, the prompt template must be designed around it.

  D. Concurrency            The design needs a cheap "should I interrupt?" classifier to run
                            WHILE the main answer is being generated. With -np 2 slots, does
                            the side call stall the main stream?

Run with llama-server already up (Qwopus3.5-4B Q4_K_M, -np 2):
    .venv\\Scripts\\python.exe experiment-streaming\\llama_prefill_probe.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:20055"
OUT = Path(__file__).resolve().parent / "runs" / "llama_prefill.json"

# A stand-in for the RAG context block: bulky, stable across a turn.
CONTEXT_BLOCK = "\n".join(
    f"[Документ {i}] Пункт {i}. Для поступления в институт кандидат представляет "
    f"заявление, документ об образовании, медицинское заключение и результаты "
    f"вступительных испытаний по профильным предметам. Сроки приёма документов "
    f"устанавливаются приказом директора института на соответствующий год."
    for i in range(1, 26)
)

SYSTEM = ("Ты — помощница приёмной комиссии института. Отвечай кратко, 1–3 предложения, "
          "только по существу заданного вопроса.")

# Growing transcript, as if arriving from streaming STT word by word.
TRANSCRIPT_STEPS = [
    "здравствуйте",
    "здравствуйте я хотел бы узнать",
    "здравствуйте я хотел бы узнать какие документы",
    "здравствуйте я хотел бы узнать какие документы нужны для поступления",
    "здравствуйте я хотел бы узнать какие документы нужны для поступления на очную форму",
    "здравствуйте я хотел бы узнать какие документы нужны для поступления на очную форму "
    "и до какого числа их принимают",
]


def timings(js: dict) -> dict:
    t = js.get("timings") or {}
    return {
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": round(t.get("prompt_ms", 0), 1),
        "prompt_tps": round(t.get("prompt_per_second") or 0, 1),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": round(t.get("predicted_ms", 0), 1),
        "predicted_tps": round(t.get("predicted_per_second") or 0, 1),
        "cache_n": t.get("cache_n"),
    }


async def complete(client: httpx.AsyncClient, prompt: str, *, n_predict: int = 1,
                   cache: bool = True, **extra) -> dict:
    r = await client.post(f"{BASE}/completion", json={
        "prompt": prompt, "n_predict": n_predict, "cache_prompt": cache,
        "temperature": 0.0, **extra,
    }, timeout=180.0)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# A. TTFT + streaming throughput
# --------------------------------------------------------------------------- #

async def exp_a(client: httpx.AsyncClient) -> dict:
    prompt = (f"{SYSTEM}\n\nКонтекст:\n{CONTEXT_BLOCK}\n\n"
              f"Вопрос: какие документы нужны для поступления?\nОтвет:")
    ttfts, first_sentence, totals, gaps_all = [], [], [], []
    text_out = ""
    for _ in range(3):
        chunks: list[tuple[float, str]] = []
        t0 = time.perf_counter()
        async with client.stream("POST", f"{BASE}/completion", json={
            "prompt": prompt, "n_predict": 120, "stream": True,
            "cache_prompt": True, "temperature": 0.0,
        }, timeout=180.0) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                js = json.loads(line[6:])
                tok = js.get("content", "")
                if tok:
                    chunks.append((time.perf_counter() - t0, tok))
                if js.get("stop"):
                    break
        if not chunks:
            continue
        ttfts.append(chunks[0][0])
        totals.append(chunks[-1][0])
        text_out = "".join(c[1] for c in chunks)
        # when is the first full sentence available? that's when TTS could start
        acc = ""
        for t, tok in chunks:
            acc += tok
            if any(p in acc for p in ".!?"):
                first_sentence.append(t)
                break
        gaps_all += [b[0] - a[0] for a, b in zip(chunks, chunks[1:])]
    return {
        "ttft_s_mean": round(statistics.fmean(ttfts), 3) if ttfts else None,
        "first_sentence_s_mean": round(statistics.fmean(first_sentence), 3) if first_sentence else None,
        "full_answer_s_mean": round(statistics.fmean(totals), 3) if totals else None,
        "inter_token_ms_p50": round(statistics.median(gaps_all) * 1000, 2) if gaps_all else None,
        "answer_sample": text_out.strip()[:300],
        "note": "first_sentence_s = earliest moment TTS could begin speaking",
    }


# --------------------------------------------------------------------------- #
# B/C. incremental prefill + prompt layout
# --------------------------------------------------------------------------- #

def build_suffix_last(transcript: str) -> str:
    """Volatile transcript at the very END -> cacheable prefix."""
    return (f"{SYSTEM}\n\nКонтекст:\n{CONTEXT_BLOCK}\n\n"
            f"Реплика абитуриента (в процессе): {transcript}")


def build_suffix_middle(transcript: str) -> str:
    """Transcript followed by a fixed instruction -> every update shifts the tail."""
    return (f"{SYSTEM}\n\nКонтекст:\n{CONTEXT_BLOCK}\n\n"
            f"Реплика абитуриента (в процессе): {transcript}\n\n"
            f"Реши, договорил ли абитуриент. Ответь строго JSON: "
            f'{{"done": true|false, "clarify": "..."}}')


async def exp_incremental(client: httpx.AsyncClient, builder, label: str) -> dict:
    rows = []
    # cold: make sure the prefix isn't already cached from a previous experiment
    await complete(client, "разогрев кэша другой строкой", n_predict=1)
    for i, tr in enumerate(TRANSCRIPT_STEPS):
        t0 = time.perf_counter()
        js = await complete(client, builder(tr), n_predict=1)
        wall = time.perf_counter() - t0
        rows.append({"step": i, "transcript_chars": len(tr),
                     "wall_s": round(wall, 3), **timings(js)})
    return {"label": label, "steps": rows,
            "first_step_prompt_n": rows[0].get("prompt_n"),
            "later_steps_prompt_n": [r.get("prompt_n") for r in rows[1:]],
            "first_step_wall_s": rows[0]["wall_s"],
            "later_steps_wall_s": [r["wall_s"] for r in rows[1:]]}


# --------------------------------------------------------------------------- #
# D. concurrency: side classifier during main generation
# --------------------------------------------------------------------------- #

async def exp_concurrency(client: httpx.AsyncClient) -> dict:
    main_prompt = (f"{SYSTEM}\n\nКонтекст:\n{CONTEXT_BLOCK}\n\n"
                   f"Вопрос: расскажи подробно про порядок приёма документов.\nОтвет:")
    side_prompt = (f"Абитуриент заговорил, пока ассистент отвечал. Его слова: "
                   f'"нет подождите я про другое хотел спросить". '
                   f'Требует ли он прервать ответ? Ответь строго JSON: {{"interrupt": true|false}}')

    async def main_stream(record: list) -> str:
        t0 = time.perf_counter()
        acc = ""
        async with client.stream("POST", f"{BASE}/completion", json={
            "prompt": main_prompt, "n_predict": 150, "stream": True,
            "cache_prompt": True, "temperature": 0.0,
        }, timeout=180.0) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                js = json.loads(line[6:])
                if js.get("content"):
                    record.append(time.perf_counter() - t0)
                    acc += js["content"]
                if js.get("stop"):
                    break
        return acc

    def gaps_of(rec: list) -> dict:
        g = [b - a for a, b in zip(rec, rec[1:])]
        if not g:
            return {}
        gs = sorted(g)
        return {"tokens": len(rec),
                "inter_token_ms_p50": round(statistics.median(gs) * 1000, 2),
                "inter_token_ms_p99": round(gs[min(len(gs) - 1, int(len(gs) * 0.99))] * 1000, 2),
                "inter_token_ms_max": round(gs[-1] * 1000, 2)}

    # 1) main stream alone
    solo_rec: list = []
    t0 = time.perf_counter()
    await main_stream(solo_rec)
    solo_wall = time.perf_counter() - t0

    # 2) main stream while a side classifier fires repeatedly on the other slot
    conc_rec: list = []
    side_lat: list = []
    stop = False

    async def side_loop():
        while not stop:
            s = time.perf_counter()
            try:
                js = await complete(client, side_prompt, n_predict=24,
                                    json_schema={"type": "object",
                                                 "properties": {"interrupt": {"type": "boolean"}},
                                                 "required": ["interrupt"]})
                side_lat.append((time.perf_counter() - s, js.get("content", "").strip()))
            except Exception as exc:  # noqa: BLE001
                side_lat.append((time.perf_counter() - s, f"ERROR {exc}"))
            await asyncio.sleep(0.05)

    task = asyncio.create_task(side_loop())
    t0 = time.perf_counter()
    await main_stream(conc_rec)
    conc_wall = time.perf_counter() - t0
    stop = True
    await task

    return {
        "solo": {"wall_s": round(solo_wall, 3), **gaps_of(solo_rec)},
        "concurrent": {"wall_s": round(conc_wall, 3), **gaps_of(conc_rec)},
        "slowdown_x": round(conc_wall / solo_wall, 2) if solo_wall else None,
        "side_calls": len(side_lat),
        "side_latency_s_mean": round(statistics.fmean([s for s, _ in side_lat]), 3) if side_lat else None,
        "side_latency_s_max": round(max(s for s, _ in side_lat), 3) if side_lat else None,
        "side_outputs_sample": [o for _, o in side_lat[:3]],
    }


# --------------------------------------------------------------------------- #

async def main() -> None:
    out: dict = {}
    async with httpx.AsyncClient() as client:
        print("=== A. streaming TTFT / throughput ===", flush=True)
        out["a_streaming"] = await exp_a(client)
        print(json.dumps(out["a_streaming"], ensure_ascii=False, indent=2), flush=True)

        print("\n=== B. incremental prefill, transcript LAST (cacheable) ===", flush=True)
        out["b_suffix_last"] = await exp_incremental(client, build_suffix_last, "transcript at end")
        print(json.dumps(out["b_suffix_last"], ensure_ascii=False, indent=2), flush=True)

        print("\n=== C. incremental prefill, instruction AFTER transcript ===", flush=True)
        out["c_suffix_middle"] = await exp_incremental(client, build_suffix_middle,
                                                      "instruction after transcript")
        print(json.dumps(out["c_suffix_middle"], ensure_ascii=False, indent=2), flush=True)

        print("\n=== D. side classifier concurrent with main generation ===", flush=True)
        out["d_concurrency"] = await exp_concurrency(client)
        print(json.dumps(out["d_concurrency"], ensure_ascii=False, indent=2), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
