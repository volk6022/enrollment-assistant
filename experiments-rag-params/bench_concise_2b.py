"""Does a concise-answer instruction fix the max_tokens=200 truncation defect?

The default SYSTEM_PROMPT yields median ~184 tok answers, so a 200-tok cap
truncates ~41% mid-sentence. Hypothesis: instructing "2-3 sentences" cuts the
natural length below the cap, keeping answers complete AND fast.

Runs 2B (Q8) on the 32 eval queries with a concise prompt at max_tokens=200/300.

Usage: uv run python experiments-rag-params/bench_concise_2b.py
"""
from __future__ import annotations

import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import RagConfig, QWEN_2B
from rag.generate import LlamaServer

RUNS = Path(__file__).resolve().parent / "runs"

CONCISE_PROMPT = (
    "Ты — помощник приёмной комиссии юридического вуза (ДВЮИ МВД). "
    "Отвечай ТОЛЬКО на основе приведённых фрагментов документов, по-русски, "
    "разговорным языком. Не выдумывай факты. Отвечай КРАТКО — 2–4 предложения, "
    "без списков и заголовков, самое важное. Если во фрагментах нет ответа — "
    "скажи, что точной информации нет, стоит уточнить в приёмной комиссии. "
    "Ссылайся на источник в скобках."
)


def run():
    retrievals = json.load(open(RUNS / "retrieved.json", encoding="utf-8"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    for max_tokens in (200, 300):
        cfg = RagConfig(llm_gguf=str(QWEN_2B), llm_port=20077, max_tokens=max_tokens,
                        temperature=0.2, disable_thinking=True)
        server = LlamaServer(cfg); server.start(); time.sleep(2)
        rows, timings = [], []
        for item in retrievals:
            ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(item["top_chunks"]))
            msgs = [{"role": "system", "content": CONCISE_PROMPT},
                    {"role": "user", "content": f"Контекст из документов:\n\n{ctx}\n\nВопрос: {item['question']}"}]
            r = server.complete(msgs, cfg)
            rows.append({
                "id": item["id"],
                "question": item["question"],
                "answer": r["answer"],
                "tokens": r.get("completion_tokens"),
                "gen_sec": round(r["gen_sec"], 2),
                "sources": [f"{c.get('source','?')}" + (f" п.{c['point']}" if c.get('point') else "")
                            for c in item["top_chunks"]],
            })
            timings.append(r["gen_sec"])
        server.stop()
        timings.sort()
        toks = [x["tokens"] or 0 for x in rows]
        capped = sum(1 for t in toks if t >= max_tokens)
        cut = sum(1 for x in rows if not x["answer"].rstrip().endswith((".", "!", "?", "»", ")")))
        avg, p90 = sum(timings) / len(timings), timings[int(len(timings) * 0.9)]
        summary = {
            "config": {"model": "Qwen3.5-2B.Q8_0", "prompt": "concise (2-4 sentences)",
                       "max_tokens": max_tokens, "temperature": 0.2,
                       "disable_thinking": True, "rephraser": False},
            "gen_avg_sec": round(avg, 2), "gen_p90_sec": round(p90, 2),
            "avg_tokens": round(sum(toks) / len(toks), 1),
            "capped": f"{capped}/{len(rows)}", "not_clean_end": f"{cut}/{len(rows)}",
            "rows": rows,
        }
        out = RUNS / f"gen_2b_concise_max{max_tokens}_{ts}.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"concise @max{max_tokens}: avg {avg:.2f}s  p90 {p90:.2f}s  avg_tok={sum(toks)/len(toks):.0f}"
              f"  capped={capped}/32  not-clean-end={cut}/32  -> {out.name}")


if __name__ == "__main__":
    run()
