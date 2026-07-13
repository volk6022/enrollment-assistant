"""Feasibility bench for the 0.8B distilled model (Q4) vs the 2B (Q8) default.

Generation-only (reuses runs/retrieved.json), same 32 grounded eval queries.
Prints latency + full answers so quality (grounding, RU fluency, truncation)
can be eyeballed against the 2B grid.

Usage:
  uv run python experiments-rag-params/bench_generation_08b.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import RagConfig
from rag.generate import LlamaServer, SYSTEM_PROMPT

RUNS = Path(__file__).resolve().parent / "runs"
QWEN_08B = r"G:\lmstudio\models\Jackrong\Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-GGUF\Qwen3.5-0.8B.Q4_K_M.gguf"


def run():
    retrievals = json.load(open(RUNS / "retrieved.json", encoding="utf-8"))
    cfg = RagConfig(llm_gguf=QWEN_08B, llm_port=20077, max_tokens=300,
                    temperature=0.2, disable_thinking=True)
    server = LlamaServer(cfg)
    server.start()
    time.sleep(2)

    rows, timings = [], []
    for item in retrievals:
        q = item["question"]
        ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(item["top_chunks"]))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст из документов:\n\n{ctx}\n\nВопрос: {q}"},
        ]
        r = server.complete(messages, cfg)
        rows.append({"id": item["id"], "question": q, "answer_gen": r["answer"],
                     "gen_sec": round(r["gen_sec"], 2), "tokens": r.get("completion_tokens")})
        timings.append(r["gen_sec"])
    server.stop()

    timings.sort()
    avg = sum(timings) / len(timings)
    p90 = timings[int(len(timings) * 0.9)]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"gen_08b_{ts}.json"
    out.write_text(json.dumps({"model": "Qwen3.5-0.8B.Q4_K_M", "gen_avg_sec": round(avg, 2),
                               "gen_p90_sec": round(p90, 2), "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"0.8B Q4  avg {avg:.2f}s  p90 {p90:.2f}s  -> {out.name}")
    for r in rows[:4]:
        print("\nQ:", r["question"])
        print("A:", r["answer_gen"][:400])


if __name__ == "__main__":
    run()
