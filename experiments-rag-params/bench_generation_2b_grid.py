"""Generation parameter grid for Qwen 2B: temperature + max_tokens sweep.

Finds the sweet spot for quality/latency on the 2B model specifically.

Usage:
  uv run python experiments-rag-params/bench_generation_2b_grid.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import RagConfig, QWEN_2B
from rag.generate import LlamaServer

RUNS = Path(__file__).resolve().parent / "runs"


def run_grid():
    retrieved_path = RUNS / "retrieved.json"
    if not retrieved_path.exists():
        raise FileNotFoundError(f"Run dump_retrieval.py first: {retrieved_path}")

    with open(retrieved_path, encoding="utf-8") as f:
        retrievals = json.load(f)  # list of dicts with 'question', 'top_chunks', etc.

    params_grid = [
        (200, 0.1),
        (200, 0.2),
        (200, 0.3),
        (300, 0.1),
        (300, 0.2),
        (300, 0.3),
        (400, 0.1),
        (400, 0.2),
        (500, 0.2),
    ]

    results = {}

    for max_tokens, temperature in params_grid:
        cfg = RagConfig(
            llm_gguf=str(QWEN_2B),
            max_tokens=max_tokens,
            temperature=temperature,
            disable_thinking=True,
        )

        print(f"\n=== 2B: max_tokens={max_tokens}, temp={temperature} ===")
        server = LlamaServer(cfg)
        server.start()
        time.sleep(2)

        rows = []
        timings = []

        for item in retrievals:
            q = item["question"]
            chunks = item["top_chunks"]
            ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))

            # Construct messages in chat format
            from rag.generate import SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Контекст из документов:\n\n{ctx}\n\nВопрос: {q}"},
            ]

            t0 = time.perf_counter()
            result = server.complete(messages, cfg)
            gen_sec = result["gen_sec"]

            answer = result["answer"]
            toks = result.get("completion_tokens") or len(answer.split())
            tps = result.get("tps") or (toks / gen_sec if gen_sec > 0 else 0)

            rows.append({
                "id": item["id"],
                "question": q,
                "answer_gen": answer,
                "gen_sec": round(gen_sec, 2),
                "tokens": toks,
                "tps": round(tps, 1),
            })
            timings.append(gen_sec)

        server.stop()

        avg_sec = sum(timings) / len(timings)
        p90_sec = sorted(timings)[int(len(timings) * 0.9)]
        avg_tps = sum(r["tps"] for r in rows) / len(rows)

        results[f"max{max_tokens}_temp{temperature:.1f}"] = {
            "params": {"max_tokens": max_tokens, "temperature": temperature},
            "gen_avg_sec": round(avg_sec, 2),
            "gen_p90_sec": round(p90_sec, 2),
            "avg_tps": round(avg_tps, 1),
            "rows": rows,
        }

        print(f"  avg {avg_sec:.2f}s | p90 {p90_sec:.2f}s | {avg_tps:.0f} tps")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"grid_2b_generation_{ts}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n\nsaved -> {out}")

    # summary table
    print("\n" + "=" * 60)
    print(f"{'config':25} {'avg_sec':>8} {'p90_sec':>8} {'tps':>8}")
    for name in sorted(results.keys()):
        r = results[name]
        print(f"{name:25} {r['gen_avg_sec']:>8.2f} {r['gen_p90_sec']:>8.2f} {r['avg_tps']:>8.1f}")


if __name__ == "__main__":
    run_grid()
