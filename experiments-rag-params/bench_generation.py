"""Generation benchmark: Qwen3.5 2B vs 4B on frozen retrieval results.

Reads runs/retrieved.json (from dump_retrieval.py) so the GPU holds only the LLM.
For each model: start llama-server, answer every eval question from its retrieved
context, record answer + generation latency + tokens/s. Writes per-model results.

Answer *correctness* grading is left to a separate LLM-judge pass — here we
capture answers + latency so quality can be judged offline without re-running.

Usage:
  uv run python experiments-rag-params/bench_generation.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import DEFAULT, QWEN_2B, QWEN_4B, RagConfig
from rag.generate import LlamaServer, build_messages

RUNS = Path(__file__).resolve().parent / "runs"
MODELS = [("qwen2b", str(QWEN_2B)), ("qwen4b", str(QWEN_4B))]


def bench_model(name: str, gguf: str, retrieved: list[dict]) -> dict:
    cfg = dataclasses.replace(DEFAULT, llm_gguf=gguf)
    rows = []
    with LlamaServer(cfg) as server:
        for r in retrieved:
            top = r["top_chunks"][: cfg.final_top]
            messages = build_messages(r["question"], top)
            g = server.complete(messages, cfg)
            rows.append({
                "id": r["id"],
                "topic": r["topic"],
                "question": r["question"],
                "answer_gold": r["answer_gold"],
                "answer_gen": g["answer"],
                "gen_sec": round(g["gen_sec"], 2),
                "completion_tokens": g["completion_tokens"],
                "tps": round(g["tps"], 1) if g["tps"] else None,
            })
            print(f"  [{name}] {r['id']:16} {g['gen_sec']:.2f}s  {g['tps'] or 0:.0f} tps")
    lat = [x["gen_sec"] for x in rows]
    tps = [x["tps"] for x in rows if x["tps"]]
    summary = {
        "model": name,
        "gguf": gguf,
        "n": len(rows),
        "gen_sec_avg": round(sum(lat) / len(lat), 2),
        "gen_sec_p90": round(sorted(lat)[int(0.9 * len(lat)) - 1], 2),
        "tps_avg": round(sum(tps) / len(tps), 1) if tps else None,
    }
    (RUNS / f"generation_{name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    retrieved = json.loads((RUNS / "retrieved.json").read_text(encoding="utf-8"))
    print(f"loaded {len(retrieved)} retrieved contexts")
    summaries = []
    for name, gguf in MODELS:
        if not Path(gguf).exists():
            print(f"skip {name}: {gguf} not found")
            continue
        print(f"\n=== {name} ===")
        summaries.append(bench_model(name, gguf, retrieved))
        time.sleep(3)  # let the previous server fully release port + VRAM

    print("\n" + "=" * 60)
    print(f"{'model':10} {'gen_avg':>9} {'gen_p90':>9} {'tps':>7}")
    for s in summaries:
        print(f"{s['model']:10} {s['gen_sec_avg']:>8.2f}s {s['gen_sec_p90']:>8.2f}s {s['tps_avg'] or 0:>7.1f}")


if __name__ == "__main__":
    main()
