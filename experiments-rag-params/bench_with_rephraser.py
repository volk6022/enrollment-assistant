"""Test pipeline with a rephraser step: question → rephrase → answer.

The rephraser is the same Qwen2B in llama.cpp. Compares:
  - baseline: question → context → answer
  - with rephraser: question → [2B rephrases] → rephrased_q → context → answer

Measures impact on latency and subjective quality.

Usage:
  uv run python experiments-rag-params/bench_with_rephraser.py
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

REPHRASER_PROMPT = (
    "Ты — помощник приёмной комиссии. Перепиши вопрос абитуриента так, чтобы он был "
    "более специфичен и прямо отражал то, что нужно узнать из предоставленных документов. "
    "Ответь ТОЛЬКО переформулированным вопросом, без пояснений."
)


def run_test():
    retrieved_path = RUNS / "retrieved.json"
    if not retrieved_path.exists():
        raise FileNotFoundError(f"Run dump_retrieval.py first: {retrieved_path}")

    with open(retrieved_path, encoding="utf-8") as f:
        retrievals = json.load(f)

    cfg = RagConfig(max_tokens=200, temperature=0.2, disable_thinking=True)
    server = LlamaServer(cfg)
    server.start()
    time.sleep(2)

    rows = []

    for item in retrievals:
        q = item["question"]
        chunks = item["top_chunks"]
        ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))

        # baseline: direct answer
        messages_baseline = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n\n{ctx}\n\nВопрос: {q}"},
        ]
        t0 = time.perf_counter()
        result_baseline = server.complete(messages_baseline, cfg)
        time_baseline = result_baseline["gen_sec"]
        answer_baseline = result_baseline["answer"]

        # with rephraser
        messages_rephrase = [
            {"role": "system", "content": REPHRASER_PROMPT},
            {"role": "user", "content": f"Контекст:\n\n{ctx}\n\nВопрос: {q}"},
        ]
        t0 = time.perf_counter()
        result_rephrase = server.complete(messages_rephrase, cfg)
        time_rephrase = result_rephrase["gen_sec"]
        q_rephrased = result_rephrase["answer"]

        # generate answer with rephrased question
        messages_answer = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n\n{ctx}\n\nВопрос: {q_rephrased}"},
        ]
        t0 = time.perf_counter()
        result_answer = server.complete(messages_answer, cfg)
        time_answer = result_answer["gen_sec"]
        answer_rephrased = result_answer["answer"]

        rows.append({
            "id": item["id"],
            "original_question": q,
            "rephrased_question": q_rephrased,
            "answer_baseline": answer_baseline,
            "time_baseline_sec": round(time_baseline, 2),
            "answer_with_rephrase": answer_rephrased,
            "time_rephrase_sec": round(time_rephrase, 2),
            "time_answer_sec": round(time_answer, 2),
            "time_total_rephrase": round(time_rephrase + time_answer, 2),
        })

        if len(rows) % 8 == 0:
            print(f"  {len(rows)}/32...")

    server.stop()

    # aggregate
    times_baseline = [r["time_baseline_sec"] for r in rows]
    times_total_rephrase = [r["time_total_rephrase"] for r in rows]

    avg_baseline = sum(times_baseline) / len(times_baseline)
    avg_rephrase = sum(times_total_rephrase) / len(times_total_rephrase)
    delta_pct = ((avg_rephrase - avg_baseline) / avg_baseline) * 100

    summary = {
        "baseline_avg_sec": round(avg_baseline, 2),
        "with_rephrase_avg_sec": round(avg_rephrase, 2),
        "delta_percent": round(delta_pct, 1),
        "rows": rows,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"with_rephraser_{ts}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")

    print("\n" + "=" * 70)
    print(f"Baseline (direct):        {avg_baseline:.2f}s")
    print(f"With rephraser (2 calls): {avg_rephrase:.2f}s")
    print(f"Delta:                    +{delta_pct:.1f}% (tradeoff: latency vs clarity)")
    print("\nSample (first item):")
    r = rows[0]
    print(f"  Q orig:      {r['original_question']}")
    print(f"  Q rephrased: {r['rephrased_question']}")
    print(f"  Baseline:    {r['answer_baseline'][:120]}...")
    print(f"  W/ rephrase: {r['answer_with_rephrase'][:120]}...")


if __name__ == "__main__":
    run_test()
