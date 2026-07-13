"""Rephraser done RIGHT: rewrite the QUESTION only, never showing it the context.

The first rephraser attempt (bench_with_rephraser.py) fed the retrieved chunks to
the rephraser, so a reasoning model just answered instead of rephrasing (0/32
outputs were questions; meaning sometimes inverted; +47% latency). This is the
corrected design:

  - rephraser sees ONLY the raw question (no chunks, so nothing to "answer")
  - few-shot forces question-in -> question-out
  - we keep only up to the first '?'  (hard guarantee it stays a question)

Then the answer stage uses the concise production SYSTEM_PROMPT, comparing:
  baseline:  question              -> context -> answer
  rephrase:  question -> rewrite_q -> context -> answer

Reports: (a) is the rephrase actually a question, (b) latency delta,
(c) how often it changed retrieval-relevant wording, (d) side-by-side answers.

Usage: uv run python experiments-rag-params/bench_rephraser_correct.py
"""
from __future__ import annotations

import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import RagConfig, QWEN_2B
from rag.generate import LlamaServer, SYSTEM_PROMPT

RUNS = Path(__file__).resolve().parent / "runs"

REPHRASER_SYSTEM = (
    "Ты переформулируешь вопрос абитуриента приёмной комиссии, чтобы он был точнее "
    "и полнее для поиска по нормативным документам. Верни РОВНО ОДИН "
    "переформулированный вопрос, заканчивающийся знаком «?». НЕ отвечай на вопрос, "
    "НЕ добавляй пояснений, НЕ используй документы — только перепиши сам вопрос."
)
# few-shot: question in -> better question out (never a statement/answer)
FEWSHOT = [
    {"role": "user", "content": "А что по здоровью надо?"},
    {"role": "assistant", "content": "Какие требования к состоянию здоровья и категории годности предъявляются к абитуриенту при поступлении?"},
    {"role": "user", "content": "Когда подавать?"},
    {"role": "assistant", "content": "В какие сроки подаётся заявление о приёме и приёме документов?"},
]


def first_question(text: str) -> str:
    """Keep text up to and including the first '?'; guarantees question form."""
    t = text.strip()
    i = t.find("?")
    return (t[: i + 1]).strip() if i != -1 else t


def run():
    retrievals = json.load(open(RUNS / "retrieved.json", encoding="utf-8"))
    cfg = RagConfig(llm_gguf=str(QWEN_2B), llm_port=20077, max_tokens=200,
                    temperature=0.2, disable_thinking=True)
    server = LlamaServer(cfg); server.start(); time.sleep(2)

    rows = []
    t_base, t_reph_total = [], []
    for item in retrievals:
        q = item["question"]
        ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(item["top_chunks"]))

        # --- rephrase stage: QUESTION ONLY, no context ---
        rmsgs = [{"role": "system", "content": REPHRASER_SYSTEM}, *FEWSHOT,
                 {"role": "user", "content": q}]
        rr = server.complete(rmsgs, cfg)
        rq = first_question(rr["answer"])
        t_reph = rr["gen_sec"]

        # --- answer baseline (raw question) ---
        bmsgs = [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": f"Контекст из документов:\n\n{ctx}\n\nВопрос: {q}"}]
        br = server.complete(bmsgs, cfg)
        t_base.append(br["gen_sec"])

        # --- answer with rephrased question ---
        amsgs = [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": f"Контекст из документов:\n\n{ctx}\n\nВопрос: {rq}"}]
        ar = server.complete(amsgs, cfg)
        t_reph_total.append(t_reph + ar["gen_sec"])

        rows.append({
            "id": item["id"], "q_orig": q, "q_rephrased": rq,
            "is_question": rq.endswith("?"),
            "changed": rq.strip().lower() != q.strip().lower(),
            "answer_baseline": br["answer"], "answer_rephrased": ar["answer"],
            "t_rephrase": round(t_reph, 2), "t_base": round(br["gen_sec"], 2),
            "t_total_rephrase": round(t_reph + ar["gen_sec"], 2),
        })
    server.stop()

    avg_b = sum(t_base) / len(t_base)
    avg_r = sum(t_reph_total) / len(t_reph_total)
    n_q = sum(r["is_question"] for r in rows)
    n_ch = sum(r["changed"] for r in rows)
    summary = {
        "design": "rephraser sees question only (no context) + few-shot + cut at first '?'",
        "baseline_avg_sec": round(avg_b, 2),
        "with_rephrase_avg_sec": round(avg_r, 2),
        "delta_percent": round((avg_r - avg_b) / avg_b * 100, 1),
        "rephrase_is_question": f"{n_q}/{len(rows)}",
        "rephrase_changed_wording": f"{n_ch}/{len(rows)}",
        "rows": rows,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"rephraser_correct_{ts}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"CORRECT rephraser (question-only, few-shot, cut at '?')")
    print(f"  rephrase IS a question:   {n_q}/{len(rows)}   (broken version: 0/32)")
    print(f"  rephrase changed wording: {n_ch}/{len(rows)}")
    print(f"  baseline avg:             {avg_b:.2f}s")
    print(f"  with rephraser avg:       {avg_r:.2f}s   (+{summary['delta_percent']:.0f}% for 2 calls)")
    print(f"  saved -> {out.name}")
    print("\n--- samples ---")
    for r in rows[:4]:
        print(f"\nQ orig: {r['q_orig']}")
        print(f"Q reph: {r['q_rephrased']}")
        print(f"  base: {r['answer_baseline'][:150]}")
        print(f"  reph: {r['answer_rephrased'][:150]}")


if __name__ == "__main__":
    run()
