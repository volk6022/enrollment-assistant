"""Does a rephraser help on CONVERSATIONAL (spoken) input? Full-pipeline A/B/C.

Motivation: the earlier rephraser tests used the clean, book-phrased eval
questions, where a rephraser has nothing to fix. Real voice input is messy
("слушайте, а кто там врачи будут?"). This bench runs the FULL pipeline
(retrieval + rerank + generation, not frozen retrieval) so we can see the
rephraser's real job: rewrite a messy spoken question into something that
retrieves the right evidence.

Three arms, scored against the SAME gold sources (experiment eval_set):
  A. formal            — original book-phrased question (upper-bound reference)
  B. conversational    — messy spoken question, no rephraser (baseline)
  C. conversational+RP — spoken question -> rephrase (question-only, few-shot,
                         cut at first '?') -> pipeline

Everything is packed into ONE json for side-by-side comparison.

Usage: uv run python experiments-rag-params/bench_conversational_rephraser.py
"""
from __future__ import annotations

import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for score.py (dir has a hyphen)

from rag.config import DEFAULT
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.generate import LlamaServer
from score import score_item

RUNS = Path(__file__).resolve().parent / "runs"
EVAL = Path(__file__).resolve().parent.parent / "experiment-docx-processing" / "out" / "eval" / "eval_set.json"

REPHRASER_SYSTEM = (
    "Ты переформулируешь вопрос абитуриента приёмной комиссии, чтобы он был точнее "
    "и полнее для поиска по нормативным документам. Верни РОВНО ОДИН "
    "переформулированный вопрос, заканчивающийся знаком «?». НЕ отвечай на вопрос, "
    "НЕ добавляй пояснений, НЕ используй документы — только перепиши сам вопрос "
    "строгим канцелярским языком с правильными терминами."
)
FEWSHOT = [
    {"role": "user", "content": "Слушайте, а что там по здоровью надо, я подхожу вообще?"},
    {"role": "assistant", "content": "Какие требования к состоянию здоровья и категории годности предъявляются к кандидату при поступлении?"},
    {"role": "user", "content": "Ну когда крайний срок бумаги подать?"},
    {"role": "assistant", "content": "В какие сроки подаётся заявление о приёме и документы кандидата?"},
    {"role": "user", "content": "А по физре сколько надо набрать чтоб сдать?"},
    {"role": "assistant", "content": "Сколько баллов нужно набрать по физической подготовке для успешной сдачи вступительного испытания?"},
]


def first_question(text: str) -> str:
    t = text.strip()
    i = t.find("?")
    return (t[: i + 1]).strip() if i != -1 else t


def rephrase(server, q: str) -> tuple[str, float]:
    msgs = [{"role": "system", "content": REPHRASER_SYSTEM}, *FEWSHOT, {"role": "user", "content": q}]
    r = server.complete(msgs, DEFAULT)
    return first_question(r["answer"]), r["gen_sec"]


def r5(chunks, item):
    s = score_item(chunks, item)
    return {"src_recall@5": s["source_recall@5"], "quote_hit@5": s["quote_hit@5"],
            "mrr": round(s["mrr"], 3), "found_rank": s["found_rank"]}


def run():
    conv = json.load(open(RUNS / "conversational_questions.json", encoding="utf-8"))
    gold = {it["id"]: it for it in json.load(open(EVAL, encoding="utf-8"))}

    print("Loading indexes (bge-m3 + reranker)...")
    idx = Indexes()
    server = LlamaServer(DEFAULT); server.start(); time.sleep(2)
    pipe = Pipeline(idx, server=server, cfg=DEFAULT)

    rows = []
    for i, c in enumerate(conv, 1):
        cid = c["id"]; item = gold[cid]
        fq, cq = c["formal"], c["conversational"]

        # A. formal (reference)
        a_f = pipe.answer(fq)
        # B. conversational, no rephraser
        a_b = pipe.answer(cq)
        # C. conversational + rephraser
        rq, t_rp = rephrase(server, cq)
        a_c = pipe.answer(rq)

        rows.append({
            "id": cid, "gold_source": item["gold_source"],
            "formal_q": fq, "conversational_q": cq, "rephrased_q": rq,
            "retrieval": {
                "formal": r5(a_f["top_chunks"], item),
                "conv_baseline": r5(a_b["top_chunks"], item),
                "conv_rephraser": r5(a_c["top_chunks"], item),
            },
            "answers": {
                "formal": a_f["answer"],
                "conv_baseline": a_b["answer"],
                "conv_rephraser": a_c["answer"],
            },
            "latency_sec": {
                "formal_total": round((a_f["timings"]["search_ms"] + a_f["timings"].get("gen_ms", 0)) / 1000, 2),
                "conv_baseline_total": round((a_b["timings"]["search_ms"] + a_b["timings"].get("gen_ms", 0)) / 1000, 2),
                "conv_rephraser_total": round(t_rp + (a_c["timings"]["search_ms"] + a_c["timings"].get("gen_ms", 0)) / 1000, 2),
                "rephrase_step": round(t_rp, 2),
            },
        })
        print(f"  [{i}/32] {cid}")
    server.stop()

    def agg(arm, metric):
        return round(sum(r["retrieval"][arm][metric] for r in rows) / len(rows), 4)

    def lat(key):
        return round(sum(r["latency_sec"][key] for r in rows) / len(rows), 2)

    aggregate = {
        arm: {"src_recall@5": agg(arm, "src_recall@5"),
              "quote_hit@5": agg(arm, "quote_hit@5"),
              "mrr": agg(arm, "mrr")}
        for arm in ("formal", "conv_baseline", "conv_rephraser")
    }
    aggregate["latency_avg_sec"] = {
        "formal": lat("formal_total"),
        "conv_baseline": lat("conv_baseline_total"),
        "conv_rephraser": lat("conv_rephraser_total"),
        "rephrase_step": lat("rephrase_step"),
    }
    # how often rephraser recovered a retrieval that the baseline missed
    recovered = sum(1 for r in rows
                    if r["retrieval"]["conv_baseline"]["src_recall@5"] == 0
                    and r["retrieval"]["conv_rephraser"]["src_recall@5"] == 1)
    lost = sum(1 for r in rows
               if r["retrieval"]["conv_baseline"]["src_recall@5"] == 1
               and r["retrieval"]["conv_rephraser"]["src_recall@5"] == 0)
    aggregate["src_recall@5_recovered_by_rephraser"] = recovered
    aggregate["src_recall@5_lost_by_rephraser"] = lost

    out = {
        "meta": {
            "model": "Qwen3.5-2B.Q8_0", "prompt": "concise", "temperature": DEFAULT.temperature,
            "max_tokens": DEFAULT.max_tokens, "fused_top": DEFAULT.fused_top, "final_top": DEFAULT.final_top,
            "input": "conversational (haiku-generated) vs formal eval questions",
            "arms": {"formal": "original book-phrased (reference)",
                     "conv_baseline": "conversational, NO rephraser",
                     "conv_rephraser": "conversational -> rephrase(question-only) -> pipeline"},
            "n": len(rows),
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RUNS / f"conversational_rephraser_ab_{ts}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"{'arm':18} {'src_recall@5':>13} {'quote_hit@5':>12} {'mrr':>7} {'lat_s':>7}")
    for arm in ("formal", "conv_baseline", "conv_rephraser"):
        a = aggregate[arm]
        print(f"{arm:18} {a['src_recall@5']:>13} {a['quote_hit@5']:>12} {a['mrr']:>7} "
              f"{aggregate['latency_avg_sec'][arm]:>7}")
    print(f"\nrephraser recovered {recovered} retrievals baseline missed; lost {lost}.")
    print(f"rephrase step avg: {aggregate['latency_avg_sec']['rephrase_step']}s")
    print(f"\nsaved -> {path.name}")


if __name__ == "__main__":
    run()
