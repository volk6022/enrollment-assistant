"""Conversational input, 4 arms — does multi-query (augment, don't replace) +
a keyword-preserving rephraser prompt beat plain replace-rephrasing?

Prior run showed: conversational input tanks retrieval (recall 0.875->0.625),
and a REPLACE-rephraser is a wash (net +2 recall but -2 quote_hit) because it
paraphrases away strong keywords (паспорт, СВО, адъюнктура) and hallucinates
unknown terms. Two fixes tested here:

  1. rephraser prompt v2 — explicitly preserve exact terms/abbreviations, never
     invent definitions (few-shot demonstrates it).
  2. multi-query retrieval — retrieve on BOTH the original conversational query
     AND the v2 rephrase, RRF-union the candidate pools, rerank with the
     ORIGINAL query (so user keywords are never lost), generate on the original.

Arms (all scored vs the same gold sources):
  A. formal              — book-phrased question (upper-bound reference)
  B. conv_baseline       — conversational, no rephraser
  C. conv_rephrase_v2    — conversational -> v2 rephrase -> REPLACE -> pipeline
  D. conv_multiquery_v2  — conversational + v2 rephrase -> RRF union -> rerank(orig)

One combined JSON out.

Usage: uv run python experiments-rag-params/bench_conversational_multiquery.py
"""
from __future__ import annotations

import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # score.py (dir has a hyphen)

from rag.config import DEFAULT
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.embed import embed_query
from rag.retrieve import rrf_fuse
from rag.rerank import rerank
from rag.generate import LlamaServer, build_messages
from score import score_item

RUNS = Path(__file__).resolve().parent / "runs"
EVAL = Path(__file__).resolve().parent.parent / "experiment-docx-processing" / "out" / "eval" / "eval_set.json"

# ---- rephraser prompt v2: preserve keywords, never invent -------------------
REPHRASER_V2 = (
    "Ты переписываешь разговорный вопрос абитуриента в чёткий поисковый вопрос "
    "для базы нормативных документов вуза (ДВЮИ МВД). Правила:\n"
    "1. СОХРАНИ дословно все конкретные термины, названия документов и аббревиатуры "
    "из вопроса (паспорт, СНИЛС, ЕГЭ, СВО, адъюнктура, физподготовка и т.п.). "
    "НЕ заменяй их синонимами или общими словами.\n"
    "2. НЕ придумывай определения и факты. Незнакомый термин оставь как есть.\n"
    "3. Убери просторечие и слова-паразиты, добавь недостающие официальные термины, "
    "но НЕ меняй предмет вопроса.\n"
    "4. Верни РОВНО ОДИН вопрос, заканчивающийся «?». Без пояснений."
)
FEWSHOT_V2 = [
    {"role": "user", "content": "А паспорт обязательно надо подавать?"},
    {"role": "assistant", "content": "Обязательно ли предоставлять паспорт при подаче документов на поступление?"},
    {"role": "user", "content": "Вот, отец в СВО участвует, есть какая-нибудь квота для нас?"},
    {"role": "assistant", "content": "Какая квота при поступлении предусмотрена для детей участников СВО?"},
    {"role": "user", "content": "По физре сколько баллов надо набрать чтоб сдать?"},
    {"role": "assistant", "content": "Сколько баллов нужно набрать по физической подготовке (физподготовке) для сдачи вступительного испытания?"},
    {"role": "user", "content": "Адъюнктура — когда туда документы подавать?"},
    {"role": "assistant", "content": "В какие сроки подаются документы для поступления в адъюнктуру?"},
]


def first_question(text: str) -> str:
    t = text.strip()
    i = t.find("?")
    return (t[: i + 1]).strip() if i != -1 else t


def rephrase_v2(server, q: str):
    msgs = [{"role": "system", "content": REPHRASER_V2}, *FEWSHOT_V2, {"role": "user", "content": q}]
    r = server.complete(msgs, DEFAULT)
    return first_question(r["answer"]), r["gen_sec"]


def multiquery_candidates(idx, queries: list[str], cfg=DEFAULT):
    """RRF-union the dense+sparse ranked lists of several queries."""
    ranked = []
    for q in queries:
        qvec = embed_query(q, cfg)
        ranked.append(idx.dense_search(qvec, cfg.dense_top))
        ranked.append(idx.bm25_search(q, cfg.bm25_top))
    fused = rrf_fuse(*ranked, k=cfg.rrf_k)[: cfg.fused_top]
    return [dict(idx.chunks[i], rrf_score=s) for i, s in fused]


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

        a_f = pipe.answer(fq)                    # A. formal
        a_b = pipe.answer(cq)                    # B. conversational baseline
        rq, t_rp = rephrase_v2(server, cq)       # v2 rephrase
        a_c = pipe.answer(rq)                    # C. replace with v2 rephrase

        # D. multi-query: union(orig, rephrase) -> rerank(orig) -> generate(orig)
        t0 = time.perf_counter()
        cand = multiquery_candidates(idx, [cq, rq])
        top_d = rerank(cq, cand)
        t_search_d = time.perf_counter() - t0
        gen_d = server.complete(build_messages(cq, top_d), DEFAULT)

        rows.append({
            "id": cid, "gold_source": item["gold_source"],
            "formal_q": fq, "conversational_q": cq, "rephrased_q_v2": rq,
            "retrieval": {
                "formal": r5(a_f["top_chunks"], item),
                "conv_baseline": r5(a_b["top_chunks"], item),
                "conv_rephrase_v2": r5(a_c["top_chunks"], item),
                "conv_multiquery_v2": r5(top_d, item),
            },
            "answers": {
                "formal": a_f["answer"],
                "conv_baseline": a_b["answer"],
                "conv_rephrase_v2": a_c["answer"],
                "conv_multiquery_v2": gen_d["answer"],
            },
            "latency_sec": {
                "conv_baseline_total": round((a_b["timings"]["search_ms"] + a_b["timings"].get("gen_ms", 0)) / 1000, 2),
                "conv_rephrase_v2_total": round(t_rp + (a_c["timings"]["search_ms"] + a_c["timings"].get("gen_ms", 0)) / 1000, 2),
                "conv_multiquery_v2_total": round(t_rp + t_search_d + gen_d["gen_sec"], 2),
                "rephrase_step": round(t_rp, 2),
            },
        })
        print(f"  [{i}/32] {cid}")
    server.stop()

    arms = ("formal", "conv_baseline", "conv_rephrase_v2", "conv_multiquery_v2")

    def agg(arm, m):
        return round(sum(r["retrieval"][arm][m] for r in rows) / len(rows), 4)

    aggregate = {a: {m: agg(a, m) for m in ("src_recall@5", "quote_hit@5", "mrr")} for a in arms}

    def wl(arm):  # vs conv_baseline
        rec = sum(1 for r in rows if r["retrieval"]["conv_baseline"]["src_recall@5"] == 0
                  and r["retrieval"][arm]["src_recall@5"] == 1)
        lost = sum(1 for r in rows if r["retrieval"]["conv_baseline"]["src_recall@5"] == 1
                   and r["retrieval"][arm]["src_recall@5"] == 0)
        return {"recovered": rec, "lost": lost}
    aggregate["vs_baseline_src_recall"] = {a: wl(a) for a in ("conv_rephrase_v2", "conv_multiquery_v2")}

    out = {
        "meta": {
            "model": "Qwen3.5-2B.Q8_0 (concise)", "input": "conversational (haiku-generated)",
            "rephraser": "v2 (keyword-preserving, no-hallucinate)",
            "arms": {
                "formal": "book-phrased question (reference)",
                "conv_baseline": "conversational, no rephraser",
                "conv_rephrase_v2": "conversational -> v2 rephrase -> REPLACE",
                "conv_multiquery_v2": "union(conv, v2 rephrase) -> RRF -> rerank(conv)",
            },
            "prior_run_v1_rephrase_replace": {"src_recall@5": 0.6875, "quote_hit@5": 0.375, "mrr": 0.4411},
            "n": len(rows),
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RUNS / f"conversational_multiquery_v2_{ts}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"{'arm':22} {'src_recall@5':>12} {'quote_hit@5':>12} {'mrr':>7}")
    for a in arms:
        x = aggregate[a]
        print(f"{a:22} {x['src_recall@5']:>12} {x['quote_hit@5']:>12} {x['mrr']:>7}")
    print("\nvs conv_baseline (src_recall@5):")
    for a in ("conv_rephrase_v2", "conv_multiquery_v2"):
        w = aggregate["vs_baseline_src_recall"][a]
        print(f"  {a:22} recovered {w['recovered']}, lost {w['lost']}")
    print(f"\nsaved -> {path.name}")


if __name__ == "__main__":
    run()
