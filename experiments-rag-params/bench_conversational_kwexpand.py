"""Conversational input — query EXPANSION: rephraser emits a canonical question
PLUS a set of keywords/phrases, to cover lexical diversity and feed retrieval.

Progression tested (all vs the same gold, one combined JSON):
  A. formal               — book-phrased question (upper-bound reference)
  B. conv_baseline        — conversational, no rephraser
  C. conv_rephrase_v2     — v2 rephrase, REPLACE (prev-best single-query)
  D. conv_multiquery_v3   — union(conv, canonical) -> rerank on CLEAN canonical
  E. conv_kwexpand_v3     — union(conv, canonical, *keywords) -> rerank on CLEAN

D isolates the "rerank with the clean query" fix (prev multi-query reranked with
the messy query and underperformed). E adds the keyword/phrase expansion on top,
so E − D = the keyword contribution.

The rephraser (one LLM call) returns:
    ВОПРОС: <canonical question, keywords preserved verbatim>
    КЛЮЧЕВЫЕ: term1; phrase2; synonym3; ...
keywords = exact terms from the question + official synonyms + likely document
wordings (different phrasings of the same thing) — this is what covers diversity.

Usage: uv run python experiments-rag-params/bench_conversational_kwexpand.py
"""
from __future__ import annotations

import json, re, time
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

EXPAND_SYSTEM = (
    "Ты помогаешь искать ответ в базе нормативных документов вуза (ДВЮИ МВД) по "
    "разговорному вопросу абитуриента. Верни РОВНО две строки:\n"
    "ВОПРОС: <один чёткий вопрос официальным языком; ДОСЛОВНО сохрани все термины и "
    "аббревиатуры из исходного вопроса — паспорт, СНИЛС, ЕГЭ, СВО, адъюнктура и т.п.>\n"
    "КЛЮЧЕВЫЕ: <6-10 ключевых слов и фраз через точку с запятой, которыми ответ может "
    "быть сформулирован в документах: точные термины из вопроса + официальные синонимы + "
    "родственные юридические формулировки. Разные способы сказать то же самое.>\n"
    "НЕ придумывай факты и определения. Незнакомый термин оставь как есть."
)
FEWSHOT = [
    {"role": "user", "content": "А паспорт обязательно надо подавать?"},
    {"role": "assistant", "content": "ВОПРОС: Обязательно ли предоставлять паспорт при подаче документов на поступление?\nКЛЮЧЕВЫЕ: паспорт; документ, удостоверяющий личность; перечень документов; подача документов; поступление; приём"},
    {"role": "user", "content": "Вот, отец в СВО участвует, есть какая-нибудь квота для нас?"},
    {"role": "assistant", "content": "ВОПРОС: Какая квота при поступлении предусмотрена для детей участников СВО?\nКЛЮЧЕВЫЕ: СВО; специальная военная операция; отдельная квота; особая квота; дети участников СВО; преимущественное право; льготы при поступлении"},
    {"role": "user", "content": "По физре сколько баллов надо набрать чтоб сдать?"},
    {"role": "assistant", "content": "ВОПРОС: Сколько баллов нужно набрать по физической подготовке для сдачи вступительного испытания?\nКЛЮЧЕВЫЕ: физическая подготовка; физподготовка; минимальный балл; вступительное испытание; контрольные упражнения; нормативы; проходной балл"},
]


def parse_expand(text: str):
    q, kws = "", []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"(?i)^ВОПРОС\s*[:\-]\s*(.+)", line)
        if m:
            q = m.group(1).strip()
        m = re.match(r"(?i)^КЛЮЧЕВЫЕ\s*[:\-]\s*(.+)", line)
        if m:
            kws = [k.strip(" .-•") for k in re.split(r"[;,\n]", m.group(1)) if k.strip(" .-•")]
    if not q:  # fallback: first sentence ending with ?
        i = text.find("?")
        q = text[: i + 1].strip() if i != -1 else text.strip().split("\n")[0]
    return q, kws[:8]


def rephrase_expand(server, q: str):
    msgs = [{"role": "system", "content": EXPAND_SYSTEM}, *FEWSHOT, {"role": "user", "content": q}]
    r = server.complete(msgs, DEFAULT)
    cq, kws = parse_expand(r["answer"])
    return cq, kws, r["gen_sec"]


def union_candidates(idx, queries, cfg=DEFAULT):
    ranked = []
    for q in queries:
        if not q or not q.strip():
            continue
        ranked.append(idx.dense_search(embed_query(q, cfg), cfg.dense_top))
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

        a_f = pipe.answer(fq)                       # A. formal
        a_b = pipe.answer(cq)                       # B. baseline
        canon, kws, t_rp = rephrase_expand(server, cq)
        a_c = pipe.answer(canon)                    # C. v2-style replace (canonical question)

        # D. union(conv, canonical) -> rerank on CLEAN canonical -> generate(conv)
        t0 = time.perf_counter()
        cand_d = union_candidates(idx, [cq, canon])
        top_d = rerank(canon, cand_d)
        t_d = time.perf_counter() - t0
        gen_d = server.complete(build_messages(cq, top_d), DEFAULT)

        # E. union(conv, canonical, *keywords) -> rerank on CLEAN canonical
        t0 = time.perf_counter()
        cand_e = union_candidates(idx, [cq, canon, *kws])
        top_e = rerank(canon, cand_e)
        t_e = time.perf_counter() - t0
        gen_e = server.complete(build_messages(cq, top_e), DEFAULT)

        rows.append({
            "id": cid, "gold_source": item["gold_source"],
            "conversational_q": cq, "canonical_q": canon, "keywords": kws,
            "retrieval": {
                "formal": r5(a_f["top_chunks"], item),
                "conv_baseline": r5(a_b["top_chunks"], item),
                "conv_rephrase_v2": r5(a_c["top_chunks"], item),
                "conv_multiquery_v3": r5(top_d, item),
                "conv_kwexpand_v3": r5(top_e, item),
            },
            "answers": {
                "conv_baseline": a_b["answer"],
                "conv_kwexpand_v3": gen_e["answer"],
            },
            "latency_sec": {
                "rephrase_step": round(t_rp, 2),
                "conv_multiquery_v3_total": round(t_rp + t_d + gen_d["gen_sec"], 2),
                "conv_kwexpand_v3_total": round(t_rp + t_e + gen_e["gen_sec"], 2),
            },
        })
        print(f"  [{i}/32] {cid}  (kw={len(kws)})")
    server.stop()

    arms = ("formal", "conv_baseline", "conv_rephrase_v2", "conv_multiquery_v3", "conv_kwexpand_v3")

    def agg(a, m):
        return round(sum(r["retrieval"][a][m] for r in rows) / len(rows), 4)
    aggregate = {a: {m: agg(a, m) for m in ("src_recall@5", "quote_hit@5", "mrr")} for a in arms}

    def wl(arm):
        rec = sum(1 for r in rows if r["retrieval"]["conv_baseline"]["src_recall@5"] == 0
                  and r["retrieval"][arm]["src_recall@5"] == 1)
        lost = sum(1 for r in rows if r["retrieval"]["conv_baseline"]["src_recall@5"] == 1
                   and r["retrieval"][arm]["src_recall@5"] == 0)
        return {"recovered": rec, "lost": lost}
    aggregate["vs_baseline_src_recall"] = {a: wl(a) for a in arms if a not in ("formal", "conv_baseline")}
    aggregate["avg_keywords"] = round(sum(len(r["keywords"]) for r in rows) / len(rows), 1)
    aggregate["latency_avg_sec"] = {
        "rephrase_step": round(sum(r["latency_sec"]["rephrase_step"] for r in rows) / len(rows), 2),
        "conv_multiquery_v3": round(sum(r["latency_sec"]["conv_multiquery_v3_total"] for r in rows) / len(rows), 2),
        "conv_kwexpand_v3": round(sum(r["latency_sec"]["conv_kwexpand_v3_total"] for r in rows) / len(rows), 2),
    }

    out = {"meta": {"model": "Qwen3.5-2B.Q8_0 (concise)", "input": "conversational (haiku-generated)",
                    "rephraser": "v3 query-expansion (canonical question + keyword set)",
                    "rerank_query": "clean canonical (D, E)", "n": len(rows)},
           "aggregate": aggregate, "rows": rows}
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RUNS / f"conversational_kwexpand_v3_{ts}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'arm':22} {'src_recall@5':>12} {'quote_hit@5':>12} {'mrr':>7}")
    for a in arms:
        x = aggregate[a]
        print(f"{a:22} {x['src_recall@5']:>12} {x['quote_hit@5']:>12} {x['mrr']:>7}")
    print(f"\navg keywords/query: {aggregate['avg_keywords']}")
    print("vs conv_baseline (src_recall@5):")
    for a in ("conv_rephrase_v2", "conv_multiquery_v3", "conv_kwexpand_v3"):
        w = aggregate["vs_baseline_src_recall"][a]
        print(f"  {a:22} recovered {w['recovered']}, lost {w['lost']}")
    print(f"\nlatency avg: multiquery {aggregate['latency_avg_sec']['conv_multiquery_v3']}s, "
          f"kwexpand {aggregate['latency_avg_sec']['conv_kwexpand_v3']}s "
          f"(rephrase step {aggregate['latency_avg_sec']['rephrase_step']}s)")
    print(f"saved -> {path.name}")


if __name__ == "__main__":
    run()
