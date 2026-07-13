"""Salvaged keyword expansion: discriminative keywords (guided prompt) fed as
ONE BM25-disjunction query (single RRF vote, sparse-only), not 7 equal-weight
dense+sparse queries.

Why v3 keyword expansion hurt (prev run): 7 generic keywords each cast a wide
dense+sparse net with equal RRF weight, out-voting the 2 focused queries and
flooding the fused_top pool with topically-adjacent distractors. Two fixes here:
  1. GUIDED prompt (grounded in real НПА vocabulary by a sub-agent) -> keywords
     are discriminative stock-phrases, not generic filler.
  2. keywords -> ONE bm25 disjunction (single sparse vote), so they can only
     ADD lexical coverage for sparse retrieval, never out-vote the focused queries.

Same rephrase output feeds both v3 and v4 arms, so (v4 - v3) isolates exactly the
keyword-bm25 contribution.

Arms (scored vs same gold; one combined JSON):
  A. formal              — reference upper bound
  B. conv_baseline       — conversational, no rephraser
  D. conv_multiquery_v3  — union(conv, canonical) dense+sparse -> rerank(canonical)
  E. conv_kwexpand_v4    — D + keywords as ONE extra bm25 query -> rerank(canonical)

Usage: uv run python experiments-rag-params/bench_conversational_kwexpand_v4.py
"""
from __future__ import annotations

import json, re, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# guided prompt — rules distilled from runs/keyword_guide.md (НПА-grounded)
EXPAND_SYSTEM = (
    "Ты помогаешь искать ответ в базе нормативных документов (НПА) вуза МВД (ДВЮИ МВД) "
    "по разговорному вопросу абитуриента. Верни РОВНО две строки:\n"
    "ВОПРОС: <один чёткий вопрос официальным языком; ДОСЛОВНО сохрани термины и "
    "аббревиатуры из вопроса — паспорт, СНИЛС, ЕГЭ, СВО, адъюнктура>\n"
    "КЛЮЧЕВЫЕ: <6-10 ДИСКРИМИНАТИВНЫХ терминов/оборотов через «;»>\n\n"
    "Правила для КЛЮЧЕВЫХ (критично для качества поиска):\n"
    "1. Специфичное вместо общего. ЗАПРЕЩЕНЫ общие слова, матчащие пол-корпуса: "
    "«поступление», «вступительные испытания», «приём», «документы», «упражнения», "
    "«нормативы», «баллы», «льготы», «выпускник», «предмет», «возможность».\n"
    "2. Бери устойчивые обороты из НПА: «медицинское освидетельствование», «категория "
    "годности», «расписание болезней», «дополнительное вступительное испытание», "
    "«контрольное упражнение», «минимальное количество баллов ЕГЭ», «сумма конкурсных "
    "баллов», «особое право при приёме», «первоочередной порядок зачисления».\n"
    "3. Для физподготовки — точные названия: «подтягивание на перекладине», «бег 100 м», "
    "«бег (кросс) 1000 м», «силовое комплексное упражнение».\n"
    "4. НЕ добавляй ключевики из соседних тем (в вопрос про льготы не тяни «физподготовка»).\n"
    "5. Сохраняй аббревиатуры дословно (ЕГЭ, СВО, СНИЛС).\n"
    "НЕ придумывай факты. Незнакомый термин оставь как есть."
)
FEWSHOT = [
    {"role": "user", "content": "А паспорт обязательно надо подавать?"},
    {"role": "assistant", "content": "ВОПРОС: Обязательно ли предоставлять паспорт при подаче документов на поступление?\nКЛЮЧЕВЫЕ: паспорт; документ, удостоверяющий личность; перечень документов, представляемых кандидатом; личное дело кандидата"},
    {"role": "user", "content": "Вот, отец в СВО участвует, есть какая-нибудь квота для нас?"},
    {"role": "assistant", "content": "ВОПРОС: Какая квота при поступлении предусмотрена для детей участников СВО?\nКЛЮЧЕВЫЕ: СВО; специальная военная операция; особое право при приёме; первоочередной порядок зачисления; дети военнослужащих; отдельная квота приёма"},
    {"role": "user", "content": "По физре сколько баллов надо набрать чтоб сдать?"},
    {"role": "assistant", "content": "ВОПРОС: Сколько баллов нужно набрать по физической подготовке для сдачи вступительного испытания?\nКЛЮЧЕВЫЕ: физическая подготовка; контрольное упражнение; подтягивание на перекладине; бег 100 м; силовое комплексное упражнение; минимальное количество баллов"},
    {"role": "user", "content": "Адъюнктура — когда туда документы подавать?"},
    {"role": "assistant", "content": "ВОПРОС: В какие сроки подаются документы для поступления в адъюнктуру?\nКЛЮЧЕВЫЕ: адъюнктура; поступление в адъюнктуру; срок приёма документов; приём в адъюнктуру"},
]


def first_question(text: str) -> str:
    t = text.strip(); i = t.find("?")
    return (t[: i + 1]).strip() if i != -1 else t


def parse_expand(text: str):
    q, kws = "", []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"(?i)^ВОПРОС\s*[:\-]\s*(.+)", line)
        if m: q = m.group(1).strip()
        m = re.match(r"(?i)^КЛЮЧЕВЫЕ\s*[:\-]\s*(.+)", line)
        if m: kws = [k.strip(" .-•") for k in re.split(r"[;,\n]", m.group(1)) if k.strip(" .-•")]
    if not q: q = first_question(text)
    return q, kws[:10]


def rephrase_expand(server, q: str):
    msgs = [{"role": "system", "content": EXPAND_SYSTEM}, *FEWSHOT, {"role": "user", "content": q}]
    r = server.complete(msgs, DEFAULT)
    cq, kws = parse_expand(r["answer"])
    return cq, kws, r["gen_sec"]


def union(idx, dense_qs, sparse_qs, cfg=DEFAULT):
    ranked = []
    for q in dense_qs:
        if q and q.strip(): ranked.append(idx.dense_search(embed_query(q, cfg), cfg.dense_top))
    for q in sparse_qs:
        if q and q.strip(): ranked.append(idx.bm25_search(q, cfg.bm25_top))
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

        a_f = pipe.answer(fq)          # A
        a_b = pipe.answer(cq)          # B
        canon, kws, t_rp = rephrase_expand(server, cq)
        kw_query = "; ".join(kws)

        # D: union(conv, canonical) dense+sparse -> rerank(canonical)
        cand_d = union(idx, [cq, canon], [cq, canon])
        top_d = rerank(canon, cand_d)
        gen_d = server.complete(build_messages(cq, top_d), DEFAULT)

        # E: D + keywords as ONE extra bm25 disjunction -> rerank(canonical)
        cand_e = union(idx, [cq, canon], [cq, canon, kw_query])
        top_e = rerank(canon, cand_e)
        gen_e = server.complete(build_messages(cq, top_e), DEFAULT)

        rows.append({
            "id": cid, "gold_source": item["gold_source"],
            "conversational_q": cq, "canonical_q": canon, "keywords": kws,
            "retrieval": {
                "formal": r5(a_f["top_chunks"], item),
                "conv_baseline": r5(a_b["top_chunks"], item),
                "conv_multiquery_v3": r5(top_d, item),
                "conv_kwexpand_v4": r5(top_e, item),
            },
            "answers": {"conv_baseline": a_b["answer"], "conv_kwexpand_v4": gen_e["answer"]},
            "latency_sec": {"rephrase_step": round(t_rp, 2)},
        })
        print(f"  [{i}/32] {cid}  (kw={len(kws)})")
    server.stop()

    arms = ("formal", "conv_baseline", "conv_multiquery_v3", "conv_kwexpand_v4")

    def agg(a, m): return round(sum(r["retrieval"][a][m] for r in rows) / len(rows), 4)
    aggregate = {a: {m: agg(a, m) for m in ("src_recall@5", "quote_hit@5", "mrr")} for a in arms}

    def wl(arm, ref):
        rec = sum(1 for r in rows if r["retrieval"][ref]["src_recall@5"] == 0 and r["retrieval"][arm]["src_recall@5"] == 1)
        lost = sum(1 for r in rows if r["retrieval"][ref]["src_recall@5"] == 1 and r["retrieval"][arm]["src_recall@5"] == 0)
        return {"recovered": rec, "lost": lost}
    aggregate["vs_baseline"] = {a: wl(a, "conv_baseline") for a in ("conv_multiquery_v3", "conv_kwexpand_v4")}
    aggregate["kwexpand_vs_multiquery"] = wl("conv_kwexpand_v4", "conv_multiquery_v3")
    aggregate["avg_keywords"] = round(sum(len(r["keywords"]) for r in rows) / len(rows), 1)

    out = {"meta": {"model": "Qwen3.5-2B.Q8_0 (concise)", "input": "conversational",
                    "rephraser": "v4 guided query-expansion (НПА-grounded discriminative keywords)",
                    "keyword_injection": "single bm25 disjunction (sparse-only, 1 RRF vote)",
                    "prior_naive_kwexpand_v3": {"src_recall@5": 0.6875, "quote_hit@5": 0.4375, "mrr": 0.4682},
                    "n": len(rows)},
           "aggregate": aggregate, "rows": rows}
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RUNS / f"conversational_kwexpand_v4_{ts}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"{'arm':22} {'src_recall@5':>12} {'quote_hit@5':>12} {'mrr':>7}")
    for a in arms:
        x = aggregate[a]
        print(f"{a:22} {x['src_recall@5']:>12} {x['quote_hit@5']:>12} {x['mrr']:>7}")
    print(f"\navg keywords/query: {aggregate['avg_keywords']}")
    print(f"kwexpand_v4 vs multiquery_v3: recovered {aggregate['kwexpand_vs_multiquery']['recovered']}, "
          f"lost {aggregate['kwexpand_vs_multiquery']['lost']}")
    print(f"saved -> {path.name}")


if __name__ == "__main__":
    run()
