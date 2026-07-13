"""Full quality run on the SHIPPED prod config (merged ~900-char index + concise
2B generation). Runs the real Pipeline end-to-end over every test question and
dumps per-item results + retrieval metrics + generation + timing to JSON, so
answer quality and latency can be assessed offline.

Two separate outputs (same live llama server + prod index):
  runs/prod_eval_formal_<ts>.json         — 32 book-phrased eval questions, conversational=False
  runs/prod_eval_conversational_<ts>.json — 32 spoken paraphrases,         conversational=True

Each: meta (config), aggregate (metrics + latency avg/p90 + tokens), rows (full
per-question detail: answer, citations, all-k metrics, timing breakdown).

Usage: uv run python experiments-rag-params/prod_full_eval.py
"""
from __future__ import annotations

import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import replace
from rag.config import DEFAULT
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.generate import LlamaServer
from score import score_item

RUNS = Path(__file__).resolve().parent / "runs"
EVAL = Path(__file__).resolve().parent.parent / "experiment-docx-processing" / "out" / "eval" / "eval_set.json"
CONV = RUNS / "conversational_questions.json"


def _pct(vals, p):
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(len(s) * p))], 1)


def run_set(pipe, questions, gold, label, cfg):
    # Warm up CUDA (embedder + reranker + llama first-call kernels): the very first
    # query otherwise costs ~28s and pollutes the latency aggregate. Discarded.
    pipe.answer(questions[0]["q"])

    rows = []
    for q in questions:
        cid, text = q["id"], q["q"]
        r = pipe.answer(text)
        s = score_item(r["top_chunks"], gold[cid])
        t = r["timings"]
        gen = r.get("gen") or {}
        rows.append({
            "id": cid,
            "question": text,
            "canonical_query": r.get("canonical_query"),
            "gold_source": gold[cid]["gold_source"],
            "answer": r["answer"],
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
            "citations": r["citations"],
            "timings_ms": {k: round(v, 1) for k, v in t.items()},
            "gen_tokens": gen.get("completion_tokens"),
            "gen_tps": round(gen["tps"], 1) if gen.get("tps") else None,
        })

    n = len(rows)
    mkeys = ["source_recall@1", "source_recall@3", "source_recall@5",
             "quote_hit@1", "quote_hit@3", "quote_hit@5", "mrr"]
    aggregate = {k: round(sum(r["metrics"][k] for r in rows) / n, 4) for k in mkeys}
    search = [r["timings_ms"].get("search_ms", 0) for r in rows]
    genms = [r["timings_ms"].get("gen_ms", 0) for r in rows]
    total = [s + g for s, g in zip(search, genms)]
    toks = [r["gen_tokens"] or 0 for r in rows]
    aggregate["latency_sec"] = {
        "search_avg": round(sum(search) / n / 1000, 2), "search_p90": round(_pct(search, 0.9) / 1000, 2),
        "gen_avg": round(sum(genms) / n / 1000, 2), "gen_p90": round(_pct(genms, 0.9) / 1000, 2),
        "total_avg": round(sum(total) / n / 1000, 2), "total_p90": round(_pct(total, 0.9) / 1000, 2),
    }
    aggregate["avg_gen_tokens"] = round(sum(toks) / n, 1)

    out = {
        "meta": {
            "label": label,
            "config": {"model": "Qwen3.5-2B.Q8_0", "prompt": "concise", "max_tokens": cfg.max_tokens,
                       "temperature": cfg.temperature, "conversational": cfg.conversational,
                       "merge_chunk_chars": cfg.merge_chunk_chars, "n_chunks": len(pipe.idx.chunks),
                       "fused_top": cfg.fused_top, "final_top": cfg.final_top},
            "n": n,
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    return out


def run():
    formal = [{"id": it["id"], "q": it["question"]} for it in json.load(open(EVAL, encoding="utf-8"))]
    conv_raw = json.load(open(CONV, encoding="utf-8"))
    conv = [{"id": c["id"], "q": c["conversational"]} for c in conv_raw]
    gold = {it["id"]: it for it in json.load(open(EVAL, encoding="utf-8"))}

    idx = Indexes()
    server = LlamaServer(DEFAULT); server.start(); time.sleep(2)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # 1) formal questions, prod default (conversational=False)
    pipe_f = Pipeline(idx, server=server, cfg=DEFAULT)
    res_f = run_set(pipe_f, formal, gold, "formal / prod default", DEFAULT)
    (RUNS / f"prod_eval_formal_{ts}.json").write_text(
        json.dumps(res_f, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) conversational questions, prod conversational mode
    cfg_c = replace(DEFAULT, conversational=True)
    pipe_c = Pipeline(idx, server=server, cfg=cfg_c)
    res_c = run_set(pipe_c, conv, gold, "conversational / prod conversational=True", cfg_c)
    (RUNS / f"prod_eval_conversational_{ts}.json").write_text(
        json.dumps(res_c, ensure_ascii=False, indent=2), encoding="utf-8")

    server.stop()

    for res, fn in ((res_f, f"prod_eval_formal_{ts}.json"), (res_c, f"prod_eval_conversational_{ts}.json")):
        a = res["aggregate"]
        print(f"\n=== {res['meta']['label']}  (n={res['meta']['n']}) -> {fn}")
        print(f"  src_recall@1/3/5 = {a['source_recall@1']} / {a['source_recall@3']} / {a['source_recall@5']}")
        print(f"  quote_hit@1/3/5  = {a['quote_hit@1']} / {a['quote_hit@3']} / {a['quote_hit@5']}")
        print(f"  mrr              = {a['mrr']}")
        print(f"  latency total    = {a['latency_sec']['total_avg']}s avg / {a['latency_sec']['total_p90']}s p90"
              f"  (search {a['latency_sec']['search_avg']}s, gen {a['latency_sec']['gen_avg']}s)")
        print(f"  avg gen tokens   = {a['avg_gen_tokens']}")


if __name__ == "__main__":
    run()
