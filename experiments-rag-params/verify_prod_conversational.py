"""Re-verify the SHIPPED conversational path (Pipeline with cfg.conversational=True)
reproduces the ~0.84 src_recall@5 measured in the bench, end-to-end through the
real production code (not the experiment harness).

Usage: uv run python experiments-rag-params/verify_prod_conversational.py
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


def run():
    conv = json.load(open(RUNS / "conversational_questions.json", encoding="utf-8"))
    gold = {it["id"]: it for it in json.load(open(EVAL, encoding="utf-8"))}
    cfg = replace(DEFAULT, conversational=True)

    idx = Indexes()
    server = LlamaServer(cfg); server.start(); time.sleep(2)
    pipe = Pipeline(idx, server=server, cfg=cfg)

    agg = {"src_recall@5": 0, "quote_hit@5": 0, "mrr": 0.0}
    lat, samples = [], []
    for c in conv:
        r = pipe.answer(c["conversational_q"] if "conversational_q" in c else c["conversational"])
        s = score_item(r["top_chunks"], gold[c["id"]])
        agg["src_recall@5"] += s["source_recall@5"]
        agg["quote_hit@5"] += s["quote_hit@5"]
        agg["mrr"] += s["mrr"]
        lat.append(r["timings"]["search_ms"] + r["timings"].get("gen_ms", 0))
        if len(samples) < 3:
            samples.append((c.get("conversational") or c.get("conversational_q"),
                            r["canonical_query"], r["answer"][:110]))
    server.stop()

    n = len(conv)
    lat.sort()
    print("=" * 60)
    print(f"SHIPPED prod path (cfg.conversational=True), n={n}")
    print(f"  src_recall@5 = {agg['src_recall@5']/n:.4f}")
    print(f"  quote_hit@5  = {agg['quote_hit@5']/n:.4f}")
    print(f"  mrr          = {agg['mrr']/n:.4f}")
    print(f"  latency      = {sum(lat)/n/1000:.2f}s avg / {lat[int(n*0.9)]/1000:.2f}s p90 (incl. rephrase+gen)")
    print("\n  samples (spoken -> canonical):")
    for spoken, canon, ans in samples:
        print(f"    Q: {spoken}")
        print(f"    ->{canon}")
        print(f"    A: {ans}\n")


if __name__ == "__main__":
    run()
