"""Verify the REBUILT prod index (merged ~900-char chunks) reproduces the
chunk-sweep numbers through the real Indexes + Pipeline.search (retrieval only,
no LLM). Expected ~ quote_hit@5 0.719, MRR 0.773 on the formal eval questions.

Usage: uv run python experiments-rag-params/verify_prod_chunking.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import DEFAULT
from rag.index import Indexes
from rag.pipeline import Pipeline
from score import score_item

EVAL = Path(__file__).resolve().parent.parent / "experiment-docx-processing" / "out" / "eval" / "eval_set.json"


def run():
    idx = Indexes()
    pipe = Pipeline(idx, server=None, cfg=DEFAULT)  # no LLM: retrieval+rerank only
    items = json.load(open(EVAL, encoding="utf-8"))

    agg = {"src_recall@5": 0, "quote_hit@3": 0, "quote_hit@5": 0, "mrr": 0.0}
    for it in items:
        top, _ = pipe.search(it["question"])
        s = score_item(top, it)
        agg["src_recall@5"] += s["source_recall@5"]
        agg["quote_hit@3"] += s["quote_hit@3"]
        agg["quote_hit@5"] += s["quote_hit@5"]
        agg["mrr"] += s["mrr"]
    n = len(items)
    print(f"REBUILT prod index: {len(idx.chunks)} chunks")
    print(f"  src_recall@5 = {agg['src_recall@5']/n:.4f}")
    print(f"  quote_hit@3  = {agg['quote_hit@3']/n:.4f}")
    print(f"  quote_hit@5  = {agg['quote_hit@5']/n:.4f}")
    print(f"  mrr          = {agg['mrr']/n:.4f}")
    print("  (sweep target=900 expected: recall 0.844, q_hit@3/@5 0.719, mrr 0.773)")


if __name__ == "__main__":
    run()
