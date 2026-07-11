"""Retrieval-quality metrics against the grounded eval set.

Each eval item has a gold_source (which .docx holds the answer) and a gold_quote
(a verbatim supporting passage). After the pipeline returns top-k reranked
chunks we measure:

  source_recall@k — was gold_source among the top-k chunk sources?
  quote_hit@k     — did the top-k chunks actually contain the supporting passage?
                    (the stronger metric — we surfaced the real evidence)
  mrr             — reciprocal rank of the first gold_source chunk

quote_hit is the metric that matters most: it tests whether the answer's evidence
was retrieved, not just the right document.
"""
from __future__ import annotations

import re


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def quote_in(chunks_text: str, gold_quote: str) -> bool:
    hay = norm(chunks_text)
    frags = [f for f in (gold_quote or "").split("…") if f.strip()]
    if not frags:
        return False
    return all(norm(f) in hay for f in frags)


def score_item(top_chunks: list[dict], item: dict, ks=(1, 3, 5)) -> dict:
    gold_source = item["gold_source"]
    gold_quote = item.get("gold_quote", "")
    sources = [c["source"] for c in top_chunks]

    res = {}
    for k in ks:
        topk = top_chunks[:k]
        res[f"source_recall@{k}"] = int(gold_source in [c["source"] for c in topk])
        res[f"quote_hit@{k}"] = int(quote_in("\n".join(c["text"] for c in topk), gold_quote))

    rank = next((i + 1 for i, s in enumerate(sources) if s == gold_source), 0)
    res["mrr"] = 1.0 / rank if rank else 0.0
    res["found_rank"] = rank
    return res


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "found_rank"]
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
