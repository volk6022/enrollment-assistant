"""Residual-gap experiment: does merging the tiny per-point chunks lift quote_hit?

Diagnosis: 31/32 gold quotes sit fully inside ONE chunk, so the quote_hit@5
ceiling (~0.69 even on formal) is NOT quote-splitting — it's FRAGMENTATION. The
corpus is 11293 chunks averaging ~446 chars (one legal point each), so the top-5
covers little text and the exact quote-chunk often isn't ranked in. Bigger chunks
(merge sibling points) widen the top-5 net and add context per candidate.

This sweeps target chunk sizes by greedily merging consecutive same-source chunks,
reindexes each (bge-m3 + BM25 in memory, no prod artifacts touched), and scores
retrieval on the FORMAL eval questions (the arm whose quote_hit ceiling we chase).

Usage: uv run python experiments-rag-params/bench_chunking.py
"""
from __future__ import annotations

import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rag.config import DEFAULT
from rag.embed import embed, embed_query
from rag.index import tokenize_ru
from rag.rerank import rerank
from rag.retrieve import rrf_fuse
from score import score_item

RUNS = Path(__file__).resolve().parent / "runs"
CHUNKS = Path(__file__).resolve().parent.parent / "rag" / "artifacts" / "chunks.jsonl"
EVAL = Path(__file__).resolve().parent.parent / "experiment-docx-processing" / "out" / "eval" / "eval_set.json"


def merge_chunks(chunks, target):
    """Greedily merge consecutive same-source chunks up to `target` chars.
    target=None -> no merge (current baseline)."""
    if target is None:
        return [dict(c) for c in chunks]
    out, cur = [], None
    for c in chunks:
        if cur and cur["source"] == c["source"] and len(cur["text"]) + len(c["text"]) + 1 <= target:
            cur["text"] += "\n" + c["text"]
            cur["_points"].append(c.get("point"))
        else:
            if cur:
                out.append(cur)
            cur = {"text": c["text"], "source": c["source"], "doc_title": c.get("doc_title"),
                   "section_path": c.get("section_path"), "_points": [c.get("point")]}
    if cur:
        out.append(cur)
    for m in out:
        m["point"] = next((p for p in m["_points"] if p), None)
    return out


class MemIndex:
    def __init__(self, chunks):
        import faiss
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        vecs = embed(texts, DEFAULT)
        self.index = faiss.IndexFlatIP(vecs.shape[1]); self.index.add(vecs)
        self.bm25 = BM25Okapi([tokenize_ru(t) for t in texts])

    def dense(self, qvec, top):
        s, idx = self.index.search(qvec.reshape(1, -1), top)
        return [(int(i), float(v)) for i, v in zip(idx[0], s[0]) if i >= 0]

    def sparse(self, q, top):
        sc = self.bm25.get_scores(tokenize_ru(q))
        order = np.argsort(sc)[::-1][:top]
        return [(int(i), float(sc[i])) for i in order]


def evaluate(mem, eval_items, cfg=DEFAULT):
    agg = {"src_recall@5": 0, "quote_hit@5": 0, "quote_hit@3": 0, "mrr": 0.0}
    for it in eval_items:
        q = it["question"]
        dense = mem.dense(embed_query(q, cfg), cfg.dense_top)
        sparse = mem.sparse(q, cfg.bm25_top)
        fused = rrf_fuse(dense, sparse, k=cfg.rrf_k)[: cfg.fused_top]
        cand = [dict(mem.chunks[i], rrf_score=s) for i, s in fused]
        top = rerank(q, cand, cfg)
        s = score_item(top, it)
        agg["src_recall@5"] += s["source_recall@5"]
        agg["quote_hit@5"] += s["quote_hit@5"]
        agg["quote_hit@3"] += s["quote_hit@3"]
        agg["mrr"] += s["mrr"]
    n = len(eval_items)
    return {k: round(v / n, 4) for k, v in agg.items()}


def run():
    chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_items = json.load(open(EVAL, encoding="utf-8"))

    results = {}
    for target in (None, 900, 1500, 2500):
        merged = merge_chunks(chunks, target)
        avg = round(sum(len(c["text"]) for c in merged) / len(merged))
        print(f"\n=== target={target}  ->  {len(merged)} chunks, avg {avg} chars ===")
        t0 = time.perf_counter()
        mem = MemIndex(merged)
        metrics = evaluate(mem, eval_items)
        metrics.update({"n_chunks": len(merged), "avg_chars": avg,
                        "build_eval_sec": round(time.perf_counter() - t0, 1)})
        results[str(target)] = metrics
        print(f"  src_recall@5={metrics['src_recall@5']}  quote_hit@3={metrics['quote_hit@3']}  "
              f"quote_hit@5={metrics['quote_hit@5']}  mrr={metrics['mrr']}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"chunking_sweep_{ts}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"{'target':>8} {'chunks':>8} {'avg_ch':>7} {'recall@5':>9} {'q_hit@3':>8} {'q_hit@5':>8} {'mrr':>7}")
    for k, m in results.items():
        print(f"{k:>8} {m['n_chunks']:>8} {m['avg_chars']:>7} {m['src_recall@5']:>9} "
              f"{m['quote_hit@3']:>8} {m['quote_hit@5']:>8} {m['mrr']:>7}")
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    run()
