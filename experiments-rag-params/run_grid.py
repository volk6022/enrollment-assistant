"""Retrieval parameter grid: quality + per-step latency against the eval set.

Sweeps retrieval/rerank params on a fixed index (chunking/embedding held constant
here — vary those via a rebuild + separate run). Headline comparison: reranker
ON vs OFF, which quantifies the biggest fix over the legacy no-reranker pipeline.

Usage:
  uv run python experiments-rag-params/run_grid.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import DEFAULT, EVAL_SET, RagConfig
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.retrieve import rrf_fuse
from rag.embed import embed_query

from score import aggregate, score_item

RUNS = Path(__file__).resolve().parent / "runs"


def make_configs() -> list[tuple[str, RagConfig, bool]]:
    """(name, cfg, use_reranker)."""
    base = dataclasses.asdict(DEFAULT)
    out = []

    def cfg(**over):
        d = dict(base)
        d.update(over)
        return RagConfig(**d)

    # reranker OFF (legacy-style: fused RRF only) vs ON, at final_top 3 and 5
    out.append(("hybrid_norerank_top5", cfg(final_top=5), False))
    out.append(("hybrid_norerank_top3", cfg(final_top=3), False))
    out.append(("hybrid_rerank_top5", cfg(final_top=5, fused_top=30), True))
    out.append(("hybrid_rerank_top3", cfg(final_top=3, fused_top=30), True))
    # deeper candidate pool before rerank
    out.append(("hybrid_rerank_top5_pool50", cfg(final_top=5, fused_top=50), True))
    # rrf smoothing constant
    out.append(("hybrid_rerank_top5_rrf10", cfg(final_top=5, fused_top=30, rrf_k=10), True))
    return out


def run_config(idx: Indexes, eval_items: list[dict], cfg: RagConfig, use_reranker: bool) -> dict:
    from rag.rerank import rerank

    pipe = Pipeline(idx, server=None, cfg=cfg)
    rows, timings = [], []
    for item in eval_items:
        q = item["question"]
        candidates, t = pipe.retrieve_timed(q)
        if use_reranker:
            t0 = time.perf_counter()
            top = rerank(q, candidates, cfg)
            t["rerank_ms"] = (time.perf_counter() - t0) * 1000
        else:
            top = candidates[: cfg.final_top]
            t["rerank_ms"] = 0.0
        t["search_ms"] = t["embed_ms"] + t["dense_ms"] + t["bm25_ms"] + t["rrf_ms"] + t["rerank_ms"]
        rows.append(score_item(top, item))
        timings.append(t)

    agg = aggregate(rows)
    lat = {k: round(sum(x[k] for x in timings) / len(timings), 2) for k in timings[0]}
    return {"quality": agg, "latency_ms_avg": lat}


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    eval_items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    print(f"loaded {len(eval_items)} eval items")
    print("loading indexes + warming embedder/reranker...")
    idx = Indexes()
    # warm caches so the first config isn't penalized by lazy model load
    from rag.rerank import get_reranker
    _ = embed_query("тест", DEFAULT)
    _ = get_reranker(DEFAULT)

    results = {}
    for name, cfg, use_rr in make_configs():
        print(f"\n=== {name} (reranker={use_rr}) ===")
        r = run_config(idx, eval_items, cfg, use_rr)
        results[name] = r
        q = r["quality"]
        lat = r["latency_ms_avg"]
        print(f"  quote_hit@1/3/5: {q['quote_hit@1']:.2f} / {q['quote_hit@3']:.2f} / {q['quote_hit@5']:.2f}"
              f" | src_recall@5: {q['source_recall@5']:.2f} | mrr: {q['mrr']:.3f}")
        print(f"  search {lat['search_ms']:.1f}ms (embed {lat['embed_ms']:.1f} + dense {lat['dense_ms']:.1f}"
              f" + bm25 {lat['bm25_ms']:.1f} + rrf {lat['rrf_ms']:.1f} + rerank {lat['rerank_ms']:.1f})")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"grid_{ts}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")

    # headline
    print("\n" + "=" * 70)
    print(f"{'config':32} {'qhit@3':>7} {'qhit@5':>7} {'rcl@5':>7} {'search_ms':>10}")
    for name, r in results.items():
        q, lat = r["quality"], r["latency_ms_avg"]
        print(f"{name:32} {q['quote_hit@3']:>7.2f} {q['quote_hit@5']:>7.2f} {q['source_recall@5']:>7.2f} {lat['search_ms']:>10.1f}")


if __name__ == "__main__":
    main()
