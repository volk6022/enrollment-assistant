"""Freeze retrieval results for the generation benchmark.

Runs one retrieval config (embedder + reranker resident on GPU) over the eval set
and writes the top-k chunks per question to runs/retrieved.json. The generation
benchmark then loads only the LLM onto the GPU — avoiding the 8 GB OOM you'd hit
trying to co-locate bge-m3 + reranker + a 4B model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import DEFAULT, EVAL_SET
from rag.index import Indexes
from rag.pipeline import Pipeline

RUNS = Path(__file__).resolve().parent / "runs"


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    eval_items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    idx = Indexes()
    pipe = Pipeline(idx, server=None, cfg=DEFAULT)

    out = []
    for item in eval_items:
        top, t = pipe.search(item["question"])
        out.append({
            "id": item["id"],
            "question": item["question"],
            "answer_gold": item["answer"],
            "gold_source": item["gold_source"],
            "gold_quote": item.get("gold_quote", ""),
            "topic": item["topic"],
            "search_ms": round(t["search_ms"], 1),
            "top_chunks": [
                {"source": c["source"], "point": c.get("point"), "text": c["text"],
                 "rerank_score": c.get("rerank_score")}
                for c in top
            ],
        })
    dst = RUNS / "retrieved.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dumped {len(out)} retrievals -> {dst}")


if __name__ == "__main__":
    main()
