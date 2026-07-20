"""Generate baseline-raw answers for a NEW candidate local LLM, split into two
isolated processes (mirrors semantic_judge_run.py's cmd_dataset/cmd_judge split):
retrieval (embed+rerank models loaded) writes a cache and exits to free VRAM,
then generation (llama-server only, fresh process) reads that cache and answers.

Doing both in one process (as compare_run.py does) pushed VRAM to ~7.7/8.2GB on
this 3060 Ti and made even the single warmup generation call hang for 5+ minutes
(Windows CUDA shared-memory fallback once physical VRAM is oversubscribed) --
see the Qwopus-4B run that triggered this split.

Usage:
  uv run python experiments-rag-params/new_model_answers.py retrieve
  RAG_LLM_GGUF_OVERRIDE=<path\to.gguf> uv run python experiments-rag-params/new_model_answers.py generate <label>

Retrieval only needs to run once (baseline-raw index + questions don't depend on
which LLM will generate the answer) -- run `generate` once per candidate model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag.config import DEFAULT
from rag.generate import LlamaServer, build_messages
from rag.index import Indexes
from rag.pipeline import Pipeline

RUNS = Path(__file__).resolve().parent / "runs"
POOL = Path(__file__).resolve().parent / "eval_set_100.json"
RETRIEVAL_CACHE = RUNS / "new_model_retrieval_baseline-raw.json"


def cmd_retrieve():
    """Phase 1: retrieval only (baseline-raw index must already be built via
    rag.ingest + rag.index). Exits after writing -- exiting is what frees the
    embed/rerank models' VRAM for phase 2."""
    RUNS.mkdir(exist_ok=True)
    pool = json.load(open(POOL, encoding="utf-8"))
    idx = Indexes()
    pipe = Pipeline(idx, server=None, cfg=DEFAULT)
    print(f"[retrieve] baseline-raw, {len(pool)} questions, {len(idx.chunks)} chunks")
    rows = []
    for q in pool:
        top, _ = pipe.search(q["question"])
        rows.append({"id": q["id"], "question": q["question"], "topic": q.get("topic", ""),
                      "source_doc": q.get("source_doc", ""), "reference": q.get("reference", ""),
                      "chunks": top})
    RETRIEVAL_CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"[retrieve] wrote {len(rows)} rows -> {RETRIEVAL_CACHE}")


def cmd_generate(label: str):
    """Phase 2: fresh process, llama-server ONLY (RAG_LLM_GGUF_OVERRIDE picks the
    model) -- no embed/rerank models ever loaded here, full VRAM headroom. Writes
    compare_<label>_<ts>.json in the same shape compare_run.py produces (rows
    with an "answers" list), so it plugs straight into
    semantic_judge_run.py's `judge-add label=path`."""
    RUNS.mkdir(exist_ok=True)
    rows_in = json.load(open(RETRIEVAL_CACHE, encoding="utf-8"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = RUNS / f"llama_newmodel_{label}_{ts}.log"
    server = LlamaServer(DEFAULT)
    server.start(log_path=str(log_path))
    print(f"[generate] llm_gguf={DEFAULT.llm_gguf}")
    print(f"[generate] server log -> {log_path}")
    rows = []
    try:
        t0 = time.time()
        for i, r in enumerate(rows_in):
            messages = build_messages(r["question"], r["chunks"], DEFAULT)
            gen = server.complete(messages, DEFAULT)
            rows.append({"id": r["id"], "question": r["question"], "topic": r["topic"],
                         "source_doc": r["source_doc"], "reference": r["reference"],
                         "answers": [gen["answer"]]})
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(rows_in)}  ({time.time() - t0:.0f}s)")
    finally:
        server.stop()
    out = {"meta": {"label": label, "model": DEFAULT.llm_gguf, "conversational": False, "ts": ts},
           "rows": rows}
    fn = RUNS / f"compare_{label}_{ts}.json"
    fn.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {fn}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "retrieve":
        cmd_retrieve()
    elif cmd == "generate":
        if len(sys.argv) < 3:
            raise SystemExit("usage: new_model_answers.py generate <label>")
        cmd_generate(sys.argv[2])
    else:
        raise SystemExit("usage: new_model_answers.py retrieve | generate <label>")
