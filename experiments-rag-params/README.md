# experiments-rag-params

Measures the new `rag/` pipeline on the grounded eval set
(`experiment-docx-processing/out/eval/eval_set.json`) — **quality + per-step
latency** — so parameter choices are data-driven, not guessed.

## What gets measured

**Quality** (`score.py`), per eval item (gold_source + verbatim gold_quote):
- `source_recall@k` — was the right document in the top-k?
- `quote_hit@k` — did the top-k chunks actually contain the supporting passage?
  (the metric that matters — evidence retrieved, not just the right doc)
- `mrr` — reciprocal rank of the gold document

**Latency** — every stage timed separately: embed · dense · bm25 · rrf · rerank ·
generate. Guideline budget: **< 1 s** for everything up to (not incl.) generation.

## Scripts

| Script | GPU residents | Output |
|---|---|---|
| `run_grid.py` | bge-m3 + reranker | `runs/grid_<ts>.json` — retrieval sweep (rerank on/off, top-k, pool, rrf_k) |
| `dump_retrieval.py` | bge-m3 + reranker | `runs/retrieved.json` — frozen top-k per question |
| `bench_generation.py` | only the LLM | `runs/generation_{qwen2b,qwen4b}.json` — answers + gen latency/tps |

Generation is decoupled from retrieval on purpose: on an 8 GB card the 4B-Q8 model
won't co-reside with bge-m3 + reranker (~9 GB total), so we freeze retrieval to
disk, free the retrieval models, then load only the LLM.

## Run order

```powershell
$env:ALL_PROXY="socks5h://127.0.0.1:10808"   # first run only, to download models
$env:PYTHONUTF8="1"
uv run python -m rag.ingest          # docx txt -> chunks.jsonl
uv run python -m rag.index           # build FAISS + BM25 (downloads bge-m3)
uv run python experiments-rag-params/run_grid.py         # retrieval quality + latency
uv run python experiments-rag-params/dump_retrieval.py   # freeze best retrieval
uv run python experiments-rag-params/bench_generation.py # 2B vs 4B generation
```

## Headline question

Does the neural reranker justify its latency? `run_grid.py` runs the same hybrid
retrieval with the reranker OFF (legacy-style RRF-only) and ON — the quote_hit
delta is the value of the single biggest change over the client's pipeline.
