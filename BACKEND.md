# Stage 1 Backend — Optimized RAG Pipeline

Production-ready enrollment assistant backend (optimized from grid tuning).

## Performance

| Component | Time | Notes |
|---|---|---|
| Search (bge-m3 hybrid + reranker FP16) | ~354 ms | 5856 merged chunks; quote_hit@3=0.72, recall@5=0.84, MRR=0.77 |
| Generation (Qwen3.5-2B, no reasoning) | 0.79 s avg / 1.13 s p90 | concise prompt, ~70 tok, 0/32 truncated |
| **Total end-to-end** | **~1.15 s avg / ~1.5 s p90** | well under the 2–3s budget |
| Conversational mode (opt-in) | +~0.4 s rephrase | spoken input: src_recall@5 0.625 → 0.78 |

> **Re-verified 2026-07-13 (Opus):** the earlier "max_tokens=200 = 1.30s grid
> winner" was a measurement artifact — 200 tokens truncated **13/32** answers
> mid-sentence, and the latency was low only *because* answers were cut (tps is
> flat ~120 across all caps). Fix: a "2–4 sentences" instruction in the system
> prompt drops natural length to ~70 tok, so answers now finish cleanly (0/32
> truncated) **and** faster (0.79s). See `bench_concise_2b.py`.

## Architecture

```
query
  ↓
[embed + dense search (FAISS)]  ∥  [BM25 sparse search]
  ↓
[RRF fusion, k=60]
  ↓
[bge-reranker-v2-m3 cross-encoder, FP16]  (rerank top-50 → top-5)
  ↓
[LlamaServer: Qwen3.5-2B GGUF]  (prefilled closed <think> to disable reasoning)
  ↓
answer + citations + latencies
```

## Quick Start

**Option 1: Interactive CLI**

```bash
uv run python demo.py "Какие анализы нужны перед медкомиссией?"
```

**Option 2: CLI loop**

```bash
uv run python backend.py --mode cli
```

Then type questions interactively (type `quit` to exit).

**Option 3: HTTP server** (requires Flask)

```bash
uv pip install flask
uv run python backend.py --mode server --listen 127.0.0.1:8000
```

Query from another terminal:
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Какие анализы нужны?"}'
```

## Configuration

Edit `rag/config.py` (RagConfig class) to tune:

- **Retrieval**: `bm25_top`, `dense_top`, `rrf_k`, `fused_top`, `final_top`
- **Chunking**: `merge_chunk_chars` (900 = merge sibling points; reindex to apply)
- **Reranking**: `rerank_fp16`, `rerank_max_length`, `rerank_batch_size`
- **Generation**: `max_tokens`, `temperature`, `llm_gguf` (switch 2B ↔ 4B)
- **Conversational**: `conversational=True` enables spoken-input mode (rephrase →
  multi-query union → rerank on the canonical query). Default `False`. See RESULTS.md.

Current defaults are the grid/sweep winners (fastest + best quality). Changing
`merge_chunk_chars` requires a reindex: `uv run python -m rag.ingest && uv run python -m rag.index`.

## Grid Results

See `experiments-rag-params/runs/`:
- `grid_20260712_020602.json` — retrieval parameter sweep (6 configs); winner `pool50`, src_recall@5=0.88 / MRR=0.71 @ 354ms
- `grid_2b_generation_20260712_165138.json` — 2B temp×max_tokens sweep (9 configs). ⚠ its "max200" pick truncated 13/32 answers — superseded by the concise-prompt fix below
- `gen_08b_*.json` — 0.8B (Q4) feasibility bench (`bench_generation_08b.py`) — see Model choice
- `<concise bench, stdout>` — `bench_concise_2b.py`: concise prompt @max200 → 0.79s avg, 0/32 truncated
- `with_rephraser_20260712_202124.json` — rephraser test (rejected: +47% latency, degraded quality)

## Model choice: 2B (Q8) is the right default; 0.8B (Q4) not worth it

| model | quant | gen avg | gen p90 | quality |
|---|---|---|---|---|
| **Qwen3.5-2B** ✅ | Q8_0 | 0.79 s | 1.13 s | grounded, correct |
| Qwen3.5-0.8B | Q4_K_M | 0.82 s | 1.50 s | misreads questions, garbled tokens |
| Qwen3.5-4B | Q8_0 | ~3.3 s | ~5.1 s | most faithful, too slow |

The 0.8B is only ~2× the raw tps of the 2B but the whole pipeline is **already
1.15 s** — there is no latency problem to solve, and 0.8B exists **only in Q4**
(vs Q8 for 2B/4B), a double quality hit. On the eval set it misread a question
(listed *categories of examinees* instead of the *doctors* asked about) and
emitted garbled tokens. For grounded legal/medical answers read aloud, that's
disqualifying. Keep 0.8B only as an emergency VRAM fallback if STT+TTS must be
co-resident on an 8 GB card — and gate it behind an LLM-judge pass first.

## Known Limitations

1. **quote_hit@5 caps at 0.69** while src_recall@5 is 0.88 — we get the right doc but not always the exact passage. Can improve via chunk-size tuning (requires reindex).

2. **Generation answers are grounded** but not auto-scored yet. To measure correctness, run an LLM-judge pass over generation results.

3. **ФЗ-152 is an empty stub** in the corpus (3.6 KB) — not indexed.

4. **VRAM:** Fits on client's 12 GB 3060: bge-m3 ~2.3 GB + reranker FP16 ~1.1 GB + Qwen2B ~2.2 GB ≈ 5.6 GB.

## Next Steps (Stage 1)

- [ ] Finalize LLM choice (2B vs 4B) with client
- [ ] STT integration (stage 3) — pick open-source model
- [ ] TTS integration (stage 4) — pick open-source model
- [ ] (Optional) Chunk-size tuning → reindex → re-benchmark

Then **Stage 5 (Streaming)** — parallelize STT/LLM/TTS for true low-latency voice.
