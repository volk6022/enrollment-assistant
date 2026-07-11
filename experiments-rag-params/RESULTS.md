# Stage-1 RAG results (2026-07-12, RTX 3060 Ti 8 GB)

New pipeline (bge-m3 hybrid + bge-reranker-v2-m3 + local Qwen3.5) vs the legacy
hand-tuned scoring RAG. Measured on 32 grounded eval questions
(`experiment-docx-processing/out/eval/eval_set.json`), corpus = 31 NPA docs →
11,293 chunks.

## Retrieval — the reranker is the win

| config | quote_hit@3 | quote_hit@5 | src_recall@5 | MRR | search latency |
|---|---|---|---|---|---|
| hybrid, **no reranker** (legacy-style) | 0.41 | 0.44 | 0.66 | 0.50 | **52 ms** |
| hybrid + reranker, pool 30 | 0.66 | 0.69 | 0.81 | 0.67 | 243 ms |
| **hybrid + reranker, pool 50** ✅ | **0.66** | **0.69** | **0.88** | **0.71** | **354 ms** |

- The neural reranker lifts **quote_hit@3 by +61%** (0.41 → 0.66) and src_recall@5
  from 0.66 → 0.88. This is the single biggest change over the client's pipeline,
  which had **no reranker** at all.
- Latency budget (guideline: <1 s for search): the full search stays at
  **243–354 ms — a ~3× margin**. Per-step: embed ~26 ms · dense ~3 ms · bm25 ~24 ms
  · rrf <1 ms · rerank ~185–300 ms.
- **FP16 was essential**: FP32 reranking was ~250 ms→900 ms/query (blew the budget);
  FP16 cut it to ~90–300 ms with no measurable quality loss.

Chosen default: `fused_top=50, final_top=5, rrf_k=60`, reranker FP16 @ max_length 256.

## Generation — 2B hits the latency target

Reasoning **disabled** (see below). Answers generated from the top-5 reranked chunks.

| model | gen avg | gen p90 | tps | end-to-end (search+gen) |
|---|---|---|---|---|
| **Qwen3.5-2B Q8** ✅ | 1.72 s | 2.52 s | 118 | **~2.1 s avg / ~2.9 s p90** |
| Qwen3.5-4B Q8 | 3.35 s | 5.13 s | 59 | ~3.7 s avg / ~5.5 s p90 |

- **2B meets the 2–3 s stage-1 target.** 4B is more faithful to the source but ~2×
  slower; keep it as a quality option, not the latency default.
- Qualitatively both are hugely better than the legacy regex-template answers. 2B
  occasionally adds a minor unsupported qualifier; 4B stays closer to the text.

## Reasoning suppression (Qwen3.5 is a reasoning-distilled model)

Thinking adds latency we don't want for voice. **No server flag suppresses it** for
these Jackrong distilled GGUFs — the "peg-native" chat template hard-opens `<think>`
and the model always fills it (`--reasoning off`, `--reasoning-budget 0`,
`enable_thinking=false` all ignored). Fix: hit raw `/completion` with a **prefilled,
already-closed `<think></think>` block** so the model generates the answer directly.
Effect on the 2B: **0 reasoning tokens, gen 1.54 s → 0.77 s** on a short answer.

## Known limitations / next steps

- **quote_hit@5 caps at 0.69** while src_recall@5 is 0.88 — we get the right doc but
  not always the exact passage. Likely chunk granularity (11k small point-chunks
  fragment context). Next: sweep chunk size (bigger chunks / parent-doc retrieval)
  — requires a reindex.
- **Answer correctness is not auto-graded yet** — generation captures answers +
  latency; an LLM-judge pass over `runs/generation_*.json` would quantify quality.
- **ФЗ-152 (personal data)** is an empty stub in the corpus — excluded from the index.
- Prod still needs the retrieval + LLM co-resident on the client's 12 GB 3060 (fits:
  bge-m3 ~2.3 + reranker fp16 ~1.1 + Qwen 2B ~2.2 ≈ 5.6 GB).

## Reproduce

See `experiments-rag-params/README.md` for the run order.
