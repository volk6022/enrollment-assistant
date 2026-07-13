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

## Generation — 2B, concise prompt (re-verified 2026-07-13, Opus)

Reasoning **disabled** (see below). Answers generated from the top-5 reranked chunks.

> **Correction to the first pass.** The earlier default `max_tokens=200` was picked
> as a "grid winner" (1.30 s) but that was a **measurement artifact**: at 200 tokens
> **13/32 answers were cut off mid-sentence**, and generation was fast *only because
> answers were truncated* — tps is flat (~120) across every cap, so `max_tokens`
> bounds truncation, not speed. Fix: a **"2–4 sentences" instruction** in the system
> prompt drops natural answer length from ~156 → ~70 tokens, so answers finish
> cleanly **and** faster. `max_tokens=200` is now a pure safety cap.

| 2B (Q8) config | gen avg | gen p90 | avg tok | truncated | end-to-end |
|---|---|---|---|---|---|
| old: verbose prompt, max200 | 1.30 s | 1.66 s | 156 | **13/32 mid-sentence** | ~1.7 s |
| **new: concise prompt, max200** ✅ | **0.79 s** | **1.13 s** | 70 | **0/32** | **~1.15 s avg / ~1.5 s p90** |

Model comparison (concise prompt, same 32 queries):

| model | quant | gen avg | gen p90 | quality |
|---|---|---|---|---|
| **Qwen3.5-2B** ✅ | Q8_0 | 0.79 s | 1.13 s | grounded, correct |
| Qwen3.5-0.8B | Q4_K_M | 0.82 s | 1.50 s | **misreads questions, garbled tokens** |
| Qwen3.5-4B | Q8_0 | ~3.3 s | ~5.1 s | most faithful, too slow |

- **2B (Q8) is the default.** It clears the 2–3 s budget with ~2× margin and answers
  correctly. 4B is more faithful but ~4× slower — keep as a quality option only.
- **0.8B (Q4) is not worth it** (`bench_generation_08b.py`): only ~2× the 2B's tps
  while the pipeline is *already* 1.15 s (no latency problem to solve), and it exists
  **only in Q4** vs Q8 for 2B/4B — a double quality hit. On the eval set it misread a
  question (listed *categories of examinees* instead of the *doctors* asked) and
  emitted garbled tokens. Disqualifying for grounded legal/medical answers read aloud.
  Keep only as an emergency VRAM fallback (STT+TTS co-resident on 8 GB), gated behind
  an LLM-judge pass.

## Conversational (spoken) input — rephrasing & the shipped fix

The eval questions above are book-phrased. Real voice input is messy ("слушайте, а
кто там врачи будут, которые мне комиссию проводить?"). A sub-agent generated a
conversational paraphrase of each of the 32 questions (same intent, same gold), and
we measured how retrieval degrades and what recovers it. All arms scored against the
same gold sources; `formal` is the book-phrased upper bound.

**Spoken input tanks retrieval** (no rephraser): src_recall@5 **0.875 → 0.625**,
quote_hit@5 0.688 → 0.438. Worse, the LLM still produced confident full answers in
31/32 cases — retrieval misses become *confidently-wrong* answers, not refusals.

### What we tried (retrieval src_recall@5 / quote_hit@5)

| arm | recall@5 | quote@5 | note |
|---|---|---|---|
| formal (reference) | 0.875 | 0.688 | book-phrased upper bound |
| conv_baseline (no rephrase) | 0.625 | 0.438 | spoken input, untouched |
| v1 rephrase — **context-fed** | — | — | broken: 0/32 stayed questions, meaning inverted, +47.7% |
| v2 rephrase — replace, keyword-preserving prompt | 0.688–0.719 | 0.469 | modest gain; fixed the quote_hit regression |
| **multiquery_v3 — union + rerank on clean query** ✅ | **0.75–0.84** | **0.50–0.53** | **shipped** |
| kwexpand v3 (naive keywords) | 0.688 | 0.438 | rejected — floods the pool |
| kwexpand v4 (guided keywords, 1 BM25 vote) | 0.812 | 0.531 | rejected — no gain over v3 |

### Key findings

1. **Never feed the rephraser the retrieved context.** v1 did, so the reasoning model
   *answered* instead of rephrasing (0/32 were questions; meaning sometimes flipped).
   Fix: rephraser sees the **question only**.
2. **Replacing the query paraphrases away strong keywords.** A plain rewrite dropped
   "паспорт"/"СВО"/"адъюнктура" and even hallucinated definitions. A keyword-preserving
   prompt (v2) + mapping colloquial words onto **NPA stock phrases** ("врачи/комиссия"
   → "медицинское освидетельствование") fixed it.
3. **The real lever is multi-query + reranking on the CLEAN query.** Retrieve on BOTH
   the spoken query and the canonical rewrite (RRF-union), then **rerank with the
   canonical** (the messy query orders the reranker badly). This is `multiquery_v3`:
   src_recall@5 **0.625 → ~0.84**, nearly closing the gap to formal (0.875).
4. **Keyword expansion does NOT help — proven twice.** Naive (7 equal-weight keyword
   queries) floods the fused pool with generic-term distractors and one even dragged an
   off-topic subtopic ("физподготовка" into a benefits question). The guided salvage
   (discriminative NPA phrases as ONE BM25-disjunction vote) still gave **no gain**
   (0.812 vs 0.844). With a strong dense retriever + reranker there's no recall left for
   keywords to recover; the residual misses are semantic/chunking, not lexical.

### Shipped

`RagConfig.conversational = True` enables the mode in `rag/pipeline.py`
(`search_conversational`): rephrase (`rag/rephrase.py`, NPA-grounded prompt) → union
of spoken+canonical → rerank on canonical → generate on the original spoken question.
**Verified end-to-end through the production Pipeline** (`verify_prod_conversational.py`):
src_recall@5 **0.781**, quote_hit@5 0.562, ~2.45 s incl. the extra rephrase call
(default `conversational=False` keeps the fast ~1.15 s path for clean input).

Benches: `bench_conversational_rephraser.py` (v1/v2), `bench_conversational_multiquery.py`
(v3), `bench_conversational_kwexpand.py` / `_v4.py` (keywords), `runs/keyword_guide.md`
(sub-agent NPA vocabulary guide).

## Reasoning suppression (Qwen3.5 is a reasoning-distilled model)

Thinking adds latency we don't want for voice. **No server flag suppresses it** for
these Jackrong distilled GGUFs — the "peg-native" chat template hard-opens `<think>`
and the model always fills it (`--reasoning off`, `--reasoning-budget 0`,
`enable_thinking=false` all ignored). Fix: hit raw `/completion` with a **prefilled,
already-closed `<think></think>` block** so the model generates the answer directly.
Effect on the 2B: **0 reasoning tokens, gen 1.54 s → 0.77 s** on a short answer.

## Known limitations / next steps

- **quote_hit@5 caps at ~0.69** while src_recall@5 is 0.88 — right doc, not always the
  exact passage. **Diagnosed (2026-07-13):** 31/32 gold quotes sit fully inside ONE
  chunk, so it's **not** quote-splitting — it's **fragmentation**. The corpus is 11293
  chunks averaging ~446 chars (one legal point each), so the top-5 covers little text
  and the quote-chunk often isn't ranked in. Swept via `bench_chunking.py` — see below.

### Chunk-size sweep (`bench_chunking.py`, formal questions)

Merge consecutive same-source point-chunks up to a target size, reindex, re-measure:

| target | chunks | avg chars | recall@5 | quote_hit@3 | quote_hit@5 | MRR |
|---|---|---|---|---|---|---|
| none (current) | 11293 | 446 | **0.875** | 0.656 | 0.688 | 0.708 |
| **~900** ✅ | 5856 | 861 | 0.844 | **0.719** | **0.719** | **0.773** |
| ~1500 | 4208 | 1198 | 0.781 | 0.625 | 0.625 | 0.669 |
| ~2500 | 2260 | 2232 | 0.750 | 0.562 | 0.562 | 0.648 |

- **~860 chars (roughly 2 points/chunk) is the optimum.** quote_hit@3 **+6 pts**,
  quote_hit@5 +3 pts, and MRR **0.708 → 0.773** — the evidence chunk ranks higher.
  Cost: −1 question on src_recall@5 (0.875 → 0.844). Net win for the LLM (it gets
  better-ranked, more-complete evidence).
- **Bigger is worse.** 1500/2500 chars degrade every metric — merging dilutes
  relevance and the cross-encoder reranker loses precision on long candidates.
- **Next:** to ship, add sibling-point merging to `rag/ingest.py` (target ~900) and
  reindex prod artifacts; or keep small chunks + parent-window expansion at retrieval
  (keeps recall, widens the quote net). Not yet wired into prod.
- **Answer correctness is not auto-graded yet** — generation captures answers +
  latency; an LLM-judge pass over `runs/generation_*.json` would quantify quality.
- **ФЗ-152 (personal data)** is an empty stub in the corpus — excluded from the index.
- Prod still needs the retrieval + LLM co-resident on the client's 12 GB 3060 (fits:
  bge-m3 ~2.3 + reranker fp16 ~1.1 + Qwen 2B ~2.2 ≈ 5.6 GB).

## Reproduce

See `experiments-rag-params/README.md` for the run order.
