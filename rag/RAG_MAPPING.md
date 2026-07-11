# Legacy RAG → new pipeline mapping (stage 1)

Where the current behavior lives in the client's code, and what replaces it. Goal
(per `raw/naive-rag-guide-line.md`): a clean LangChain-style hybrid retriever +
neural reranker + capable local Qwen3.5, replacing the hand-tuned scoring engine.
The Yandex engine (`orchestrator_v11`, `app/yandex/*`) is **out of scope** — untouched.

## Legacy components (in `backend/app/`)

| Concern | Legacy location | What it does | Verdict |
|---|---|---|---|
| docx → text | `npa_indexer._extract_docx_paragraphs` | paragraphs only, **drops tables** | replace — our `experiment-docx-processing/extract_docx.py` keeps tables |
| chunking | `npa_indexer.chunk_docx` | splits on numbered legal points (`^\d+(\.\d+)*[.)]`), tracks РАЗДЕЛ/ГЛАВА section path, 1200-char/180-overlap fallback | **keep the idea** — structure-aware, good for НПА |
| chunk metadata | `npa_indexer.index_docx_folder` | source, doc_title, revision_date, legal_force_{code,name,level}, doc_kind, program_scope, topic_scope, point, section_path | **keep a subset** (source, doc_title, revision_date, legal_force_level, point, section_path) |
| embeddings | `rag.RAG.__init__` + `npa_indexer` | `fastembed` `paraphrase-multilingual-MiniLM-L12-v2` (384-d, weak RU) | replace → **bge-m3** (multilingual, RU-strong) |
| vector store | Qdrant (`docker-compose`) | cosine ANN | experiments → local **FAISS**; prod can stay Qdrant |
| BM25 | `rag.RAG._bm25_candidates` | `rank_bm25.BM25Okapi` over cached chunk texts | **keep** |
| fusion | `rag.RAG._search_one_collection` | RRF (k=60) over BM25+dense variants | **keep** — this part is already right |
| ranking | `rag._score_hit`, `_source_specific_bonus`, `_edition_bonus`, `_topic_overlap`, `_program_overlap`, `legal_hierarchy.*` | ~300 lines of hand-tuned regex/keyword score bonuses | **remove** → replaced by a neural reranker |
| reranker | — | **none exists** | add → **bge-reranker-v2-m3** cross-encoder (biggest quality lever) |
| answer gen | `llm_generator._structured_answer` | ~500-line per-intent regex template engine; LLM only if this fails | **remove** → LLM answers from clean reranked context |
| LLM call | `llm_generator._llm_answer` | Ollama `qwen2.5:3b`, rigid `VOICE/USED_IDS/ANSWER` format | replace → **Qwen3.5 2B/4B via llama.cpp**, single clean prompt + citations |

## Why the legacy answers were unreliable (confirmed reading the code)

1. The LLM is a **last resort** — `generate_llm_answer` returns a regex-built
   template whenever `_structured_answer` matches, so the "AI" rarely speaks.
2. Retrieval ranking is a **tower of manual weights** (`0.28`, `-0.46`, `0.24`…)
   tuned by hand per intent — brittle, untestable, overfit to sample questions.
3. No reranker — the single biggest, cheapest quality win is missing.
4. The rigid output format is hard for a 3B model, so even when the LLM runs it
   often mis-formats and loses its answer.

## New pipeline (`rag/`)

```
query
  → embed (bge-m3)            ─┐
  → BM25 (rank_bm25)          ─┴→ RRF(k=60) → top ~30
  → rerank (bge-reranker-v2-m3) → top 3–5
  → Qwen3.5 (llama.cpp server) → grounded answer + citations
```

Modules: `ingest.py` (docx/txt → chunks+meta), `embed.py` (bge-m3),
`index.py` (FAISS dense + BM25), `retrieve.py` (hybrid + RRF),
`rerank.py` (bge-reranker-v2-m3), `generate.py` (llama.cpp client),
`pipeline.py` (end-to-end with per-step timings). Params & latency/quality grid
live in `experiments-rag-params/`, scored against
`experiment-docx-processing/out/eval/eval_set.json`.
```
