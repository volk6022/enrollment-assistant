# Enrollment Assistant — RAG (stage 1)

Голосовой ассистент приёмной комиссии ДВЮИ МВД. Пайплайн retrieval:
`embed (bge-m3) → BM25+dense → RRF → rerank (bge-reranker-v2-m3) → Qwen3.5 (llama.cpp)`.
Подробности: `BACKEND.md`, `rag/RAG_MAPPING.md`, `experiments-rag-params/RESULTS.md`.

## Индексы: переключение между корпусами

Пайплайн ищет по одному индексу в `rag/artifacts/` (`chunks.jsonl` + `dense.faiss` +
`bm25.pkl`). Собрать его можно из **двух источников**, и между ними можно переключаться:

| Индекс | Источник | Чем хорош |
| --- | --- | --- |
| **raw-корпус** | 31 исходный НПА (`experiment-docx-processing/out/txt/`) | полный текст, дословные цитаты |
| **compiled KB** | компилированная база `data/npa/knowledge-base/wiki/` | чище/меньше (43 чанка), выше src_recall на разговорном вводе и MRR |

Любая сборка индекса — это **два шага**: сначала `chunks.jsonl` (чанкинг), затем
переэмбеддинг в FAISS+BM25 (`rag.index`). **Второй шаг обязателен** после смены чанков.

### Переключиться на compiled KB (wiki)

```powershell
uv run python experiments-rag-params/build_wiki_index.py   # wiki/ -> chunks.jsonl (+ chunks.jsonl.bak)
uv run python -m rag.index                                 # переэмбеддинг FAISS + BM25
```

### Переключиться обратно на raw-корпус

```powershell
uv run python -m rag.ingest    # txt-корпус -> chunks.jsonl (структурный чанкинг + merge ~900)
uv run python -m rag.index     # переэмбеддинг FAISS + BM25
```

### Прогнать тесты на текущем индексе

```powershell
uv run python experiments-rag-params/prod_full_eval.py
# -> experiments-rag-params/runs/prod_eval_{formal,conversational}_<ts>.json
#    (src_recall@k, quote_hit@k, mrr, latency, сгенерированные ответы)
```

### Примечания

- `build_wiki_index.py` перед перезаписью кладёт текущий `chunks.jsonl` в
  `chunks.jsonl.bak` (страховка на один шаг назад; на несколько переключений не
  полагаться — raw-индекс всегда детерминированно пересобирается через `rag.ingest`).
- Если машина офлайн (эмбеддер `bge-m3` уже скачан в `models/hub`), включи офлайн-режим
  HuggingFace перед `rag.index`:
  ```powershell
  $env:HF_HUB_OFFLINE = '1'; uv run python -m rag.index
  ```
- `quote_hit` на compiled KB ≈ 0 по построению (wiki перефразирует НПА, а метрика ищет
  дословную цитату) — качество compiled-индекса меряется по `src_recall`/MRR и по самим
  ответам, не по `quote_hit`. См. `experiments-rag-params/RESULTS.md`.
