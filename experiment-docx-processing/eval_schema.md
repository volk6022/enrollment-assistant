# Eval item contract

Each eval file is a JSON array of items. One item:

```json
{
  "id": "pravila-001",
  "question": "До какого числа нужно подать документы для поступления в 2025 году?",
  "answer": "Заявление с документами подаётся с 20 июня по 20 июля 2025 года.",
  "gold_source": "!!!! ПРАВИЛА ПРИЕМА в  2025 РЕДАКЦИЯ ОТ 6 июня 2025 № 1093.docx",
  "gold_quote": "приём документов … проводится с 20 июня по 20 июля",
  "section": "п. 4.2",
  "topic": "deadlines",
  "difficulty": "easy"
}
```

## Field rules

- **question** — in Russian, phrased the way a real applicant (абитуриент) or
  parent would ask. Natural, conversational, sometimes underspecified.
- **answer** — concise ground-truth answer (1–3 sentences), fully supported by
  `gold_quote`. No information beyond the source.
- **gold_source** — the exact source `.docx` filename (from `out/manifest.json`).
- **gold_quote** — a **verbatim substring** of the source `.txt` that supports the
  answer. Must be copyable-exact (checked mechanically). Keep it short but
  sufficient; you may use `…` only *between* two verbatim fragments if needed —
  each fragment must itself be an exact substring.
- **section** — пункт / раздел / статья label if the text gives one, else "".
- **topic** — one of:
  `apply | deadlines | documents | exams | min_scores | benefits | health |
   physical | without_ege | achievements | hostel | general`
- **difficulty** —
  `easy` (direct lookup) | `medium` (needs light synthesis) |
  `hard` (multi-part, or requires combining two places in the doc).

## Quality bar

- Prefer questions that discriminate retrieval quality: specific numbers, dates,
  document lists, score thresholds, eligibility conditions.
- Spread across topics and difficulties.
- Never invent facts not in the source. If unsure, drop the item.
