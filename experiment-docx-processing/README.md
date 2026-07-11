# experiment-docx-processing

Turns the client's `data/npa/2025/*.docx` legal corpus into clean, RAG-ready
plain text, plus a grounded evaluation set and a cross-document contradiction
report.

## Pipeline

1. **Extract** — `extract_docx.py` parses `word/document.xml` straight from each
   docx zip (the corpus is a КонсультантПлюс export whose OPC packages break
   `python-docx`'s loader). Preserves paragraph structure and renders tables as
   pipe-separated rows. Parses issue/revision dates from filenames.
   → `out/txt/<stem>.txt`, `out/manifest.json`

2. **Eval generation** — haiku sub-agents read applicant-facing documents and
   write grounded question / ground-truth / gold-quote items. Every `gold_quote`
   must be verbatim from the source (checked by `verify_eval.py`).
   → `out/eval/<slug>.json`, merged `out/eval/eval_set.json`

3. **Contradiction check** — a haiku sub-agent compares the local ПРАВИЛА ПРИЕМА
   (2025) against the federal/ministry acts it derives from, for the applicant
   topics, and reports discrepancies with the "newer / more specific wins"
   resolution. **Non-destructive** — see policy below.
   → `out/contradictions.json`, `out/contradictions.md`

## Contradiction policy (important)

These are **legal source documents**. We do **not** rewrite them to "resolve"
conflicts — altering the text of a federal law or ministry order would falsify
the corpus. Instead the "newer information wins" rule is expressed as:

- a **contradictions report** (what each doc says + which supersedes + why), and
- **priority metadata** (`revision_date`, legal-force rank) the retrieval layer
  already uses to prefer the current edition.

The RAG layer ranks by edition/authority; the sources stay intact and auditable.

## Corpus notes / data quality

- **ФЗ-152 «О персональных данных»** — the source docx is a near-empty 3.6 KB
  stub (33 chars extracted, 0 paragraphs). The actual law text was never
  included in the client's export. Flagged, not fabricated. If the assistant
  must answer personal-data questions, re-export this file.
- The corpus is a КонсультантПлюс export: revision notes like
  `(в ред. приказов … )` and `(см. текст в предыдущей редакции)` are wrapped in
  single-column tables. They carry edition history but are noise for answers —
  candidates for filtering at chunk time.
- 31 real documents + 1 stub (ФЗ-152) = 32 files.

## Run

```powershell
$env:ALL_PROXY="socks5h://127.0.0.1:10808"   # this box: uv/hf need the SOCKS proxy
$env:PYTHONUTF8="1"
uv run python experiment-docx-processing/extract_docx.py
uv run python experiment-docx-processing/verify_eval.py
```
