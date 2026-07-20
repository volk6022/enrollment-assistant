"""Build a RAG chunks.jsonl from the COMPILED knowledge base (data/npa/knowledge-base/wiki)
instead of the raw NPA txt corpus, so we can eval the recompiled KB with the existing harness.

Chunking strategy: one chunk per section (## / ### heading) of each wiki note, split if
over-long. Each chunk's `source` is mapped back to the ORIGINAL .docx name (parsed from the
"Источник: <doc>, п. N" citations in the section) so score.py's source_recall works and is
comparable to the raw-corpus baseline. quote_hit is expected to be low (the wiki paraphrases;
the gold quotes are verbatim NPA) — that's why answer quality is judged separately.

Writes rag/artifacts/chunks.jsonl (backs up the previous one to chunks.jsonl.bak first).
Then run:  python -m rag.index   (re-embeds)  and  prod_full_eval.py.
Restore raw corpus anytime with:  python -m rag.ingest && python -m rag.index
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> import rag.*
try:  # Cyrillic source names in the summary crash on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag.config import ARTIFACTS, DEFAULT
from rag.ingest import _split_long

# Wiki dir can be overridden as argv[1] (e.g. knowledge-base-full/wiki) so the same
# builder serves KB-relevant and KB-full. Defaults to the relevant KB.
WIKI = Path(sys.argv[1] if len(sys.argv) > 1
            else r"E:/voice-agent/enrollment-assistant/data/npa/knowledge-base/wiki")
SKIP = {"00-context.md", "INDEX.md"}

# distinctive token in the citation -> exact eval gold_source .docx string.
# The 4 docs that appear as eval gold_source must map EXACTLY; the rest are best-effort
# (they are not gold sources, so they never affect source_recall).
SRC_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"1093"), "!!!! ПРАВИЛА ПРИЕМА в  2025 РЕДАКЦИЯ ОТ 6 июня 2025 № 1093.docx"),
    (re.compile(r"[N№]\s*370"), "Приказ МВД России от 14_06_2018 N 370 (ред_ от 23_12_2022).docx"),
    (re.compile(r"876"), "Приказ Рособрнадзора от 26_06_2019 N 876 (ред_ от 22_04_2024.docx"),
    (re.compile(r"268"), "Указ Президента РФ от 09_05_2022 N 268 (ред_ от 05_08_2022).docx"),
    (re.compile(r"[N№]\s*620"), "Приказ МВД России от 21_10_2024 N 620  О требованиях к состо.docx"),
    (re.compile(r"[N№]\s*568"), "Приказ МВД России от 07_07_2014 N 568 (ред_ от 29_08_2023).docx"),
    (re.compile(r"2122"), "Постановление Правительства РФ от 30_11_2021 N 2122  Об утве.docx"),
    (re.compile(r"[N№]\s*565|565"), "Постановление Правительства РФ от 04_07_2013 N 565 (ред_ от.docx"),
    (re.compile(r"19-ФЗ|19-фз|[N№]\s*19"), "Федеральный закон от 17_02_2023 N 19-ФЗ  Об особенностях пра.docx"),
    (re.compile(r"[N№]\s*820|820"), "Приказ Минобрнауки России от 27_11_2024 N 820  Об утверждени.docx"),
    (re.compile(r"[N№]\s*177|177"), "Приказ МВД России от 31_03_2025 N 177  Об утверждении Порядк.docx"),
]
CIT_RE = re.compile(r"Источник:\s*([^*\n]+)")
POINT_RE = re.compile(r"п\.\s*([\d.]+(?:[–-]\d[\d.]*)?)")


def map_source(cite_text: str) -> str | None:
    """Map a citation to a .docx. When a citation names several docs (e.g.
    '...№ 876...; Правила...(№ 1093)...'), the LEFTMOST-mentioned one wins — my
    citations list the primary source first."""
    best, best_pos = None, 10**9
    for pat, docx in SRC_MAP:
        m = pat.search(cite_text)
        if m and m.start() < best_pos:
            best, best_pos = docx, m.start()
    return best


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:]
    return text


def note_primary_source(text: str) -> str | None:
    m = re.search(r"^sources:\s*\[(.+?)\]", text, re.M)
    if m:
        first = m.group(1).split(",")[0].strip().strip('"')
        return map_source(first)
    return None


def units(body: str):
    """Split a note body into citation-units: text accumulates until a line with
    'Источник:' closes a unit. Yields (heading, text). Drops the 'См. также' nav."""
    cur_head, buf = "", []
    for line in body.splitlines():
        if line.startswith("#"):
            h = line.lstrip("# ").strip()
            if buf:
                yield cur_head, "\n".join(buf); buf = []
            cur_head = None if h.lower().startswith("см. также") else h
            continue
        if cur_head is None:
            continue
        buf.append(line)
        if "Источник:" in line:
            yield cur_head, "\n".join(buf); buf = []
    if buf and cur_head is not None:
        yield cur_head, "\n".join(buf)


def build() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "chunks.jsonl"
    if out.exists():
        shutil.copy(out, ARTIFACTS / "chunks.jsonl.bak")

    n_notes = n_chunks = 0
    records = []
    for path in sorted(WIKI.glob("*.md")):
        if path.name in SKIP:
            continue
        raw = path.read_text(encoding="utf-8")
        body = strip_frontmatter(raw)
        primary = note_primary_source(raw)
        note_title = next((ln.lstrip("# ").strip() for ln in body.splitlines()
                           if ln.startswith("# ")), path.stem)
        n_notes += 1

        # 1) build per-citation units with an accurate source each
        raw_units = []
        for head, text in units(body):
            t = text.strip()
            if len(t) < 25:
                continue
            cites = [m for m in (map_source(c) for c in CIT_RE.findall(text)) if m]
            src = Counter(cites).most_common(1)[0][0] if cites else (primary or "unknown")
            pt = POINT_RE.search(text)
            raw_units.append({"head": head or "", "text": t, "src": src,
                              "point": pt.group(1) if pt else None})

        # 2) merge consecutive same-source units up to ~900 chars (cut fragmentation)
        merged = []
        cur = None
        for u in raw_units:
            if cur and cur["src"] == u["src"] and len(cur["text"]) + len(u["text"]) + 1 <= 900:
                cur["text"] += "\n" + u["text"]
                cur["point"] = cur["point"] or u["point"]
            else:
                if cur:
                    merged.append(cur)
                cur = dict(u)
        if cur:
            merged.append(cur)

        # 3) emit, prefixing the heading for context; split any over-long chunk
        for u in merged:
            full = (f"{u['head']}\n{u['text']}" if u["head"] else u["text"]).strip()
            for part in _split_long(full, 1100, DEFAULT.overlap):
                records.append({
                    "text": part,
                    "source": u["src"],
                    "doc_title": note_title[:200],
                    "revision_date": None,
                    "point": u["point"],
                    "section_path": [note_title, u["head"]] if u["head"] else [note_title],
                })

    with out.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            rec["id"] = f"wiki-{i:04d}"
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_chunks += 1

    by_src = Counter(r["source"] for r in records)
    print(f"{n_notes} notes -> {n_chunks} chunks -> {out}")
    print("chunks per source:")
    for s, c in by_src.most_common():
        print(f"  {c:3d}  {s}")
    return out


if __name__ == "__main__":
    build()
