"""Structure-aware chunking of the extracted NPA text into RAG chunks.

Reuses the legacy idea (npa_indexer.chunk_docx): start a new chunk at each
numbered legal point (`^N.N.N ...`), carry a section path from headings, and
split over-long chunks with overlap. Adapted to our extractor's output, where
headings are prefixed `## ` and tables are pipe-separated rows.

Metadata kept per chunk is deliberately minimal — the neural reranker replaces
the legacy tower of metadata score-bonuses. We keep only what's genuinely useful
for citations and edition-preference: source, doc_title, revision_date, point,
section_path.

Output: rag/artifacts/chunks.jsonl  (one JSON object per line)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rag.config import ARTIFACTS, MANIFEST, TXT_DIR, DEFAULT

POINT_RE = re.compile(r"^(\d+(?:\.\d+){0,6})[.)]\s+")
CONSULTANT_NOISE = re.compile(
    r"^\(?(?:см\.\s+текст|в ред\.|КонсультантПлюс|Список изменяющих)", re.IGNORECASE
)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if s.startswith("## "):
        return True
    if re.match(r"^(РАЗДЕЛ|ГЛАВА|ПРИЛОЖЕНИЕ|СТАТЬЯ)\b", s, re.IGNORECASE):
        return True
    if re.match(r"^[IVXLCDM]+\.\s+", s):
        return True
    if s and s.upper() == s and len(s) <= 160 and not re.search(r"\d{2}\.\d{2}\.\d{4}", s):
        return True
    return False


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    t = text.strip()
    if len(t) <= max_chars:
        return [t]
    out, start = [], 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        # try not to cut mid-sentence
        if end < len(t):
            dot = t.rfind(". ", start + max_chars - overlap, end)
            if dot != -1:
                end = dot + 1
        part = t[start:end].strip()
        if part:
            out.append(part)
        if end >= len(t):
            break
        start = max(0, end - overlap)
    return out


def merge_siblings(chunks: list[dict], target: int) -> list[dict]:
    """Merge consecutive point-chunks (within one doc) up to `target` chars.

    The base chunker emits one chunk per legal point (~446 chars avg), which
    fragments context: the top-5 covers little text so the exact quote-chunk often
    isn't ranked in. Merging sibling points to ~860 chars lifts quote_hit@3 +6pts
    and MRR 0.708->0.773 (see experiments-rag-params/RESULTS.md chunk-size sweep).
    Keeps the first point number of the group for citation. target=0 disables.
    """
    if not target:
        return chunks
    out: list[dict] = []
    cur: dict | None = None
    for c in chunks:
        if cur is not None and len(cur["text"]) + len(c["text"]) + 1 <= target:
            cur["text"] += "\n" + c["text"]
            if not cur.get("point"):
                cur["point"] = c.get("point")
        else:
            if cur is not None:
                out.append(cur)
            cur = dict(c)
    if cur is not None:
        out.append(cur)
    return out


def chunk_text(lines: list[str], max_chars: int, overlap: int) -> list[dict]:
    section_path: list[str] = []
    chunks: list[dict] = []
    buf: list[str] = []
    buf_point: str | None = None
    buf_section: list[str] = []

    def flush():
        nonlocal buf, buf_point, buf_section
        if not buf:
            return
        joined = "\n".join(buf).strip()
        if joined:
            for part in _split_long(joined, max_chars, overlap):
                chunks.append({"text": part, "point": buf_point, "section_path": list(buf_section)})
        buf, buf_point, buf_section = [], None, []

    for raw in lines:
        line = raw.strip()
        if not line or CONSULTANT_NOISE.match(line):
            continue
        if _is_heading(line):
            heading = line[3:].strip() if line.startswith("## ") else line
            if buf:
                buf.append(heading)
            else:
                if len(section_path) >= 4:
                    section_path = section_path[:3]
                section_path.append(heading[:120])
            continue
        m = POINT_RE.match(line)
        if m:
            flush()
            buf_point = m.group(1)
            buf_section = list(section_path)
            buf = [line]
            continue
        if not buf:
            # preamble text before any numbered point still deserves a chunk
            buf_section = list(section_path)
            buf = [line]
            continue
        buf.append(line)
    flush()
    return chunks


def build_chunks(cfg=DEFAULT) -> Path:
    manifest = {m["source"]: m for m in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS / "chunks.jsonl"

    n_docs = n_chunks = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for meta in manifest.values():
            if "txt" not in meta:
                continue
            txt_path = TXT_DIR / meta["txt"]
            if not txt_path.exists():
                continue
            text = txt_path.read_text(encoding="utf-8")
            if len(text.strip()) < 50:  # skip the ФЗ-152 stub
                continue
            lines = text.splitlines()
            doc_title = next((ln[3:].strip() for ln in lines if ln.startswith("## ")), meta["source"])
            n_docs += 1
            doc_chunks = [c for c in chunk_text(lines, cfg.max_chars, cfg.overlap)
                          if len(c["text"]) >= 40]
            doc_chunks = merge_siblings(doc_chunks, cfg.merge_chunk_chars)
            for i, c in enumerate(doc_chunks):
                rec = {
                    "id": f"{n_docs:02d}-{i:04d}",
                    "text": c["text"],
                    "source": meta["source"],
                    "doc_title": doc_title[:200],
                    "revision_date": meta.get("revision_date"),
                    "point": c["point"],
                    "section_path": c["section_path"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"{n_docs} docs -> {n_chunks} chunks -> {out_path}")
    return out_path


if __name__ == "__main__":
    build_chunks()
