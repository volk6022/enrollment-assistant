"""Extract text from the NPA .docx corpus into structure-preserving .txt files.

The corpus is exported from КонсультантПлюс/Garant, which produces slightly
non-conformant OPC packages (python-docx's Document() chokes on them with a
'word/word/settings.xml' error). So we bypass the package loader and parse
`word/document.xml` straight from the zip with lxml — robust for every file.

Walks the document body in reading order so paragraphs and tables stay
interleaved (legal docs put score/deadline tables inline). Tables render as
pipe-separated rows. Paragraph breaks are preserved; runs of blank paragraphs
collapse to a single blank line. Headings (w:pStyle ~ Heading/Заголовок) are
prefixed with '## '.

Also parses issue/revision dates from the file name (the corpus encodes them,
e.g. "... от 29_12_2012 N 273-ФЗ (ред_ от 15_10_2025 ...") for the downstream
"newer edition wins" rule.

Outputs:
  out/txt/<stem>.txt        one plain-text file per source docx
  out/manifest.json         per-doc metadata (dates, counts, source path)
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC_DIR = REPO / "data" / "npa" / "2025"
OUT_TXT = HERE / "out" / "txt"
OUT_MANIFEST = HERE / "out" / "manifest.json"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def w(tag: str) -> str:
    return f"{{{W}}}{tag}"

DATE_RE = re.compile(r"(\d{2})[._](\d{2})[._](\d{4})")
REV_RE = re.compile(r"ред[._]?\s*от\s*(\d{2})[._](\d{2})[._](\d{4})", re.IGNORECASE)


def _para_text(p: etree._Element) -> str:
    """Concatenate all text in a paragraph, honoring tabs and line breaks."""
    parts: list[str] = []
    for node in p.iter():
        tag = node.tag
        if tag == w("t"):
            parts.append(node.text or "")
        elif tag == w("tab"):
            parts.append("\t")
        elif tag in (w("br"), w("cr")):
            parts.append("\n")
    return "".join(parts)


def _is_heading(p: etree._Element) -> bool:
    ppr = p.find(w("pPr"))
    if ppr is None:
        return False
    style = ppr.find(w("pStyle"))
    if style is None:
        return False
    val = (style.get(w("val")) or "").lower()
    return "heading" in val or "заголовок" in val or val.startswith("h")


def _render_table(tbl: etree._Element) -> str:
    lines = []
    for tr in tbl.findall(w("tr")):
        cells = []
        for tc in tr.findall(w("tc")):
            cell_text = " ".join(
                " ".join(_para_text(p).split()) for p in tc.findall(w("p"))
            ).strip()
            cells.append(cell_text)
        deduped = []
        for c in cells:  # drop repeats from horizontally-merged cells
            if not deduped or deduped[-1] != c:
                deduped.append(c)
        lines.append(" | ".join(deduped))
    return "\n".join(lines)


def extract_one(path: Path) -> tuple[str, dict]:
    with zipfile.ZipFile(str(path)) as z:
        xml = z.read("word/document.xml")
    root = etree.fromstring(xml)
    body = root.find(w("body"))
    if body is None:
        raise ValueError("no <w:body>")

    out_lines: list[str] = []
    para_count = 0
    table_count = 0
    blank_streak = 0

    for child in body:
        if child.tag == w("p"):
            text = _para_text(child).strip()
            if not text:
                blank_streak += 1
                if blank_streak <= 1:
                    out_lines.append("")
                continue
            blank_streak = 0
            if _is_heading(child):
                out_lines.append(f"## {text}")
            else:
                out_lines.append(text)
            para_count += 1
        elif child.tag == w("tbl"):
            blank_streak = 0
            out_lines.append("")
            out_lines.append(_render_table(child))
            out_lines.append("")
            table_count += 1

    text = "\n".join(out_lines).strip() + "\n"

    name = path.name
    all_dates = DATE_RE.findall(name)
    rev = REV_RE.search(name)
    issued = None
    if all_dates:
        d, m, y = all_dates[0]
        issued = f"{y}-{m}-{d}"
    revision = None
    if rev:
        d, m, y = rev.groups()
        revision = f"{y}-{m}-{d}"
    elif len(all_dates) > 1:
        d, m, y = all_dates[-1]
        revision = f"{y}-{m}-{d}"

    meta = {
        "source": name,
        "issued_date": issued,
        "revision_date": revision or issued,
        "char_count": len(text),
        "para_count": para_count,
        "table_count": table_count,
    }
    return text, meta


def main() -> int:
    OUT_TXT.mkdir(parents=True, exist_ok=True)
    docx_files = sorted(SRC_DIR.glob("*.docx"))
    if not docx_files:
        print(f"No .docx found in {SRC_DIR}", file=sys.stderr)
        return 1

    manifest = []
    for path in docx_files:
        try:
            text, meta = extract_one(path)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {path.name}: {e}", file=sys.stderr)
            manifest.append({"source": path.name, "error": str(e)})
            continue
        stem = path.stem.strip().rstrip(".")
        out_path = OUT_TXT / f"{stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        meta["txt"] = out_path.name
        manifest.append(meta)
        print(f"OK  {meta['char_count']:>8} chars  {meta['table_count']:>3} tbl  {meta['para_count']:>4} par  {path.name}")

    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ok = [m for m in manifest if "error" not in m]
    print(f"\n{len(ok)}/{len(manifest)} extracted -> {OUT_TXT}")
    print(f"manifest -> {OUT_MANIFEST}")
    return 0 if len(ok) == len(manifest) else 2


if __name__ == "__main__":
    raise SystemExit(main())
