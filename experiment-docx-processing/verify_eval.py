"""Validate agent-generated eval items and merge the good ones.

Checks each item in out/eval/*.json (except the merged file):
  - required fields present, topic/difficulty in allowed sets
  - gold_source exists in the corpus
  - every gold_quote fragment (split on '…') is a whitespace-normalized
    substring of the source .txt  ← the anti-hallucination gate

Writes out/eval/eval_set.json with only the items that pass, and prints a report.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TXT_DIR = HERE / "out" / "txt"
EVAL_DIR = HERE / "out" / "eval"
MANIFEST = HERE / "out" / "manifest.json"
MERGED = EVAL_DIR / "eval_set.json"

TOPICS = {
    "apply", "deadlines", "documents", "exams", "min_scores", "benefits",
    "health", "physical", "without_ege", "achievements", "hostel", "general",
}
DIFFS = {"easy", "medium", "hard"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load_sources() -> dict[str, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    src = {}
    for m in manifest:
        if "txt" in m:
            p = TXT_DIR / m["txt"]
            if p.exists():
                src[m["source"]] = norm(p.read_text(encoding="utf-8"))
    return src


def check_item(item: dict, sources: dict[str, str]) -> list[str]:
    errs = []
    for f in ("id", "question", "answer", "gold_source", "gold_quote", "topic", "difficulty"):
        if not item.get(f) and f != "gold_quote":
            errs.append(f"missing {f}")
    if item.get("topic") not in TOPICS:
        errs.append(f"bad topic {item.get('topic')!r}")
    if item.get("difficulty") not in DIFFS:
        errs.append(f"bad difficulty {item.get('difficulty')!r}")
    gs = item.get("gold_source")
    if gs not in sources:
        errs.append(f"unknown gold_source {gs!r}")
        return errs
    hay = sources[gs]
    quote = item.get("gold_quote") or ""
    for frag in (f for f in quote.split("…") if f.strip()):
        if norm(frag) not in hay:
            errs.append(f"quote fragment not found: {frag[:60]!r}")
    return errs


def main() -> int:
    sources = load_sources()
    files = sorted(p for p in EVAL_DIR.glob("*.json") if p.name != MERGED.name)
    if not files:
        print("no eval files yet in", EVAL_DIR)
        return 1

    good, bad = [], []
    seen_ids = set()
    for fp in files:
        try:
            items = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {fp.name}: unreadable ({e})")
            continue
        for item in items:
            iid = item.get("id", "?")
            if iid in seen_ids:
                item["id"] = iid = f"{iid}-{fp.stem}"
            seen_ids.add(iid)
            errs = check_item(item, sources)
            if errs:
                bad.append((fp.name, iid, errs))
            else:
                item["_file"] = fp.name
                good.append(item)

    MERGED.write_text(json.dumps(good, ensure_ascii=False, indent=2), encoding="utf-8")

    by_topic: dict[str, int] = {}
    for it in good:
        by_topic[it["topic"]] = by_topic.get(it["topic"], 0) + 1
    print(f"\n{'='*60}")
    print(f"VALID:   {len(good)} items -> {MERGED.name}")
    print(f"REJECTED:{len(bad)} items")
    print(f"topics:  {dict(sorted(by_topic.items()))}")
    if bad:
        print(f"\n--- rejects ---")
        for fn, iid, errs in bad[:40]:
            print(f"  {fn} [{iid}]: {'; '.join(errs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
