"""Statistical reading of the grid: which differences are real and which are noise.

The raw aggregate ranks top_k=15 best on answer_score (0.783) and top_k=5 worst (0.735),
with the three backends within 0.021 of each other. Those gaps are small relative to the
spread of a 0..1 score over 100-300 samples, so reporting the ranking as-is would be
overclaiming.

Two things done here:

  1. Unpaired CIs — standard error of each cell/margin mean, so the size of the noise
     floor is explicit.
  2. PAIRED comparisons — the same 100 questions run through every cell, so comparing
     cells question-by-question removes question-difficulty variance entirely. This is
     far more sensitive than comparing the marginal means, and it is the honest test of
     "is top_k=15 actually better than top_k=10".

Parse failures (judge did not emit valid JSON) are excluded pairwise, and counted.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

data = json.loads((RUNS / "grid_judge_results.json").read_text(encoding="utf-8"))
res = data["results"]
BACKENDS = ["torch-fp16", "gguf-q8_0", "gguf-q4_k_m"]
TOP_KS = [5, 10, 15, 20]

by_key = {(r["backend"], r["top_k"], r["id"]): r for r in res}
qids = sorted({r["id"] for r in res})


def vals(pred, field="answer_score"):
    return [r[field] for r in res if pred(r) and isinstance(r[field], (int, float))]


def mean_se(xs):
    if not xs:
        return None, None
    m = statistics.fmean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, se


print("=== доля непарсящихся вердиктов по ячейкам ===")
for b in BACKENDS:
    row = []
    for k in TOP_KS:
        n_bad = sum(1 for r in res if r["backend"] == b and r["top_k"] == k
                    and not isinstance(r["answer_score"], (int, float)))
        row.append(f"k{k}={n_bad}")
    print(f"  {b:14s} " + "  ".join(row))

for field in ("answer_score", "chunks_score"):
    print(f"\n=== {field}: средние ± 1 SE ===")
    print("  по top_k:")
    for k in TOP_KS:
        m, se = mean_se(vals(lambda r, k=k: r["top_k"] == k, field))
        print(f"    k={k:<3d} {m:.3f} ± {se:.3f}   (95% ДИ {m-1.96*se:.3f}..{m+1.96*se:.3f})")
    print("  по бэкенду:")
    for b in BACKENDS:
        m, se = mean_se(vals(lambda r, b=b: r["backend"] == b, field))
        print(f"    {b:14s} {m:.3f} ± {se:.3f}   (95% ДИ {m-1.96*se:.3f}..{m+1.96*se:.3f})")


def paired(sel_a, sel_b, field="answer_score"):
    """Paired diff over the questions where BOTH cells produced a valid score."""
    d = []
    for q in qids:
        va, vb = [], []
        for b in BACKENDS:
            for k in TOP_KS:
                r = by_key.get((b, k, q))
                if not r or not isinstance(r[field], (int, float)):
                    continue
                if sel_a(b, k):
                    va.append(r[field])
                if sel_b(b, k):
                    vb.append(r[field])
        if va and vb:
            d.append(statistics.fmean(va) - statistics.fmean(vb))
    if len(d) < 2:
        return None
    m = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    t = m / se if se else 0.0
    return {"n": len(d), "diff": m, "se": se, "t": t,
            "ci": (m - 1.96 * se, m + 1.96 * se), "sig": abs(t) >= 1.96}


def show(title, cmp):
    if cmp is None:
        print(f"  {title:38s} — недостаточно данных")
        return
    verdict = "ЗНАЧИМО" if cmp["sig"] else "в пределах шума"
    print(f"  {title:38s} Δ={cmp['diff']:+.4f} ± {cmp['se']:.4f}  "
          f"ДИ [{cmp['ci'][0]:+.4f}, {cmp['ci'][1]:+.4f}]  t={cmp['t']:+.2f}  {verdict}")


print("\n=== ПАРНЫЕ сравнения по top_k (усреднено по бэкендам, answer_score) ===")
for a, b in [(10, 5), (15, 5), (20, 5), (15, 10), (20, 10), (20, 15)]:
    show(f"top_k={a} против top_k={b}",
         paired(lambda bk, k, a=a: k == a, lambda bk, k, b=b: k == b))

print("\n=== ПАРНЫЕ сравнения по top_k (chunks_score) ===")
for a, b in [(10, 5), (15, 10), (20, 15)]:
    show(f"top_k={a} против top_k={b}",
         paired(lambda bk, k, a=a: k == a, lambda bk, k, b=b: k == b, "chunks_score"))

print("\n=== ПАРНЫЕ сравнения бэкендов (усреднено по top_k, answer_score) ===")
for a, b in [("gguf-q8_0", "torch-fp16"), ("gguf-q4_k_m", "torch-fp16"),
             ("gguf-q4_k_m", "gguf-q8_0")]:
    show(f"{a} против {b}",
         paired(lambda bk, k, a=a: bk == a, lambda bk, k, b=b: bk == b))

print("\n=== ПАРНЫЕ сравнения бэкендов (chunks_score) ===")
for a, b in [("gguf-q8_0", "torch-fp16"), ("gguf-q4_k_m", "torch-fp16")]:
    show(f"{a} против {b}",
         paired(lambda bk, k, a=a: bk == a, lambda bk, k, b=b: bk == b, "chunks_score"))

print("\n=== top_k внутри каждого бэкенда (answer_score, парно) ===")
for bk in BACKENDS:
    print(f"  {bk}:")
    for a, b in [(10, 5), (15, 10), (20, 15)]:
        show(f"    k={a} против k={b}",
             paired(lambda x, k, bk=bk, a=a: x == bk and k == a,
                    lambda x, k, bk=bk, b=b: x == bk and k == b))
