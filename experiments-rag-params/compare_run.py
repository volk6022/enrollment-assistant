"""Run the 100-question pool through the CURRENTLY-BUILT index, N repeats each,
capturing answers + timing for side-by-side (3-column) quality review.

Usage:  uv run python experiments-rag-params/compare_run.py <variant-label> [repeats]
Build the index for the variant FIRST (rag.ingest / build_wiki_index + rag.index),
then run this. Writes runs/compare_<label>_<ts>.json.

Do this 3x (baseline / kb-relevant / kb-full), rebuilding the index between each.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from rag.config import DEFAULT
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.generate import LlamaServer

RUNS = Path(__file__).resolve().parent / "runs"
POOL = Path(__file__).resolve().parent / "eval_set_100.json"

def pct(vals, p):
    s = sorted(vals); return round(s[min(len(s)-1, int(len(s)*p))], 1)

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "variant"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    pool = json.load(open(POOL, encoding="utf-8"))

    idx = Indexes()
    log_path = RUNS / f"llama_compare_{label}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    RUNS.mkdir(exist_ok=True)
    server = LlamaServer(DEFAULT); server.start(log_path=str(log_path)); time.sleep(2)
    print(f"[compare_run] llm_gguf={DEFAULT.llm_gguf}")
    print(f"[compare_run] server log -> {log_path}")
    pipe = Pipeline(idx, server=server, cfg=DEFAULT)
    t_warm = time.time()
    pipe.answer(pool[0]["question"])  # warmup (discarded)
    print(f"[compare_run] warmup ok in {time.time() - t_warm:.1f}s")

    rows, all_search, all_gen, all_total, all_tok = [], [], [], [], []
    try:
        t_loop = time.time()
        for qi, q in enumerate(pool):
            runs = []
            for _ in range(repeats):
                r = pipe.answer(q["question"])
                t = r["timings"]; gen = r.get("gen") or {}
                s_ms = round(t.get("search_ms", 0), 1); g_ms = round(t.get("gen_ms", 0), 1)
                runs.append({"answer": r["answer"], "search_ms": s_ms, "gen_ms": g_ms,
                             "total_ms": round(s_ms + g_ms, 1), "gen_tokens": gen.get("completion_tokens"),
                             "citations": r.get("citations")})
                all_search.append(s_ms); all_gen.append(g_ms); all_total.append(s_ms + g_ms)
                all_tok.append(gen.get("completion_tokens") or 0)
            answers = [x["answer"] for x in runs]
            rows.append({
                "id": q["id"], "question": q["question"], "topic": q.get("topic", ""),
                "source_doc": q.get("source_doc", ""), "reference": q.get("reference", ""),
                "answers": answers,                      # <- the 3 (or N) answers for column view
                "stable": len(set(a.strip() for a in answers)) == 1,
                "runs": runs,
            })
            if (qi + 1) % 20 == 0:
                print(f"    {qi + 1}/{len(pool)}  ({time.time() - t_loop:.0f}s)")
    finally:
        server.stop()

    n = len(all_total)
    agg = {
        "n_questions": len(pool), "repeats": repeats, "n_calls": n,
        "search_avg_s": round(sum(all_search)/n/1000, 3), "search_p90_s": round(pct(all_search, 0.9)/1000, 3),
        "gen_avg_s": round(sum(all_gen)/n/1000, 3), "gen_p90_s": round(pct(all_gen, 0.9)/1000, 3),
        "total_avg_s": round(sum(all_total)/n/1000, 3), "total_p90_s": round(pct(all_total, 0.9)/1000, 3),
        "avg_gen_tokens": round(sum(all_tok)/n, 1),
        "stable_pct": round(100*sum(1 for r in rows if r["stable"])/len(rows), 1),
    }
    out = {"meta": {"label": label, "index_chunks": len(idx.chunks),
                    "model": DEFAULT.llm_gguf, "conversational": False,
                    "ts": time.strftime("%Y%m%d_%H%M%S")},
           "aggregate": agg, "rows": rows}

    RUNS.mkdir(exist_ok=True)
    fn = RUNS / f"compare_{label}_{out['meta']['ts']}.json"
    fn.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {label}  chunks={len(idx.chunks)}  -> {fn.name}")
    print(f"  search {agg['search_avg_s']}s avg / {agg['search_p90_s']}s p90")
    print(f"  gen    {agg['gen_avg_s']}s avg / {agg['gen_p90_s']}s p90   tokens~{agg['avg_gen_tokens']}")
    print(f"  total  {agg['total_avg_s']}s avg / {agg['total_p90_s']}s p90")
    print(f"  stable {agg['stable_pct']}% (identical across {repeats} repeats)")

if __name__ == "__main__":
    main()
