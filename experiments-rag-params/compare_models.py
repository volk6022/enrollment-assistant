"""Run the 100-question pool through the CURRENTLY-BUILT index, generating each
answer with 3 OpenRouter models at temperature=0 (single run per question).

Retrieval (embed+rerank) is LOCAL on GPU and done ONCE per question; the same
context/prompt is then sent to each of the 3 models. Output: one JSON per index
variant, each row holding 3 answers (one per model) — for side-by-side review.

Usage:
  uv run python experiments-rag-params/compare_models.py <label>          # full 100
  uv run python experiments-rag-params/compare_models.py smoke --smoke     # 2 Qs, 3 models

Needs env OPENROUTER_API_KEY. Network goes through the box SOCKS proxy.
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import requests
from rag.config import DEFAULT
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.generate import build_messages

RUNS = Path(__file__).resolve().parent / "runs"
POOL = Path(__file__).resolve().parent / "eval_set_100.json"

MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.6-27b", "deepseek/deepseek-v4-flash"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
PROXY = os.getenv("ALL_PROXY", "socks5h://127.0.0.1:10808")


def or_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.proxies = {"http": PROXY, "https": PROXY}
    return s


def or_complete(sess, messages, model, api_key, temperature=0.0, max_tokens=1200, retries=3):
    # reasoning.enabled=false -> turn OFF thinking (these Qwen slugs are reasoning
    # models; otherwise the token budget is spent on hidden reasoning and content
    # comes back empty). No thinking is also what we want for a voice assistant.
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens, "stream": False,
               "reasoning": {"enabled": False}}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = sess.post(OR_URL, json=payload, headers=headers, timeout=(10, 240))
            dt = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                ch = (data.get("choices") or [{}])[0]
                msg = ch.get("message", {}) or {}
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning") or "")
                # fallback: if content empty but the model returned reasoning, some
                # providers put the answer there — use its tail rather than nothing.
                if not content and reasoning:
                    content = reasoning.strip()
                usage = data.get("usage", {}) or {}
                return {"answer": content, "gen_ms": round(dt * 1000, 1),
                        "completion_tokens": usage.get("completion_tokens"),
                        "finish": ch.get("finish_reason"),
                        "had_reasoning": bool(reasoning), "error": None}
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 * (attempt + 1)); continue
            return {"answer": "", "gen_ms": round(dt * 1000, 1), "error": last}
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return {"answer": "", "gen_ms": None, "error": last}


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "variant"
    smoke = "--smoke" in sys.argv
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("!! OPENROUTER_API_KEY not set"); sys.exit(2)

    pool = json.load(open(POOL, encoding="utf-8"))
    if smoke:
        pool = pool[:2]
    sessions = {m: or_session() for m in MODELS}  # one Session per model (thread-safe use)

    idx = Indexes()
    pipe = Pipeline(idx, server=None, cfg=DEFAULT)  # no local gen; retrieval only
    pipe.search(pool[0]["question"])  # warmup: loads embedder+reranker (one-time)

    rows = []
    tot_search, per_model_ms = 0.0, {m: [] for m in MODELS}
    for qi, q in enumerate(pool):
        top, t = pipe.search(q["question"])
        s_ms = round(t.get("search_ms", 0.0), 1); tot_search += s_ms
        messages = build_messages(q["question"], top, DEFAULT)
        answers, gen = {}, {}
        with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
            futs = {m: ex.submit(or_complete, sessions[m], messages, m, api_key) for m in MODELS}
            results = {m: f.result() for m, f in futs.items()}
        for m in MODELS:
            res = results[m]
            answers[m] = res["answer"] if not res.get("error") else f"[ERROR] {res['error']}"
            gen[m] = {"gen_ms": res.get("gen_ms"), "tokens": res.get("completion_tokens"),
                      "finish": res.get("finish"), "error": res.get("error")}
            if res.get("gen_ms"): per_model_ms[m].append(res["gen_ms"])
            if res.get("error"): print(f"  [{label}] q{qi} {m} ERROR: {res['error'][:120]}")
        rows.append({
            "id": q["id"], "question": q["question"], "topic": q.get("topic", ""),
            "source_doc": q.get("source_doc", ""), "reference": q.get("reference", ""),
            "answers": answers, "gen": gen, "search_ms": s_ms,
            "citations": [{"source": c["source"], "point": c.get("point")} for c in top],
        })
        if smoke:
            for m in MODELS:
                g = gen[m]
                print(f"\n--- {m}  [finish={g['finish']} tok={g['tokens']}]\n{answers[m][:400]}")

    agg = {"n_questions": len(pool), "models": MODELS,
           "search_avg_s": round(tot_search / len(pool) / 1000, 3),
           "gen_avg_s": {m: (round(sum(v) / len(v) / 1000, 3) if v else None)
                         for m, v in per_model_ms.items()},
           "errors": {m: sum(1 for r in rows if r["gen"][m]["error"]) for m in MODELS}}
    out = {"meta": {"label": label, "index_chunks": len(idx.chunks),
                    "temperature": 0.0, "single_run": True,
                    "ts": time.strftime("%Y%m%d_%H%M%S")},
           "aggregate": agg, "rows": rows}

    if smoke:
        print("\n=== SMOKE OK ===", json.dumps(agg, ensure_ascii=False)); return
    RUNS.mkdir(exist_ok=True)
    fn = RUNS / f"models_{label}_{out['meta']['ts']}.json"
    fn.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"=== {label}  chunks={len(idx.chunks)}  -> {fn.name}")
    print(f"  search {agg['search_avg_s']}s avg | gen(s) {agg['gen_avg_s']} | errors {agg['errors']}")


if __name__ == "__main__":
    main()
