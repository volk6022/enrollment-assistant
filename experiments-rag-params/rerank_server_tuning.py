"""Tune llama-server flags for the GGUF reranker (bge-reranker-v2-m3 Q8_0).

Starting point: 525 ms for 50 pairs at -np 4 -b 4096 -ub 1024, vs 289 ms for the
torch fp16 CrossEncoder it would replace. GGUF halves VRAM (510 vs 1083 MB) — the
question is how much of that 1.8x latency penalty is actually inherent and how much is
just bad flags.

Prior art from this repo (llama_local.py, found the hard way while tuning the judge
server): llama-server runs TWO caches that are pure overhead when every request is
unique — per-slot context checkpoints (-ctxcp, ~50 MB snapshots each) and a global
prompt cache (-cram, searches all cached prompts for a reusable prefix on every
request). Reranking 50 distinct documents NEVER has a reusable prefix, so both should
be dead weight here. That is stage 1.

Staged greedy search — each stage fixes the winner and moves on, so the run stays ~15
minutes instead of a full factorial:

  1. cache flags      the suspected big win
  2. -np              server slots (how many docs get scored concurrently)
  3. -b / -ub         logical / physical batch
  4. -c               context size (also a VRAM lever)
  5. misc             flash-attn, continuous batching, KV dtype

CORRECTNESS IS CHECKED EVERY TIME: each config's scores are Spearman-correlated against
the torch fp16 reference (runs/_gguf_ref.json) and the top-5 overlap is recorded. A
config that is fast but reorders the results is not a win, so any config below
SPEARMAN_FLOOR is rejected regardless of speed.

Also sweeps fused_top (30/50/100) at the end: THAT is the real lever on rerank cost —
rerank scores `fused_top` candidates regardless of final_top/top_k, so raising top_k
from 5 to 20 does not cost the reranker anything.

Run (nothing else on the GPU):
    .venv\\Scripts\\python.exe experiments-rag-params\\rerank_server_tuning.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(__file__).resolve().parent / "runs"
LLAMA_SERVER = r"C:\Users\bhunp\work-software\llama-cpp\llama-server.exe"
MODEL = REPO / "models" / "gguf" / "bge-reranker-v2-m3-q8_0.gguf"
REF = RUNS / "_gguf_ref.json"
OUT = RUNS / "rerank_server_tuning.json"
PORT = 20079
REPS = 3
SPEARMAN_FLOOR = 0.95     # below this the config is wrong, not fast

TORCH_FP16_MS = 289.0     # what we are trying to beat (runs/quant_vram.json)
TORCH_FP16_VRAM = 1083.0


def spearman(a, b) -> float:
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return round(num / (da * db), 4) if da and db else 0.0


def vram_mb() -> float:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1.0


ref = json.loads(REF.read_text(encoding="utf-8"))
QUERY, DOCS = ref["query"], ref["pairs_docs"]
REF_SCORES = ref["rerank_scores"]
REF_TOP5 = set(sorted(range(len(REF_SCORES)), key=lambda i: -REF_SCORES[i])[:5])
IDLE = vram_mb()


def run_config(label: str, flags: list[str], n_docs: int = 50) -> dict:
    """Start a server with `flags`, time `n_docs` reranks, verify ordering, stop."""
    row: dict = {"label": label, "flags": " ".join(flags), "n_docs": n_docs}
    args = [LLAMA_SERVER, "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
            "-ngl", "99", "--no-webui", "--reranking"] + flags
    RUNS.mkdir(parents=True, exist_ok=True)
    log = open(RUNS / "llama_rerank_tuning.log", "w", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=log, stderr=log)
    sess = requests.Session()
    sess.trust_env = False
    try:
        t0 = time.time()
        ok = False
        while time.time() - t0 < 180:
            if proc.poll() is not None:
                row["error"] = "server exited at startup"
                return row
            try:
                if sess.get(f"http://127.0.0.1:{PORT}/health",
                            timeout=2).json().get("status") == "ok":
                    ok = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.7)
        if not ok:
            row["error"] = "not healthy in 180s"
            return row

        docs = DOCS[:n_docs]
        body = {"query": QUERY, "documents": docs, "model": "x", "top_n": len(docs)}
        url = f"http://127.0.0.1:{PORT}/v1/rerank"
        r = sess.post(url, json=body, timeout=600)          # warm
        r.raise_for_status()
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = sess.post(url, json=body, timeout=600)
            times.append((time.perf_counter() - t0) * 1000)
        rows = r.json().get("results") or r.json().get("data") or []
        got = [0.0] * len(docs)
        for d in rows:
            got[d["index"]] = float(d["relevance_score"])

        row["ms_min"] = round(min(times), 1)
        row["ms_mean"] = round(sum(times) / len(times), 1)
        row["vram_mb"] = round(vram_mb() - IDLE, 1)
        row["spearman"] = spearman(got, REF_SCORES[:n_docs])
        top5 = set(sorted(range(len(got)), key=lambda i: -got[i])[:5])
        row["top5_overlap"] = len(top5 & set(i for i in REF_TOP5 if i < n_docs))
        row["ok"] = row["spearman"] >= SPEARMAN_FLOOR
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                proc.kill()
        log.close()
        time.sleep(1.5)
    return row


def show(row: dict) -> None:
    if "error" in row:
        print(f"  {row['label']:34s} FAILED: {row['error']}", flush=True)
        return
    flag = "" if row.get("ok") else "   <-- REJECTED (ordering changed)"
    print(f"  {row['label']:34s} {row['ms_min']:7.1f}ms  VRAM {row['vram_mb']:6.1f}MB  "
          f"spearman {row['spearman']:.4f}  top5 {row['top5_overlap']}/5{flag}", flush=True)


def best_of(rows: list[dict]) -> dict | None:
    good = [r for r in rows if r.get("ok") and "error" not in r]
    return min(good, key=lambda r: r["ms_min"]) if good else None


def main() -> None:
    all_results: dict[str, list[dict]] = {}
    print(f"idle VRAM {IDLE} MB | torch fp16 baseline: {TORCH_FP16_MS}ms / "
          f"{TORCH_FP16_VRAM}MB\n")

    # ---------------- stage 1: cache flags ----------------
    print("=== stage 1: cache flags (base -np 4 -b 4096 -ub 1024 -c 16384) ===")
    BASE = ["-np", "4", "-b", "4096", "-ub", "1024", "-c", "16384"]
    stage1 = [
        ("defaults (all caches on)", []),
        ("--no-cache-prompt", ["--no-cache-prompt"]),
        ("-cram 0", ["-cram", "0"]),
        ("no-cache-prompt + cram 0", ["--no-cache-prompt", "-cram", "0"]),
        ("+ ctxcp 0 cpent -1", ["--no-cache-prompt", "-cram", "0",
                                 "-ctxcp", "0", "-cpent", "-1"]),
        ("+ no-kv-unified", ["--no-cache-prompt", "-cram", "0", "-ctxcp", "0",
                              "-cpent", "-1", "--no-kv-unified"]),
    ]
    rows = []
    for label, extra in stage1:
        r = run_config(label, BASE + extra)
        r["stage"] = "cache"
        rows.append(r)
        show(r)
    all_results["stage1_cache"] = rows
    b = best_of(rows)
    CACHE = next(e for l, e in stage1 if l == b["label"]) if b else []
    print(f"  -> winner: {b['label'] if b else 'none'} ({b['ms_min'] if b else '-'}ms)\n")

    # ---------------- stage 2: -np ----------------
    print("=== stage 2: -np (context scaled to keep >=512 tok/slot) ===")
    rows = []
    for np_ in [1, 2, 4, 8, 16, 32]:
        ctx = max(8192, np_ * 1024)
        flags = ["-np", str(np_), "-b", "4096", "-ub", "1024", "-c", str(ctx)] + CACHE
        r = run_config(f"np={np_} (c={ctx})", flags)
        r["stage"] = "np"
        r["np"] = np_
        rows.append(r)
        show(r)
    all_results["stage2_np"] = rows
    b = best_of(rows)
    NP = b["np"] if b else 4
    CTX = max(8192, NP * 1024)
    print(f"  -> winner: np={NP} ({b['ms_min'] if b else '-'}ms)\n")

    # ---------------- stage 3: batches ----------------
    print(f"=== stage 3: -b / -ub (np={NP}) ===")
    rows = []
    for bsz, ub in [(1024, 256), (2048, 512), (4096, 1024), (8192, 2048), (8192, 4096)]:
        flags = ["-np", str(NP), "-b", str(bsz), "-ub", str(ub), "-c", str(CTX)] + CACHE
        r = run_config(f"b={bsz} ub={ub}", flags)
        r["stage"] = "batch"
        r["b"], r["ub"] = bsz, ub
        rows.append(r)
        show(r)
    all_results["stage3_batch"] = rows
    b = best_of(rows)
    B, UB = (b["b"], b["ub"]) if b else (4096, 1024)
    print(f"  -> winner: b={B} ub={UB} ({b['ms_min'] if b else '-'}ms)\n")

    # ---------------- stage 4: -c ----------------
    print(f"=== stage 4: -c (np={NP} b={B} ub={UB}) — also a VRAM lever ===")
    rows = []
    for ctx in [4096, 8192, 16384, 32768]:
        if ctx < NP * 512:
            continue          # would not fit one sequence per slot
        flags = ["-np", str(NP), "-b", str(B), "-ub", str(UB), "-c", str(ctx)] + CACHE
        r = run_config(f"c={ctx}", flags)
        r["stage"] = "ctx"
        r["c"] = ctx
        rows.append(r)
        show(r)
    all_results["stage4_ctx"] = rows
    b = best_of(rows)
    CTX_BEST = b["c"] if b else CTX
    print(f"  -> winner: c={CTX_BEST} ({b['ms_min'] if b else '-'}ms)\n")

    # ---------------- stage 5: misc ----------------
    print(f"=== stage 5: misc (np={NP} b={B} ub={UB} c={CTX_BEST}) ===")
    CORE = ["-np", str(NP), "-b", str(B), "-ub", str(UB), "-c", str(CTX_BEST)] + CACHE
    rows = []
    for label, extra in [
        ("(stage-4 winner)", []),
        ("-fa on", ["-fa", "on"]),
        ("-fa off", ["-fa", "off"]),
        ("--no-cont-batching", ["--no-cont-batching"]),
        ("-ctk q8_0 -ctv q8_0", ["-ctk", "q8_0", "-ctv", "q8_0"]),
        ("--pooling rank (explicit)", ["--pooling", "rank"]),
    ]:
        r = run_config(label, CORE + extra)
        r["stage"] = "misc"
        rows.append(r)
        show(r)
    all_results["stage5_misc"] = rows
    b = best_of(rows)
    MISC = next((e for l, e in [
        ("(stage-4 winner)", []), ("-fa on", ["-fa", "on"]), ("-fa off", ["-fa", "off"]),
        ("--no-cont-batching", ["--no-cont-batching"]),
        ("-ctk q8_0 -ctv q8_0", ["-ctk", "q8_0", "-ctv", "q8_0"]),
        ("--pooling rank (explicit)", ["--pooling", "rank"])] if l == b["label"]), []) if b else []
    print(f"  -> winner: {b['label'] if b else 'none'} ({b['ms_min'] if b else '-'}ms)\n")

    FINAL = CORE + MISC

    # ---------------- stage 6: fused_top — the real cost lever ----------------
    print(f"=== stage 6: fused_top sweep with the tuned config ===")
    rows = []
    for n in [10, 20, 30, 50]:
        r = run_config(f"fused_top={n}", FINAL, n_docs=n)
        r["stage"] = "fused_top"
        r["fused_top"] = n
        rows.append(r)
        show(r)
    all_results["stage6_fused_top"] = rows

    summary = {
        "torch_fp16_baseline": {"ms": TORCH_FP16_MS, "vram_mb": TORCH_FP16_VRAM},
        "final_flags": " ".join(FINAL),
        "final": next((r for r in all_results["stage6_fused_top"]
                       if r.get("fused_top") == 50), None),
        "stages": all_results,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    f = summary["final"]
    print("\n=== FINAL ===")
    print(f"  flags: {summary['final_flags']}")
    if f and "ms_min" in f:
        print(f"  50 pairs: {f['ms_min']}ms  VRAM {f['vram_mb']}MB  "
              f"(torch fp16: {TORCH_FP16_MS}ms / {TORCH_FP16_VRAM}MB)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
