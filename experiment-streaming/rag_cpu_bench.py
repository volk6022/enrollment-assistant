"""Can bge-m3 (embedder) and bge-reranker-v2-m3 move to CPU to free VRAM?

Context: the streaming findings (docs/streaming-research-findings.md) concluded that GPU
is the project's scarce resource — whisper MUST be on GPU, and LLM generation competes
with it. The embedder and reranker are the two other GPU residents (~2.5 GB combined in
FP16). If they run acceptably on CPU, that VRAM goes back to whisper + a bigger KV cache.

Measured against the real production shapes from rag/config.py:
  embedder  bge-m3, `embed_query` = 1 short query; 2 queries in conversational
            multi-query mode; batch of 32 as an indexing-ish reference
  reranker  bge-reranker-v2-m3 CrossEncoder, max_length=256, batch_size=32,
            fused_top=50 pairs in prod (30 pairs also run, to compare against the
            recorded "254ms fp32 -> 93ms fp16 for 30 pairs on the 3060 Ti")

Real corpus chunks are used as reranker candidates, not synthetic strings — cross-encoder
cost is driven by actual token length.

Search-stage budget on record: < 1 s total, of which retrieval+rerank was ~354 ms on GPU.

Also probes GIL behaviour (same method as gil_probe.py) for the CPU variants, since on CPU
they would be sharing the main process with the event loop.

Run:  .venv\\Scripts\\python.exe experiment-streaming\\rag_cpu_bench.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO / "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are cached; no proxy needed

import sys  # noqa: E402
sys.path.insert(0, str(REPO))

from gil_probe import Probe, TIGHT_TICK, TIMER_TICK  # noqa: E402

CHUNKS = REPO / "rag" / "artifacts" / "chunks.jsonl"
OUT = Path(__file__).resolve().parent / "runs" / "rag_cpu_bench.json"

QUERY = "Какие документы нужны для поступления и до какого числа их принимают?"
QUERY_CANON = "Перечень документов для приёма и сроки приёма документов"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LEN = 256
RERANK_BATCH = 32
REPS = 3


def load_candidates(n: int) -> list[str]:
    """Real chunks from the built index — representative token lengths."""
    texts = []
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line).get("text", "").strip()
            if t:
                texts.append(t)
            if len(texts) >= n:
                break
    return texts


def bench(fn, reps: int = REPS) -> dict:
    fn()  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"ms_mean": round(statistics.fmean(ts) * 1000, 1),
            "ms_min": round(min(ts) * 1000, 1),
            "ms_max": round(max(ts) * 1000, 1)}


# --------------------------------------------------------------------------- #
# model builders
# --------------------------------------------------------------------------- #

def build_embedder(device: str, fp16: bool, threads: int | None):
    import torch
    if device == "cpu" and threads:
        torch.set_num_threads(threads)
    from sentence_transformers import SentenceTransformer
    mk = {}
    if fp16 and device != "cpu":
        mk["torch_dtype"] = "float16"
    return SentenceTransformer(EMBED_MODEL, device=device, model_kwargs=mk or None)


def build_reranker(device: str, fp16: bool, threads: int | None, int8: bool = False):
    import torch
    if device == "cpu" and threads:
        torch.set_num_threads(threads)
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(RERANK_MODEL, device=device, max_length=RERANK_MAX_LEN)
    if fp16 and device == "cuda":
        m.model.half()
    if int8 and device == "cpu":
        # dynamic int8 on the Linear layers — the cheap CPU quantisation route
        m.model = torch.quantization.quantize_dynamic(
            m.model, {torch.nn.Linear}, dtype=torch.qint8)
    return m


# --------------------------------------------------------------------------- #
# GIL probe around a CPU workload
# --------------------------------------------------------------------------- #

async def gil_check(fn) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    tight, timer = Probe("t", TIGHT_TICK), Probe("m", TIMER_TICK)
    tt, tm = asyncio.create_task(tight.run()), asyncio.create_task(timer.run())
    await asyncio.sleep(0.15)
    pool = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    for _ in range(REPS):
        await loop.run_in_executor(pool, fn)
    pool.shutdown(wait=True)
    tight.stop(); timer.stop()
    await asyncio.gather(tt, tm)
    ts, ms = tight.stats(), timer.stats()
    return {"tight_max_ms": ts.get("gap_ms_max"), "tight_p99_ms": ts.get("gap_ms_p99"),
            "timer_max_ms": ms.get("gap_ms_max"),
            "verdict": ("releases GIL" if (ts.get("gap_ms_max") or 0) < 45
                        else "HOLDS GIL")}


# --------------------------------------------------------------------------- #

async def main() -> None:
    cands = load_candidates(50)
    lens = [len(c) for c in cands]
    print(f"candidates: {len(cands)} real chunks, chars mean "
          f"{statistics.fmean(lens):.0f} / max {max(lens)}\n")

    results: dict = {"candidate_chars_mean": round(statistics.fmean(lens)),
                     "reps": REPS, "embed": {}, "rerank": {}}

    import torch
    n_threads_default = torch.get_num_threads()
    print(f"torch default threads: {n_threads_default}\n")
    results["torch_default_threads"] = n_threads_default

    # ---------------- embedder ----------------
    embed_cfgs = [
        ("cuda/fp16 (prod)", "cuda", True, None),
        ("cpu/fp32, threads=default", "cpu", False, None),
        ("cpu/fp32, threads=12", "cpu", False, 12),
    ]
    for label, dev, fp16, thr in embed_cfgs:
        try:
            m = build_embedder(dev, fp16, thr)
            enc = lambda ts: m.encode(ts, batch_size=32, convert_to_numpy=True,  # noqa: E731
                                      normalize_embeddings=True, show_progress_bar=False)
            row = {
                "q1": bench(lambda: enc([QUERY])),
                "q2_conversational": bench(lambda: enc([QUERY, QUERY_CANON])),
                "batch32": bench(lambda: enc(cands[:32])),
            }
            if dev == "cpu":
                row["gil_q1"] = await gil_check(lambda: enc([QUERY]))
            results["embed"][label] = row
            print(f"EMBED {label:28s} 1q={row['q1']['ms_mean']:7.1f}ms  "
                  f"2q={row['q2_conversational']['ms_mean']:7.1f}ms  "
                  f"b32={row['batch32']['ms_mean']:8.1f}ms"
                  + (f"  [{row['gil_q1']['verdict']}, max {row['gil_q1']['tight_max_ms']}ms]"
                     if "gil_q1" in row else ""))
            del m
        except Exception as exc:  # noqa: BLE001
            results["embed"][label] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"EMBED {label:28s} FAILED: {type(exc).__name__}: {exc}")
        if dev == "cuda":
            torch.cuda.empty_cache()

    print()

    # ---------------- reranker ----------------
    rr_cfgs = [
        ("cuda/fp16 (prod)", "cuda", True, None, False),
        ("cpu/fp32, threads=default", "cpu", False, None, False),
        ("cpu/fp32, threads=12", "cpu", False, 12, False),
        ("cpu/int8-dynamic, threads=12", "cpu", False, 12, True),
    ]
    for label, dev, fp16, thr, int8 in rr_cfgs:
        try:
            m = build_reranker(dev, fp16, thr, int8)
            pred = lambda pairs: m.predict(pairs, batch_size=RERANK_BATCH,  # noqa: E731
                                           show_progress_bar=False)
            p50 = [(QUERY, c) for c in cands]
            row = {
                "pairs50_prod": bench(lambda: pred(p50)),
                "pairs30_ref": bench(lambda: pred(p50[:30])),
                "pairs10": bench(lambda: pred(p50[:10])),
            }
            if dev == "cpu":
                row["gil_pairs50"] = await gil_check(lambda: pred(p50))
            results["rerank"][label] = row
            print(f"RERANK {label:28s} 50={row['pairs50_prod']['ms_mean']:8.1f}ms  "
                  f"30={row['pairs30_ref']['ms_mean']:8.1f}ms  "
                  f"10={row['pairs10']['ms_mean']:7.1f}ms"
                  + (f"  [{row['gil_pairs50']['verdict']}, max "
                     f"{row['gil_pairs50']['tight_max_ms']}ms]"
                     if "gil_pairs50" in row else ""))
            del m
        except Exception as exc:  # noqa: BLE001
            results["rerank"][label] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"RERANK {label:28s} FAILED: {type(exc).__name__}: {exc}")
        if dev == "cuda":
            torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
