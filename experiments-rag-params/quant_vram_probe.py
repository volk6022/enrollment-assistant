"""Resident VRAM + latency of bge-m3 / bge-reranker under fp16 vs int8 (bitsandbytes).

Motivation: the full streaming stack does not fit in 8 GB (see
docs/streaming-research-findings.md §7), so the question is whether int8 weights buy
back enough VRAM to matter.

An earlier smoke test measured `max_memory_allocated` from process start, which conflates
the LOAD transient (fp16 path materialises fp32 weights before `.half()`) with what the
model actually holds afterwards. Resident size is what determines whether the stack fits,
so it is measured here separately from the inference peak:

  resident_mb   allocated after load + empty_cache      -> does the stack fit?
  infer_peak_mb max_memory_allocated with stats reset   -> how much headroom inference needs
                AFTER load                                 (this is the reranker's "big cache")

Each variant runs in its OWN process (VARIANT env var) so nothing leaks between them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "runs" / "quant_vram.json"

QUERY = "Какие документы нужны для поступления и до какого числа их принимают?"
VARIANTS = ["embed-fp16", "embed-int8", "rerank-fp16", "rerank-int8"]
RERANK_BATCHES = [8, 16, 32, 64]


def child() -> None:
    os.environ.setdefault("HF_HOME", str(REPO / "models"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch

    variant = os.environ["VARIANT"]
    kind, prec = variant.split("-")
    out: dict = {"variant": variant}

    def mb(x):
        return round(x / 1024 ** 2, 1)

    if kind == "embed":
        from sentence_transformers import SentenceTransformer
        mk = {}
        if prec == "fp16":
            mk["torch_dtype"] = "float16"
        else:
            from transformers import BitsAndBytesConfig
            mk["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        t0 = time.perf_counter()
        m = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs=mk or None)
        out["load_s"] = round(time.perf_counter() - t0, 1)

        torch.cuda.empty_cache()
        out["resident_mb"] = mb(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()          # measure inference peak only

        enc = lambda ts, bs=32: m.encode(ts, normalize_embeddings=True,  # noqa: E731
                                         convert_to_numpy=True, batch_size=bs)
        enc([QUERY])                                   # warm
        t0 = time.perf_counter()
        for _ in range(5):
            enc([QUERY])
        out["q1_ms"] = round((time.perf_counter() - t0) / 5 * 1000, 1)
        # corpus-shaped batch, the reindex path
        texts = [QUERY] * 32
        t0 = time.perf_counter()
        enc(texts)
        out["batch32_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["infer_peak_mb"] = mb(torch.cuda.max_memory_allocated())
        v = enc([QUERY])
        np.save(Path(os.environ["VEC_OUT"]), v)

    else:
        from sentence_transformers import CrossEncoder
        kw = {}
        if prec == "int8":
            from transformers import BitsAndBytesConfig
            kw["model_kwargs"] = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        t0 = time.perf_counter()
        m = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=256, **kw)
        if prec == "fp16":
            m.model.half()
        out["load_s"] = round(time.perf_counter() - t0, 1)

        torch.cuda.empty_cache()
        out["resident_mb"] = mb(torch.cuda.memory_allocated())

        # real corpus chunks -> realistic token lengths
        chunks = []
        with (REPO / "rag" / "artifacts" / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                t = json.loads(line).get("text", "").strip()
                if t:
                    chunks.append(t)
                if len(chunks) >= 50:
                    break
        pairs = [(QUERY, c) for c in chunks]

        m.predict(pairs, batch_size=32, show_progress_bar=False)  # warm
        by_batch = {}
        for bs in RERANK_BATCHES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()       # per-batch inference peak
            t0 = time.perf_counter()
            s = m.predict(pairs, batch_size=bs, show_progress_bar=False)
            ms = (time.perf_counter() - t0) * 1000
            by_batch[bs] = {
                "ms_50pairs": round(ms, 1),
                "infer_peak_mb": mb(torch.cuda.max_memory_allocated()),
                "activation_mb": mb(torch.cuda.max_memory_allocated()) - out["resident_mb"],
                "scores_sha": hash(tuple(round(float(x), 6) for x in s)),
                "scores_head": [round(float(x), 6) for x in s[:5]],
            }
        out["by_batch"] = by_batch

    print("RESULT " + json.dumps(out, ensure_ascii=False))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    vec_paths = {}
    for v in VARIANTS:
        env = {**os.environ, "VARIANT": v, "CHILD": "1",
               "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        vec_out = OUT.parent / f"_vec_{v}.npy"
        env["VEC_OUT"] = str(vec_out)
        vec_paths[v] = vec_out
        print(f"--- {v} ---", flush=True)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                           env=env, capture_output=True, text=True, encoding="utf-8")
        line = next((l for l in (r.stdout or "").splitlines() if l.startswith("RESULT ")), None)
        if line:
            row = json.loads(line[len("RESULT "):])
            results.append(row)
            print("   " + json.dumps(row, ensure_ascii=False), flush=True)
        else:
            err = (r.stderr or "")[-400:]
            results.append({"variant": v, "error": err})
            print(f"   FAILED: {err}", flush=True)

    # cosine agreement fp16 vs int8 embeddings
    try:
        import numpy as np
        a = np.load(vec_paths["embed-fp16"]); b = np.load(vec_paths["embed-int8"])
        cos = float((a * b).sum(axis=1).mean())
        print(f"\ncosine(embed fp16, embed int8) = {cos:.4f}")
        results.append({"cosine_fp16_int8": round(cos, 4)})
    except Exception as e:  # noqa: BLE001
        print(f"cosine check skipped: {e}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    if os.environ.get("CHILD"):
        child()
    else:
        main()
