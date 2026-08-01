"""Are the GGUF quants of bge-m3 / bge-reranker-v2-m3 usable via llama-server?

Motivation: bitsandbytes int8 turned out WORSE than fp16 on these 568M encoders
(more resident VRAM, 3-7x slower — see runs/quant_vram.json). GGUF quants are a
different mechanism and much more promising: 417-606 MB on disk vs 1083 MB resident
for fp16 torch, and it moves both encoders out of the Python process entirely, behind
an HTTP API (which is the architecture we want anyway).

But a GGUF conversion can be silently wrong — especially for a reranker, where the
classification head must survive the conversion. So nothing is assumed: every quant is
checked against the torch fp16 model that currently runs in production.

  embedder   cosine(gguf_vec, torch_fp16_vec) per text; also Spearman of the
             pairwise similarity structure, which is what retrieval actually uses
  reranker   Spearman + top-5 overlap of the ranking vs torch fp16 on real chunks,
             which is what the pipeline consumes — absolute scores may be on a
             different scale and that is fine, the ORDER is what matters

Also records resident VRAM (nvidia-smi, whole process) and latency, to compare against
fp16 torch: embed 1083 MB / 25.6 ms per query, rerank 1083 MB / 289.9 ms per 50 pairs.

Two phases in separate processes so torch and llama-server never share VRAM:
    python gguf_encoders_probe.py ref     # torch fp16 reference -> runs/_gguf_ref.json
    python gguf_encoders_probe.py gguf    # start each GGUF server, compare
    python gguf_encoders_probe.py         # both, in order
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(__file__).resolve().parent / "runs"
GGUF_DIR = REPO / "models" / "gguf"
LLAMA_SERVER = r"C:\Users\bhunp\work-software\llama-cpp\llama-server.exe"
REF_PATH = RUNS / "_gguf_ref.json"
OUT = RUNS / "gguf_encoders.json"
PORT = 20077

QUERY = "Какие документы нужны для поступления и до какого числа их принимают?"
N_TEXTS = 24      # texts to embed
N_PAIRS = 50      # reranker candidates — the production fused_top

EMBED_GGUFS = {"gguf-q8_0": "bge-m3-q8_0.gguf", "gguf-q4_k_m": "bge-m3-q4_k_m.gguf"}
RERANK_GGUFS = {"gguf-q8_0": "bge-reranker-v2-m3-q8_0.gguf",
                "gguf-q4_k_m": "bge-reranker-v2-m3-q4_k_m.gguf"}


def load_chunks(n: int) -> list[str]:
    out = []
    with (REPO / "rag" / "artifacts" / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line).get("text", "").strip()
            if t:
                out.append(t)
            if len(out) >= n:
                break
    return out


def vram_mb() -> float:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1.0


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without scipy (scipy is present, but this keeps the probe
    dependency-free and the ranks are what we care about)."""
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


# --------------------------------------------------------------------------- #
# phase 1: torch fp16 reference
# --------------------------------------------------------------------------- #

def cmd_ref() -> None:
    os.environ.setdefault("HF_HOME", str(REPO / "models"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(REPO))
    import numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder

    texts = load_chunks(N_TEXTS)
    pairs_docs = load_chunks(N_PAIRS)

    m = SentenceTransformer("BAAI/bge-m3", device="cuda",
                            model_kwargs={"torch_dtype": "float16"})
    vecs = m.encode([QUERY] + texts, normalize_embeddings=True, convert_to_numpy=True,
                    batch_size=16)
    t0 = time.perf_counter()
    for _ in range(5):
        m.encode([QUERY], normalize_embeddings=True, convert_to_numpy=True)
    embed_ms = (time.perf_counter() - t0) / 5 * 1000
    del m
    import torch
    torch.cuda.empty_cache()

    ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=256)
    ce.model.half()
    pairs = [(QUERY, d) for d in pairs_docs]
    ce.predict(pairs, batch_size=32, show_progress_bar=False)
    t0 = time.perf_counter()
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    rerank_ms = (time.perf_counter() - t0) * 1000

    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    REF_PATH.write_text(json.dumps({
        "query": QUERY, "texts": texts, "pairs_docs": pairs_docs,
        "vecs": [[round(float(x), 6) for x in v] for v in vecs],
        "rerank_scores": [float(x) for x in scores],
        "embed_ms": round(embed_ms, 1), "rerank_ms": round(rerank_ms, 1),
    }, ensure_ascii=False), encoding="utf-8")
    print(f"[ref] torch fp16: embed 1q {embed_ms:.1f}ms, rerank {N_PAIRS} pairs "
          f"{rerank_ms:.1f}ms -> {REF_PATH.name}")


# --------------------------------------------------------------------------- #
# phase 2: GGUF via llama-server
# --------------------------------------------------------------------------- #

class Server:
    def __init__(self, model: Path, mode: str) -> None:
        self.model, self.mode = model, mode
        self.proc = None
        self.base = f"http://127.0.0.1:{PORT}"

    def __enter__(self):
        args = [LLAMA_SERVER, "-m", str(self.model), "--host", "127.0.0.1",
                "--port", str(PORT), "-ngl", "99", "--no-webui", "-c", "8192"]
        args += ["--embedding"] if self.mode == "embed" else ["--reranking"]
        log = open(RUNS / f"llama_gguf_{self.model.stem}.log", "w", encoding="utf-8")
        self._log = log
        self.proc = subprocess.Popen(args, stdout=log, stderr=log)
        import requests
        self.sess = requests.Session()
        self.sess.trust_env = False
        t0 = time.time()
        while time.time() - t0 < 180:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited; see {log.name}")
            try:
                r = self.sess.get(f"{self.base}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return self
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
        raise TimeoutError("server not healthy")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self._log.close()
        time.sleep(2)


def cmd_gguf() -> None:
    import requests  # noqa: F401
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    ref_vecs = ref["vecs"]
    results: dict = {"reference": {"embed_ms": ref["embed_ms"], "rerank_ms": ref["rerank_ms"],
                                    "resident_mb_fp16_torch": 1083.0}}
    idle = vram_mb()
    print(f"idle VRAM: {idle} MB\n")

    # ---- embedders ----
    for label, fname in EMBED_GGUFS.items():
        path = GGUF_DIR / fname
        row: dict = {"file": fname, "size_mb": round(path.stat().st_size / 1024**2, 1)}
        try:
            with Server(path, "embed") as s:
                row["vram_mb"] = round(vram_mb() - idle, 1)
                texts = [ref["query"]] + ref["texts"]
                t0 = time.perf_counter()
                r = s.sess.post(f"{s.base}/v1/embeddings",
                                json={"input": texts, "model": "x"}, timeout=180)
                r.raise_for_status()
                got = [d["embedding"] for d in r.json()["data"]]
                row["batch_ms"] = round((time.perf_counter() - t0) * 1000, 1)

                t0 = time.perf_counter()
                for _ in range(5):
                    s.sess.post(f"{s.base}/v1/embeddings",
                                json={"input": [ref["query"]], "model": "x"}, timeout=60)
                row["q1_ms"] = round((time.perf_counter() - t0) / 5 * 1000, 1)

                row["dim"] = len(got[0]) if got else None
                # cosine vs torch fp16, per text (vectors are L2-normalised on both sides;
                # normalise defensively in case llama.cpp's --embd-normalize differs)
                def norm(v):
                    n = sum(x * x for x in v) ** 0.5
                    return [x / n for x in v] if n else v
                cos = [sum(a * b for a, b in zip(norm(g), rv))
                       for g, rv in zip(got, ref_vecs)]
                row["cosine_mean"] = round(sum(cos) / len(cos), 4)
                row["cosine_min"] = round(min(cos), 4)
                # what retrieval actually uses: similarity of query to each text
                sim_gguf = [sum(a * b for a, b in zip(norm(got[0]), norm(g)))
                            for g in got[1:]]
                sim_ref = [sum(a * b for a, b in zip(ref_vecs[0], rv))
                           for rv in ref_vecs[1:]]
                row["sim_spearman"] = spearman(sim_gguf, sim_ref)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.setdefault("embed", {})[label] = row
        print(f"EMBED  {label:12s} {json.dumps(row, ensure_ascii=False)}", flush=True)

    # ---- rerankers ----
    ref_scores = ref["rerank_scores"]
    ref_top5 = set(sorted(range(len(ref_scores)), key=lambda i: -ref_scores[i])[:5])
    for label, fname in RERANK_GGUFS.items():
        path = GGUF_DIR / fname
        row = {"file": fname, "size_mb": round(path.stat().st_size / 1024**2, 1)}
        try:
            with Server(path, "rerank") as s:
                row["vram_mb"] = round(vram_mb() - idle, 1)
                t0 = time.perf_counter()
                r = s.sess.post(f"{s.base}/v1/rerank",
                                json={"query": ref["query"], "documents": ref["pairs_docs"],
                                      "model": "x", "top_n": len(ref["pairs_docs"])},
                                timeout=300)
                r.raise_for_status()
                row["ms_50pairs"] = round((time.perf_counter() - t0) * 1000, 1)
                data = r.json().get("results") or r.json().get("data")
                got = [0.0] * len(ref["pairs_docs"])
                for d in data:
                    got[d["index"]] = float(d["relevance_score"])
                row["spearman_vs_fp16"] = spearman(got, ref_scores)
                top5 = set(sorted(range(len(got)), key=lambda i: -got[i])[:5])
                row["top5_overlap"] = len(top5 & ref_top5)
                row["scores_head"] = [round(x, 6) for x in got[:5]]
                row["ref_head"] = [round(x, 6) for x in ref_scores[:5]]
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.setdefault("rerank", {})[label] = row
        print(f"RERANK {label:12s} {json.dumps(row, ensure_ascii=False)}", flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "ref":
        cmd_ref()
    elif cmd == "gguf":
        cmd_gguf()
    else:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "ref"],
                       check=True, cwd=str(REPO))
        cmd_gguf()
