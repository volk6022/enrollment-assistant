"""Drop-in embedder/reranker backed by llama-server GGUF instead of torch.

Measured on this box (runs/gguf_encoders.json, runs/quant_vram.json):

                        resident VRAM   1 query / 50 pairs   fidelity vs torch fp16
  torch fp16 (current)     1083 MB        28.9 ms / 289 ms    (reference)
  GGUF Q8_0                 474 MB         8.9 ms / 525 ms    cos 0.9993 / spearman 0.9905
  GGUF Q4_K_M               346 MB         9.1 ms / 1268 ms   cos 0.9692 / spearman 0.9674

Q8_0 halves VRAM at essentially no quality cost and makes the whole streaming stack fit
in 8 GB (6531 MB vs 7810 MB for the torch path). The reranker is ~1.8x slower than torch
even tuned, which is the trade being bought.

The point of this shape is that both encoders become plain HTTP services — the same
thing production will point at via .env URLs, no torch model resident in the Python
process at all.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parent.parent
GGUF_DIR = REPO / "models" / "gguf"
LLAMA_SERVER = r"C:\Users\bhunp\work-software\llama-cpp\llama-server.exe"

# Cache flags: llama-server keeps per-slot context checkpoints (-ctxcp) and a global
# prompt cache (-cram) that it saves to and re-scans on every request. Neither can ever
# hit here — each embedded chunk and each reranked document is a distinct text.
#
# For RERANKING they are provably free: rerank_server_tuning.py stage 1 measured all six
# on/off combinations at 520.2-521.3 ms, i.e. identical. Reranking never touches that path.
# For EMBEDDING they are NOT free — they are fatal. With the cache on, the server saved a
# prompt per chunk and re-listed the whole cache on every update; embedding the 5856-chunk
# corpus CRASHED it at ~784 texts (llama_bge-m3-q8_0_20081.log ends mid-write).
# So: disabled on both, load-bearing on the embedder, insurance on the reranker (which in
# a real run does 100 questions x 50 docs = 5000 tasks in one server lifetime).
_NO_CACHE = ["--no-cache-prompt", "-cram", "0", "-ctxcp", "0", "-cpent", "-1"]

# Tuned in rerank_server_tuning.py: np=4 is the ceiling (667/552/522 ms at np 1/2/4, flat
# to np=32), b4096/ub1024 optimal, -c affects neither speed nor VRAM, -fa off is much worse.
RERANK_FLAGS = ["-np", "4", "-b", "4096", "-ub", "1024", "-c", "16384", "-fa", "on"] + _NO_CACHE
# -b must cover the tokens of one client request: EMBED_BATCH texts x ~350 tok.
EMBED_FLAGS = ["-np", "4", "-c", "8192", "-b", "8192", "-ub", "2048"] + _NO_CACHE

EMBED_BATCH = 16   # texts per /v1/embeddings request (16 x ~350 tok fits -b 8192)

QUANTS = {
    "gguf-q8_0": {"embed": "bge-m3-q8_0.gguf", "rerank": "bge-reranker-v2-m3-q8_0.gguf"},
    "gguf-q4_k_m": {"embed": "bge-m3-q4_k_m.gguf", "rerank": "bge-reranker-v2-m3-q4_k_m.gguf"},
}


class GgufServer:
    """One llama-server process hosting one encoder (llama.cpp is single-model)."""

    def __init__(self, model_file: str, mode: str, port: int, log_dir: Path) -> None:
        self.model = GGUF_DIR / model_file
        self.mode = mode          # "embed" | "rerank"
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None
        self.sess = requests.Session()
        self.sess.trust_env = False   # never send localhost through the SOCKS proxy

    def start(self, timeout: float = 240.0) -> "GgufServer":
        if not self.model.exists():
            raise FileNotFoundError(self.model)
        args = [LLAMA_SERVER, "-m", str(self.model), "--host", "127.0.0.1",
                "--port", str(self.port), "-ngl", "99", "--no-webui"]
        args += ["--embedding", *EMBED_FLAGS] if self.mode == "embed" else \
                ["--reranking", *RERANK_FLAGS]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_dir / f"llama_{self.model.stem}_{self.port}.log",
                         "w", encoding="utf-8")
        self.proc = subprocess.Popen(args, stdout=self._log, stderr=self._log)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited; see {self._log.name}")
            try:
                r = self.sess.get(f"{self.base}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return self
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
        raise TimeoutError(f"{self.model.name} not healthy in {timeout}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self.proc = None
        if getattr(self, "_log", None):
            self._log.close()
        time.sleep(1.5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # --- embed ---
    def embed(self, texts: list[str], batch_size: int = EMBED_BATCH) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            r = self.sess.post(f"{self.base}/v1/embeddings",
                               json={"input": chunk, "model": "x"}, timeout=600)
            if r.status_code >= 400:
                # surface the server's own message — a bare 500 hides whether this was
                # a batch-too-big rejection or the process dying under us
                alive = self.proc is not None and self.proc.poll() is None
                raise RuntimeError(
                    f"embed failed at text {i}/{len(texts)} (HTTP {r.status_code}, "
                    f"server_alive={alive}): {r.text[:300]}")
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        v = np.asarray(out, dtype=np.float32)
        # L2-normalise so a FAISS inner-product index computes cosine, matching
        # rag/embed.py's normalize_embeddings=True.
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (v / n).astype(np.float32)

    # --- rerank ---
    def rerank_scores(self, query: str, docs: list[str]) -> list[float]:
        r = self.sess.post(f"{self.base}/v1/rerank",
                           json={"query": query, "documents": docs, "model": "x",
                                 "top_n": len(docs)}, timeout=900)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("results") or payload.get("data") or []
        scores = [0.0] * len(docs)
        for d in rows:
            scores[d["index"]] = float(d["relevance_score"])
        return scores
