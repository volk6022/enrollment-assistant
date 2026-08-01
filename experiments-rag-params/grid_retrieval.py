"""Phase R of the quant/top_k grid: retrieval for every (encoder backend x vector dtype).

Backends (all measured in runs/gguf_encoders.json, runs/quant_vram.json):
  torch-fp16    production today      embed 1083 MB / 28.9 ms   rerank 1083 MB / 289 ms
  gguf-q8_0     llama-server GGUF     embed  474 MB /  8.9 ms   rerank  510 MB / 521 ms
  gguf-q4_k_m   llama-server GGUF     embed  346 MB /  9.1 ms   rerank  346 MB / 1268 ms

Vector dtypes are the FAISS *storage* format, independent of which encoder produced the
vectors:
  fp32   IndexFlatIP                       (production today)
  fp16   IndexScalarQuantizer QT_fp16
  int8   IndexScalarQuantizer QT_8bit

Note up front: the FAISS index lives on the CPU and is 22.9 MB, so this axis saves at
most ~17 MB of ordinary RAM and NOTHING on VRAM. It is swept here at the retrieval level
only — if the retrieved chunk sets are identical across dtypes there is nothing for a
judge to score, and the axis stays out of the (expensive) generation+judge grid. If they
diverge, it gets promoted. That decision is made by the overlap numbers this script
prints, not by assumption.

Writes runs/grid_retrieval_<backend>_<dtype>.json with the top-20 reranked chunks per
question, which the generation phase then slices to top_k in {5,10,15,20}.

Each backend runs in its OWN process (BACKEND env var) so torch and llama-server never
hold VRAM at the same time.

    .venv\\Scripts\\python.exe experiments-rag-params\\grid_retrieval.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
POOL = HERE / "eval_set_100.json"

BACKENDS = [b for b in os.environ.get("ONLY_BACKENDS", "").split(",") if b] or \
           ["torch-fp16", "gguf-q8_0", "gguf-q4_k_m"]
VEC_DTYPES = ["fp32", "fp16", "int8"]
KEEP_TOP = 20          # max top_k the generation phase will need
EMBED_PORT, RERANK_PORT = 20081, 20082


# --------------------------------------------------------------------------- #
# child: one backend, all three vector dtypes
# --------------------------------------------------------------------------- #

def build_index(vecs, dtype: str):
    import faiss
    d = vecs.shape[1]
    if dtype == "fp32":
        idx = faiss.IndexFlatIP(d)
    elif dtype == "fp16":
        idx = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16,
                                         faiss.METRIC_INNER_PRODUCT)
    elif dtype == "int8":
        # QT_8bit is per-dimension min/max affine; vectors are L2-normalised so the
        # dynamic range is well behaved. Needs training on the corpus.
        idx = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit,
                                         faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(dtype)
    if not idx.is_trained:
        idx.train(vecs)
    idx.add(vecs)
    return idx


def child() -> None:
    os.environ.setdefault("HF_HOME", str(REPO / "models"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(REPO))
    import numpy as np

    from rag.config import DEFAULT
    from rag.index import load_chunks, tokenize_ru
    from rag.retrieve import rrf_fuse
    from rank_bm25 import BM25Okapi

    backend = os.environ["BACKEND"]
    cfg = DEFAULT
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"[{backend}] {len(chunks)} chunks, {len(pool)} questions", flush=True)

    bm25 = BM25Okapi([tokenize_ru(t) for t in texts])

    # ---- encoders ----
    servers = []
    if backend == "torch-fp16":
        from sentence_transformers import SentenceTransformer, CrossEncoder
        st = SentenceTransformer("BAAI/bge-m3", device="cuda",
                                 model_kwargs={"torch_dtype": "float16"})
        ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda",
                          max_length=cfg.rerank_max_length)
        ce.model.half()
        ce.predict([("прогрев", "прогрев модели")] * 8, batch_size=cfg.rerank_batch_size)

        def embed_many(ts):
            return st.encode(ts, batch_size=32, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False
                             ).astype("float32")

        def rerank_scores(q, docs):
            return [float(x) for x in ce.predict([(q, d) for d in docs],
                                                 batch_size=cfg.rerank_batch_size,
                                                 show_progress_bar=False)]
    else:
        sys.path.insert(0, str(HERE))
        from gguf_encoder_client import GgufServer, QUANTS
        q = QUANTS[backend]
        emb_srv = GgufServer(q["embed"], "embed", EMBED_PORT, RUNS).start()
        rr_srv = GgufServer(q["rerank"], "rerank", RERANK_PORT, RUNS).start()
        servers = [emb_srv, rr_srv]

        def embed_many(ts):
            return emb_srv.embed(ts, batch_size=64)

        def rerank_scores(q_, docs):
            return rr_srv.rerank_scores(q_, docs)

    try:
        t0 = time.time()
        vecs = embed_many(texts)
        embed_corpus_s = time.time() - t0
        print(f"[{backend}] corpus embedded in {embed_corpus_s:.0f}s, dim={vecs.shape[1]}",
              flush=True)

        for dtype in VEC_DTYPES:
            idx = build_index(vecs, dtype)
            rows = []
            t_start = time.time()
            for qi, item in enumerate(pool):
                question = item["question"]
                t0 = time.perf_counter()
                qvec = embed_many([question])[0]
                t_embed = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                sc, ids = idx.search(qvec.reshape(1, -1).astype("float32"), cfg.dense_top)
                dense = [(int(i), float(s)) for i, s in zip(ids[0], sc[0]) if i >= 0]
                t_dense = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                bs = bm25.get_scores(tokenize_ru(question))
                order = np.argsort(bs)[::-1][:cfg.bm25_top]
                sparse = [(int(i), float(bs[i])) for i in order]
                t_bm25 = (time.perf_counter() - t0) * 1000

                fused = rrf_fuse(dense, sparse, k=cfg.rrf_k)[: cfg.fused_top]
                cand = [chunks[i] for i, _ in fused]

                t0 = time.perf_counter()
                scores = rerank_scores(question, [c["text"] for c in cand])
                t_rerank = (time.perf_counter() - t0) * 1000

                ranked = sorted(zip(cand, scores), key=lambda p: -p[1])[:KEEP_TOP]
                rows.append({
                    "id": item["id"], "question": question,
                    "reference": item.get("reference", ""), "topic": item.get("topic", ""),
                    "chunks": [{"source": c["source"], "point": c.get("point"),
                                "text": c["text"], "rerank_score": s}
                               for c, s in ranked],
                    "timings_ms": {"embed": round(t_embed, 1), "dense": round(t_dense, 1),
                                   "bm25": round(t_bm25, 1), "rerank": round(t_rerank, 1)},
                })
                if (qi + 1) % 25 == 0:
                    print(f"[{backend}/{dtype}] {qi + 1}/{len(pool)} "
                          f"({time.time() - t_start:.0f}s)", flush=True)

            out = RUNS / f"grid_retrieval_{backend}_{dtype}.json"
            out.write_text(json.dumps({
                "backend": backend, "vec_dtype": dtype,
                "embed_corpus_s": round(embed_corpus_s, 1),
                "fused_top": cfg.fused_top, "keep_top": KEEP_TOP,
                "rows": rows}, ensure_ascii=False), encoding="utf-8")
            mean_search = sum(sum(r["timings_ms"].values()) for r in rows) / len(rows)
            print(f"[{backend}/{dtype}] wrote {out.name}  mean search {mean_search:.0f}ms",
                  flush=True)
    finally:
        for s in servers:
            s.stop()


# --------------------------------------------------------------------------- #
# parent: run each backend in its own process, then analyse
# --------------------------------------------------------------------------- #

def analyse() -> None:
    """Does the vector dtype change what retrieval returns? If not, that axis does not
    need the generation+judge grid."""
    print("\n=== vector-dtype effect on retrieval (top-k chunk-set overlap) ===")
    report = {}
    for backend in BACKENDS:
        loaded = {}
        for dtype in VEC_DTYPES:
            p = RUNS / f"grid_retrieval_{backend}_{dtype}.json"
            if p.exists():
                loaded[dtype] = json.loads(p.read_text(encoding="utf-8"))["rows"]
        if "fp32" not in loaded:
            continue
        base = loaded["fp32"]
        for dtype, rows in loaded.items():
            if dtype == "fp32":
                continue
            per_k = {}
            for k in (5, 10, 20):
                ov, ident = [], 0
                for rb, rd in zip(base, rows):
                    a = [c["text"] for c in rb["chunks"][:k]]
                    b = [c["text"] for c in rd["chunks"][:k]]
                    ov.append(len(set(a) & set(b)) / max(1, k))
                    ident += int(a == b)
                per_k[f"top{k}"] = {"mean_overlap": round(sum(ov) / len(ov), 4),
                                    "identical_order_pct": round(100 * ident / len(base), 1)}
            report[f"{backend}: fp32 vs {dtype}"] = per_k
            s = "  ".join(f"{k} overlap={v['mean_overlap']:.4f} "
                          f"identical={v['identical_order_pct']}%"
                          for k, v in per_k.items())
            print(f"  {backend:12s} fp32 vs {dtype:5s}  {s}")

    verdict = all(v["top10"]["mean_overlap"] >= 0.99 for v in report.values()) if report else False
    print(f"\n  -> vector dtype {'does NOT change retrieval (keep it out of the judge grid)' if verdict else 'DOES change retrieval (promote it into the judge grid)'}")
    (RUNS / "grid_retrieval_dtype_analysis.json").write_text(
        json.dumps({"report": report, "dtype_is_neutral": verdict},
                   ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        print(f"\n########## {backend} ##########", flush=True)
        env = {**os.environ, "BACKEND": backend, "CHILD": "1",
               "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        r = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                           env=env, cwd=str(REPO))
        if r.returncode != 0:
            print(f"  !! {backend} failed with exit {r.returncode}", flush=True)
    analyse()


if __name__ == "__main__":
    if os.environ.get("CHILD"):
        child()
    else:
        main()
