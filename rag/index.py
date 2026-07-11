"""Build and load the two retrieval indexes: FAISS dense + BM25.

Dense: FAISS IndexFlatIP over L2-normalized bge-m3 vectors (exact cosine; the
corpus is a few thousand chunks, so exact search is <5 ms — no HNSW needed).
Sparse: rank_bm25 BM25Okapi over Russian-tokenized chunk texts.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

from rag.config import ARTIFACTS, DEFAULT, RagConfig
from rag.embed import embed

CHUNKS = ARTIFACTS / "chunks.jsonl"
FAISS_INDEX = ARTIFACTS / "dense.faiss"
VECS = ARTIFACTS / "dense.npy"
BM25_PKL = ARTIFACTS / "bm25.pkl"


def tokenize_ru(text: str) -> list[str]:
    text = re.sub(r"[^a-zа-яё0-9]+", " ", (text or "").lower())
    return [t for t in text.split() if len(t) >= 2]


def load_chunks() -> list[dict]:
    return [json.loads(ln) for ln in CHUNKS.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build(cfg: RagConfig = DEFAULT) -> dict:
    import faiss
    from rank_bm25 import BM25Okapi

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    vecs = embed(texts, cfg)
    np.save(VECS, vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(FAISS_INDEX))

    bm25 = BM25Okapi([tokenize_ru(t) for t in texts])
    BM25_PKL.write_bytes(pickle.dumps(bm25))

    print(f"indexed {len(chunks)} chunks | dim={vecs.shape[1]} | faiss+bm25 saved")
    return {"chunks": len(chunks), "dim": int(vecs.shape[1])}


class Indexes:
    """Loaded retrieval state kept resident in memory."""

    def __init__(self, cfg: RagConfig = DEFAULT):
        import faiss

        self.cfg = cfg
        self.chunks = load_chunks()
        self.index = faiss.read_index(str(FAISS_INDEX))
        self.bm25 = pickle.loads(BM25_PKL.read_bytes())

    def dense_search(self, qvec: np.ndarray, top: int) -> list[tuple[int, float]]:
        scores, idxs = self.index.search(qvec.reshape(1, -1), top)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def bm25_search(self, query: str, top: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize_ru(query))
        order = np.argsort(scores)[::-1][:top]
        return [(int(i), float(scores[i])) for i in order]


if __name__ == "__main__":
    build()
