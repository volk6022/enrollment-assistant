"""Dense (FAISS) + sparse (BM25) retrieval indexes.

Ported from `rag/index.py` (legacy, untouched by this task). The retrieval
algorithm is identical; the only change is *how* dense vectors are produced:
via `GgufEncoderClient` over HTTP (bge-m3 GGUF Q8_0 on llama-server), never a
resident torch model -- plan.md §6, findings.md §5.1.

Chunk *building* (docx -> text -> chunk+metadata) is out of scope for T-04
(encoder/backend migration only) and stays entirely on the legacy `rag/`
package. `_LEGACY_CHUNKS` below is a READ of that package's build output
(`rag/artifacts/chunks.jsonl`, 5856 chunks) -- not an import of its code and
not a modification of it, so it doesn't violate "leave rag/ untouched".
Whoever ports chunk-building to `backend/` should point `chunks_path` at that
new location instead; nothing else in this module needs to change.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

from backend.rag.config import RagSettingsProtocol
from backend.rag.gguf_encoder_client import GgufEncoderClient

_LEGACY_CHUNKS = Path(__file__).resolve().parent.parent.parent / "rag" / "artifacts" / "chunks.jsonl"

# The three files `Indexes` needs resident (`build()` above writes all three,
# plus `dense.npy` which nothing at load time actually reads back).
_REQUIRED_ARTIFACTS = ("chunks.jsonl", "dense.faiss", "bm25.pkl")


class RagArtifactsMissingError(RuntimeError):
    """A-13: `artifacts_dir` doesn't have a built RAG index yet.

    This is the expected state of `backend/rag/artifacts/` right after a
    clean clone -- the directory is `.gitignore`d (chunk/vector data is not
    committed, README.md "RAG-индекс"). Before this exception existed,
    `Indexes.__init__` let a bare `FileNotFoundError` from `Path.read_text()`
    (or a faiss/pickle error, depending on which file was missing first)
    propagate straight out of FastAPI's `lifespan`, which is *not* an
    internal message a human can act on, and under `restart: unless-stopped`
    (docker-compose.yml) that repeats forever with no way to see it without
    already knowing to run `docker compose logs`.

    `backend/app.py`'s lifespan catches this specific exception (not RAG
    errors in general) and keeps the process running with RAG disabled
    instead of letting the exception crash startup -- see the comment there.
    """

    def __init__(self, artifacts_dir: Path, missing: list[str]) -> None:
        self.artifacts_dir = artifacts_dir
        self.missing = missing
        super().__init__(
            f"RAG index not found in {artifacts_dir} (missing: {', '.join(missing)}). "
            "This is expected on a clean clone -- backend/rag/artifacts/ is "
            ".gitignore'd, no index is committed. To fix, either:\n"
            "  (a) copy an already-built backend/rag/artifacts/ from another "
            "machine (chunks.jsonl + dense.faiss + dense.npy + bm25.pkl), or\n"
            "  (b) build it here: start the embedding llama-server "
            "(scripts/serve-models.ps1 / .sh, EMBEDDING_ENDPOINT), then run "
            "`Copy-Item rag\\artifacts\\chunks.jsonl backend\\rag\\artifacts\\chunks.jsonl` "
            "followed by `python -c \"from pathlib import Path; from backend.rag.config "
            "import DEFAULT; from backend.rag.index import build; "
            f"build(DEFAULT, Path(r'{artifacts_dir}'))\"`.\n"
            "Full procedure: README.md 'RAG-индекс' / 'Legacy: пересборка RAG-индекса'."
        )


def tokenize_ru(text: str) -> list[str]:
    text = re.sub(r"[^a-zа-яё0-9]+", " ", (text or "").lower())
    return [t for t in text.split() if len(t) >= 2]


def load_chunks(chunks_path: Path = _LEGACY_CHUNKS) -> list[dict]:
    return [json.loads(ln) for ln in chunks_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build(
    settings: RagSettingsProtocol,
    artifacts_dir: Path,
    encoder: GgufEncoderClient | None = None,
    chunks_path: Path = _LEGACY_CHUNKS,
) -> dict:
    """(Re)build dense.faiss / dense.npy / bm25.pkl / chunks.jsonl, embedding
    every chunk through the GGUF encoder endpoint.

    This is the call T-04's acceptance test exercises directly: run all 5856
    chunks through `POST /v1/embeddings` without an HTTP 500 -- the crash
    findings.md §5.1/§5.2 documents at ~chunk 784 when the embedding
    llama-server's prompt cache is left enabled, fixed by the
    `--no-cache-prompt -cram 0 -ctxcp 0 -cpent -1` launch flags.
    """
    import faiss
    from rank_bm25 import BM25Okapi

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    enc = encoder or GgufEncoderClient(settings)
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    vecs = enc.embed(texts, batch_size=settings.rag_batch_size)
    np.save(artifacts_dir / "dense.npy", vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(artifacts_dir / "dense.faiss"))

    bm25 = BM25Okapi([tokenize_ru(t) for t in texts])
    (artifacts_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    (artifacts_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )
    return {"chunks": len(chunks), "dim": int(vecs.shape[1])}


class Indexes:
    """Loaded retrieval state kept resident in the "rag" pool worker."""

    def __init__(self, artifacts_dir: Path):
        import faiss

        artifacts_dir = Path(artifacts_dir)
        missing = [name for name in _REQUIRED_ARTIFACTS if not (artifacts_dir / name).is_file()]
        if missing:
            raise RagArtifactsMissingError(artifacts_dir, missing)

        self.chunks = load_chunks(artifacts_dir / "chunks.jsonl")
        self.index = faiss.read_index(str(artifacts_dir / "dense.faiss"))
        self.bm25 = pickle.loads((artifacts_dir / "bm25.pkl").read_bytes())

    def dense_search(self, qvec: np.ndarray, top: int) -> list[tuple[int, float]]:
        scores, idxs = self.index.search(qvec.reshape(1, -1), top)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def bm25_search(self, query: str, top: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize_ru(query))
        order = np.argsort(scores)[::-1][:top]
        return [(int(i), float(scores[i])) for i in order]
