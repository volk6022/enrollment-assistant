"""Unit coverage for A-13's core fix: `backend.rag.index.Indexes` on a
directory that doesn't have a built RAG index yet (chunks.jsonl/dense.faiss/
bm25.pkl missing) must raise a specific, actionable
`RagArtifactsMissingError` -- never a bare `FileNotFoundError` (which used to
propagate straight out of `backend/app.py`'s `lifespan` and crash startup,
see that module's comment and README.md "Известные ограничения").

This is deliberately a fast, dependency-free unit test (tempdir only, no
GPU/llama-server/faiss data file needed) -- the actual end-to-end claim ("the
FastAPI process stays up and /health reports it") was verified by a real
`python -m backend.app` run against a temporarily-renamed
`backend/rag/artifacts/` during T-12's own delivery (see the delivery
report); reproducing that here would need whisper/silero/scenario startup
(tens of seconds, real GPU) for a claim this test already pins at the layer
that actually raises.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.rag.index import RagArtifactsMissingError, _REQUIRED_ARTIFACTS, Indexes, build


def test_indexes_raises_actionable_error_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(RagArtifactsMissingError) as exc_info:
        Indexes(tmp_path)

    err = exc_info.value
    assert err.artifacts_dir == tmp_path
    assert set(err.missing) == set(_REQUIRED_ARTIFACTS)
    # A-13: "внятная ошибка... что с этим делать" -- the message must name
    # the missing files and point at how to build/copy them, not just say
    # "something's wrong".
    message = str(err)
    for name in _REQUIRED_ARTIFACTS:
        assert name in message
    assert "backend.rag.index" in message  # points at the build() escape hatch
    assert "README.md" in message


def test_indexes_raises_when_partially_built(tmp_path: Path) -> None:
    """Only `chunks.jsonl` copied (README.md's documented first step of
    rebuilding from the legacy `rag/` package) -- still missing dense.faiss/
    bm25.pkl, still must raise the same descriptive error, not a bare
    `faiss`/`pickle` traceback from whichever file happens to be checked
    first.
    """
    (tmp_path / "chunks.jsonl").write_text(
        json.dumps({"text": "x", "point": "1", "source": "s"}) + "\n", encoding="utf-8"
    )

    with pytest.raises(RagArtifactsMissingError) as exc_info:
        Indexes(tmp_path)

    assert set(exc_info.value.missing) == {"dense.faiss", "bm25.pkl"}


def test_build_then_load_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of the same contract: once the three files genuinely
    exist, `Indexes` must load cleanly (no false positive on the presence
    check). Uses a fake encoder so this stays a fast unit test -- no llama-
    server involved, matching this project's own unit/contract split
    (plan.md §12: unit tests don't touch live infra, contract tests do).
    """
    import numpy as np

    from backend.rag.config import DEFAULT

    chunks_path = tmp_path / "src_chunks.jsonl"
    chunks_path.write_text(
        "\n".join(
            json.dumps({"text": f"chunk {i}", "point": str(i), "source": "s.docx"}) for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeEncoder:
        def embed(self, texts: list[str], batch_size: int) -> np.ndarray:
            return np.eye(len(texts), 8, dtype="float32")

    artifacts_dir = tmp_path / "artifacts"
    build(DEFAULT, artifacts_dir, encoder=_FakeEncoder(), chunks_path=chunks_path)

    idx = Indexes(artifacts_dir)
    assert len(idx.chunks) == 3
