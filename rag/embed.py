"""Dense embeddings via bge-m3 (multilingual, strong on Russian).

bge-m3 needs no query/passage instruction prefixes (unlike e5). We L2-normalize
so a FAISS inner-product index computes cosine similarity.
"""
from __future__ import annotations

import os

import numpy as np

from rag.config import DEFAULT, RagConfig

_MODEL = None


def get_embedder(cfg: RagConfig = DEFAULT):
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("HF_HOME", cfg.hf_env()["HF_HOME"])
        from sentence_transformers import SentenceTransformer

        # FP16 halves bge-m3's VRAM (~2.27 -> ~1.13 GB) on CUDA with negligible quality
        # loss; embeddings are cast back to float32 for FAISS in embed().
        model_kwargs = {}
        if cfg.embed_fp16 and cfg.device != "cpu":
            model_kwargs["torch_dtype"] = "float16"
        _MODEL = SentenceTransformer(
            cfg.embed_model, device=cfg.device,
            model_kwargs=model_kwargs or None,
        )
    return _MODEL


def embed(texts: list[str], cfg: RagConfig = DEFAULT, batch_size: int = 32) -> np.ndarray:
    model = get_embedder(cfg)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 256,
    )
    return vecs.astype("float32")


def embed_query(text: str, cfg: RagConfig = DEFAULT) -> np.ndarray:
    return embed([text], cfg)[0]
