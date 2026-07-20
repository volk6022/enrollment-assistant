"""Metric helpers: WER, VRAM snapshots, RAM usage."""

from __future__ import annotations

import re

import psutil


def _normalize_russian(text: str) -> str:
    """Lowercase, fold ё→е, strip a BOM/punctuation for WER comparison.

    ё→е folding removes a large class of *false* errors: our references are
    yo-ified ("трёх") while ASR output is inconsistent ("трех"/"трёх").
    The leading UTF-8 BOM (\\ufeff) on reference files is also dropped.
    """
    text = text.replace("﻿", "").lower().replace("ё", "е")
    # Remove punctuation (keep Cyrillic, Latin, digits, spaces)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis strings.

    Both strings are normalized (lowercased, ё-folded, punctuation stripped)
    before comparison. Returns a float in [0, inf); values > 1.0 are possible
    when the hypothesis is longer than the reference.

    CAVEAT: our references spell numbers as words ("двести восемьдесят четыре",
    "тридцать первого") while ASR emits digits ("284", "31"). WER counts these
    as substitutions/deletions even when the transcription is semantically
    perfect, inflating the score on number-heavy samples. Use `cer()` for a
    reading less sensitive to this and to word-boundary noise.
    """
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError("Install jiwer: uv add jiwer") from exc

    ref_norm = _normalize_russian(reference)
    hyp_norm = _normalize_russian(hypothesis)

    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    return jiwer.wer(ref_norm, hyp_norm)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate — same normalization as wer(), char-level distance.

    Less sensitive than WER to word-boundary splits and to the word-vs-digit
    number mismatch (a wrong digit costs a few chars, not two whole words).
    """
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError("Install jiwer: uv add jiwer") from exc

    ref_norm = _normalize_russian(reference)
    hyp_norm = _normalize_russian(hypothesis)

    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    return jiwer.cer(ref_norm, hyp_norm)


# ---------------------------------------------------------------------------
# VRAM helpers (safe when CUDA is not available)
# ---------------------------------------------------------------------------

def vram_snapshot() -> float:
    """Return currently allocated VRAM in MB, or 0.0 if CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except ImportError:
        pass
    return 0.0


def vram_peak_reset() -> None:
    """Reset peak VRAM statistics so the next call to vram_peak_mb() is accurate."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def vram_peak_mb() -> float:
    """Return the peak VRAM allocated since the last vram_peak_reset(), in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except ImportError:
        pass
    return 0.0


def ram_mb() -> float:
    """Return the current process resident set size (RAM) in MB."""
    return psutil.Process().memory_info().rss / 1e6
