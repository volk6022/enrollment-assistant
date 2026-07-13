"""Central configuration for the stage-1 RAG pipeline.

Paths are absolute to this machine; models/params are the defaults the
experiment grid sweeps over.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TXT_DIR = REPO / "experiment-docx-processing" / "out" / "txt"
MANIFEST = REPO / "experiment-docx-processing" / "out" / "manifest.json"
EVAL_SET = REPO / "experiment-docx-processing" / "out" / "eval" / "eval_set.json"

ARTIFACTS = REPO / "rag" / "artifacts"           # built indexes/chunks (gitignored)
MODELS_DIR = REPO / "models"                      # HF cache (gitignored)

# Local Qwen GGUFs + llama.cpp server binary
LLAMA_SERVER = Path(r"C:\Users\bhunp\work-software\llama-cpp\llama-server.exe")
QWEN_2B = Path(r"G:\lmstudio\models\Jackrong\Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF\Qwen3.5-2B.Q8_0.gguf")
QWEN_4B = Path(r"G:\lmstudio\models\Jackrong\Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF\Qwen3.5-4B.Q8_0.gguf")


@dataclass
class RagConfig:
    # models
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    rerank_fp16: bool = True        # 254ms -> 93ms for 30 pairs on the 3060 Ti
    rerank_max_length: int = 256    # legal chunks; 256 tokens ≈ same speed as 192
    rerank_batch_size: int = 64

    # chunking
    max_chars: int = 1200        # split threshold for a single over-long legal point
    overlap: int = 180
    merge_chunk_chars: int = 900  # merge consecutive sibling point-chunks up to this
                                  # size. Halves the count (~446 -> ~860 avg chars) and
                                  # lifts quote_hit@3 +6pts / MRR 0.708->0.773 by cutting
                                  # fragmentation (RESULTS.md chunk-size sweep). 0 = off.

    # retrieval
    bm25_top: int = 50
    dense_top: int = 50
    rrf_k: int = 60
    fused_top: int = 50      # candidates handed to the reranker (grid winner:
                             # best src_recall@5=0.88 / MRR=0.71 at ~354ms search)
    final_top: int = 5       # chunks handed to the LLM

    # generation (llama.cpp server)
    llm_gguf: str = str(QWEN_2B)
    llm_host: str = "127.0.0.1"
    llm_port: int = 20055
    llm_ctx: int = 8192
    llm_ngl: int = 99
    max_tokens: int = 200        # SAFETY cap, not a target: with the concise
                                 # SYSTEM_PROMPT answers are ~70 tok and complete
                                 # (0/32 truncated). Gen 0.79s avg / 1.13s p90.
                                 # (The old "200 = grid winner" was a false win —
                                 #  it was fast only because it cut 13/32 answers
                                 #  mid-sentence; tps is flat ~120 regardless.)
    temperature: float = 0.2      # grid winner: best quality/latency ratio
    disable_thinking: bool = True # Qwen3.5 is a reasoning model; thinking adds
                                  # latency we don't want for a voice assistant

    # conversational (spoken) input mode: rephrase the messy query to a clean
    # canonical one, retrieve on BOTH (RRF-union), rerank with the canonical.
    # Lifts src_recall@5 on spoken input 0.625 -> ~0.84 (see RESULTS.md, the
    # "conv_multiquery_v3" arm). Costs one extra ~0.4s rephrase LLM call.
    conversational: bool = False

    def hf_env(self) -> dict[str, str]:
        """Env for HuggingFace downloads: local cache + this box's SOCKS proxy."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "HF_HOME": str(MODELS_DIR),
            "ALL_PROXY": os.getenv("ALL_PROXY", "socks5h://127.0.0.1:10808"),
            "HTTPS_PROXY": os.getenv("HTTPS_PROXY", "socks5h://127.0.0.1:10808"),
            "HTTP_PROXY": os.getenv("HTTP_PROXY", "socks5h://127.0.0.1:10808"),
        }


DEFAULT = RagConfig()
