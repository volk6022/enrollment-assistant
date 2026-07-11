"""Smoke test for the local LLM: verify reasoning is actually suppressed.

Qwen3.5 is a reasoning-distilled model. For a voice assistant we don't want it
spending latency on <think> tokens. This runs the SAME prompt with thinking
disabled vs enabled and prints raw output + latency, so we can confirm the
suppression works (no <think> block, fewer tokens, lower latency).

Runs one model at a time (2B by default) — no retrieval stack, no torch GPU use
beyond llama-server itself.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import DEFAULT, QWEN_2B
from rag.generate import LlamaServer, build_messages

RUNS = Path(__file__).resolve().parent / "runs"

CONTEXT = [
    {"source": "ПРАВИЛА ПРИЕМА 2025.docx", "point": "4.2",
     "text": "Приём документов, необходимых для поступления, проводится с 20 июня по 20 июля 2025 года."},
    {"source": "ПРАВИЛА ПРИЕМА 2025.docx", "point": "4.3",
     "text": "Документы подаются лично, направляются через операторов почтовой связи либо в электронной форме."},
]
QUESTION = "До какого числа можно подать документы и какими способами?"


def run(disable_thinking: bool) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(DEFAULT, llm_gguf=str(QWEN_2B), disable_thinking=disable_thinking,
                              llm_port=20056 if disable_thinking else 20057)
    log = RUNS / f"llama_{'nothink' if disable_thinking else 'think'}.log"
    tag = "THINKING OFF" if disable_thinking else "THINKING ON"
    print(f"\n{'='*66}\n{tag}\n{'='*66}")
    server = LlamaServer(cfg)
    try:
        server.start(log_path=str(log))
        messages = build_messages(QUESTION, CONTEXT)
        g = server.complete(messages, cfg)
        reasoning_len = len(g.get("reasoning") or "")
        print(f"gen: {g['gen_sec']:.2f}s | completion_tokens: {g['completion_tokens']} | tps: {g['tps']:.0f}")
        print(f"reasoning_content chars: {reasoning_len}  ({'THOUGHT' if reasoning_len else 'no reasoning emitted'})")
        print(f"answer chars: {len(g['answer'])}")
        print(f"--- answer ---\n{g['answer'][:350]}")
    finally:
        server.stop()


if __name__ == "__main__":
    run(disable_thinking=True)
    run(disable_thinking=False)
