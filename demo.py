#!/usr/bin/env python
"""Quick demo: ask a question to the enrollment assistant.

Usage:
  uv run python demo.py "Какие анализы нужны перед медкомиссией?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import DEFAULT
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.generate import LlamaServer


def demo():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <question>")
        print("Example: python demo.py 'Какие анализы нужны?'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    print(f"Q: {question}\n")
    print("Loading indexes...")
    idx = Indexes()

    print("Starting server...")
    server = LlamaServer(DEFAULT)
    server.start()

    try:
        print("Querying...\n")
        pipe = Pipeline(idx, server=server, cfg=DEFAULT)
        result = pipe.answer(question)

        print(f"A: {result['answer']}\n")
        print(f"Latency: {result['timings'].get('search_ms', 0):.0f}ms search "
              f"+ {result['timings'].get('gen_ms', 0):.0f}ms generation\n")

        if result["citations"]:
            print("Источники:")
            for c in result["citations"]:
                src = c.get("source", "?")
                pt = c.get("point", "")
                score = c.get("rerank_score", 0)
                label = f"{src} ({pt})" if pt else f"{src}"
                print(f"  [{score:.2f}] {label}")
    finally:
        server.stop()


if __name__ == "__main__":
    demo()
