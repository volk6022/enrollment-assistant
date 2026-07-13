"""Production backend: optimized RAG pipeline (stage 1).

Minimal API to query the enrollment assistant:
  - Loads the built indexes (FAISS + BM25)
  - Starts a local llama.cpp server with Qwen2B
  - Serves queries: question → search + rerank → generate answer + citations

Optimized config (from grid tuning):
  - Search: bge-m3 hybrid + bge-reranker-v2-m3 (FP16), ~354ms
  - Generation: Qwen3.5-2B, max_tokens=200, temp=0.2, reasoning disabled, ~1.3s
  - End-to-end: ~1.7s (well under 2–3s target)

Usage:
  # Start the backend (llama.cpp server + indexes)
  uv run python backend.py --listen 127.0.0.1:8000

  # In another terminal, query it
  curl -X POST http://127.0.0.1:8000/answer \\
    -H "Content-Type: application/json" \\
    -d '{"question": "Какие анализы нужны перед медкомиссией?"}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import DEFAULT, RagConfig
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.generate import LlamaServer


def create_pipeline(cfg: RagConfig = DEFAULT) -> Pipeline:
    """Build indexes + start llama.cpp server, return ready pipeline."""
    print("Loading indexes...")
    idx = Indexes()

    print("Starting llama.cpp server...")
    server = LlamaServer(cfg)
    server.start()

    return Pipeline(idx, server=server, cfg=cfg)


def answer_question(pipe: Pipeline, question: str) -> dict:
    """Query the pipeline, return structured answer + metadata."""
    result = pipe.answer(question)
    return {
        "question": question,
        "answer": result["answer"],
        "citations": result["citations"],
        "latencies": {
            "search_ms": result["timings"].get("search_ms"),
            "generation_ms": result["timings"].get("gen_ms"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Enrollment Assistant backend (stage 1)")
    parser.add_argument("--listen", default="127.0.0.1:8000", help="Listen address:port")
    parser.add_argument("--mode", choices=["cli", "server"], default="cli", help="cli or server mode")
    args = parser.parse_args()

    # Build the pipeline
    pipe = create_pipeline(DEFAULT)
    print("✓ Pipeline ready\n")

    if args.mode == "cli":
        # Interactive CLI mode
        print("=" * 70)
        print("Enrollment Assistant (CLI mode)")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 70 + "\n")

        while True:
            try:
                q = input("Q: ").strip()
                if q.lower() in ("quit", "exit"):
                    break
                if not q:
                    continue

                result = answer_question(pipe, q)
                print(f"\nA: {result['answer']}\n")
                if result["citations"]:
                    print("Источники:")
                    for c in result["citations"][:3]:
                        src = c.get("source", "?")
                        pt = c.get("point", "")
                        print(f"  - {src} ({pt})" if pt else f"  - {src}")
                    print()
                print(f"Latency: {result['latencies']['search_ms']:.0f}ms search "
                      f"+ {result['latencies']['generation_ms']:.0f}ms gen = "
                      f"{result['latencies']['search_ms'] + result['latencies']['generation_ms']:.0f}ms total\n")
            except KeyboardInterrupt:
                print("\nExit")
                break
            except Exception as e:
                print(f"Error: {e}\n")

    elif args.mode == "server":
        # HTTP server mode (requires Flask/FastAPI)
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            print("ERROR: Flask not installed. Run: uv pip install flask")
            sys.exit(1)

        app = Flask(__name__)

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok"})

        @app.route("/answer", methods=["POST"])
        def answer():
            data = request.get_json() or {}
            question = data.get("question", "").strip()
            if not question:
                return jsonify({"error": "question required"}), 400

            try:
                result = answer_question(pipe, question)
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        host, port = args.listen.split(":")
        print(f"Starting server on {host}:{port}")
        app.run(host=host, port=int(port), debug=False)


if __name__ == "__main__":
    main()
