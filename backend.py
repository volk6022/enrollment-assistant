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
import uuid
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import DEFAULT, RagConfig
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.generate import LlamaServer

# Greeting for the voice-gateway `/voice/start` handshake (spoken by TTS if wired).
GREETING = (
    "Здравствуйте. Вас приветствует голосовой ассистент приёмной комиссии. "
    "Задайте, пожалуйста, ваш вопрос о поступлении."
)


def create_pipeline(cfg: RagConfig = DEFAULT) -> Pipeline:
    """Build indexes + start llama.cpp server, return ready pipeline."""
    print("Loading indexes...")
    idx = Indexes()

    print("Starting llama.cpp server...")
    server = LlamaServer(cfg)
    server.start()

    pipe = Pipeline(idx, server=server, cfg=cfg)

    # Warm up CUDA kernels (embedder + reranker + llama): the first real query
    # otherwise costs ~26s while kernels compile. Discard the result.
    print("Warming up (first-query CUDA compile)...")
    try:
        pipe.answer("тест")
    except Exception as e:
        print(f"  warmup skipped: {e}")
    return pipe


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
    parser.add_argument("--conversational", action="store_true",
                        help="enable spoken-input mode (rephrase + multi-query). "
                             "Recommended when serving the voice-gateway web GUI.")
    args = parser.parse_args()

    # Build the pipeline
    cfg = replace(DEFAULT, conversational=True) if args.conversational else DEFAULT
    pipe = create_pipeline(cfg)
    print(f"✓ Pipeline ready (conversational={cfg.conversational})\n")

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

        # --- voice-gateway adapter: the web GUI (services/voice-gateway) talks to
        # these endpoints. Maps its /voice/* contract onto this stage-1 pipeline so
        # the client's existing GUI runs against the new RAG. TTS/STT stay in the
        # gateway (Yandex) and fail gracefully; here we only serve text answers.
        @app.route("/voice/start", methods=["POST"])
        def voice_start():
            data = request.get_json(silent=True) or {}
            return jsonify({
                "call_id": data.get("call_id") or uuid.uuid4().hex,
                "session_id": data.get("session_id") or uuid.uuid4().hex,
                "answer": GREETING, "voice_answer": GREETING, "tts_text": GREETING,
                "citations": [], "need_clarification": False,
                "meta": {"engine": "stage1-rag", "conversational": cfg.conversational},
            })

        @app.route("/voice/turn", methods=["POST"])
        def voice_turn():
            data = request.get_json(silent=True) or {}
            transcript = (data.get("transcript") or "").strip()
            if not transcript:
                return jsonify({"error": "transcript required"}), 400
            try:
                r = pipe.answer(transcript)
                gen = r.get("gen") or {}
                # Full intermediate trace surfaced to the GUI "Технические данные" block.
                meta = {
                    "engine": "stage1-rag",
                    "conversational": cfg.conversational,
                    "transcript": transcript,
                    "canonical_query": r.get("canonical_query"),
                    "config": {"model": "Qwen3.5-2B.Q8_0", "max_tokens": cfg.max_tokens,
                               "temperature": cfg.temperature, "fused_top": cfg.fused_top,
                               "final_top": cfg.final_top, "n_chunks": len(pipe.idx.chunks)},
                    "timings_ms": {k: round(v, 1) for k, v in r["timings"].items()},
                    "generation": {
                        "answer_tokens": gen.get("completion_tokens"),
                        "prompt_tokens": gen.get("prompt_tokens"),
                        "tps": round(gen["tps"], 1) if gen.get("tps") else None,
                        "reasoning": gen.get("reasoning") or "",
                    },
                    "retrieval": [
                        {"rank": i + 1, "source": c.get("source"), "point": c.get("point"),
                         "section_path": c.get("section_path"),
                         "rerank_score": round(c["rerank_score"], 4) if c.get("rerank_score") is not None else None,
                         "rrf_score": round(c["rrf_score"], 5) if c.get("rrf_score") is not None else None,
                         "text": c.get("text")}
                        for i, c in enumerate(r.get("top_chunks") or [])
                    ],
                }
                return jsonify({
                    "answer": r["answer"], "voice_answer": r["answer"], "tts_text": r["answer"],
                    "citations": r["citations"], "need_clarification": False,
                    "meta": meta,
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route("/voice/handoff", methods=["POST"])
        def voice_handoff():
            msg = "Соединяю вас с оператором приёмной комиссии."
            return jsonify({"answer": msg, "voice_answer": msg, "tts_text": msg,
                            "citations": [], "meta": {"engine": "stage1-rag", "handoff": True}})

        host, port = args.listen.split(":")
        print(f"Starting server on {host}:{port}")
        app.run(host=host, port=int(port), debug=False)


if __name__ == "__main__":
    main()
