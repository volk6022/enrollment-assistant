"""Phase 2: feed every judge comment (from semantic_judge_run.py) back into ONE
big prompt for a single local model, np=1 / ctx=120000, reasoning ON, and ask it
to name the most severe and most frequent problems across the whole grid.

Usage: uv run python experiments-rag-params/meta_analysis_run.py <judge_results.json>
  (defaults to runs/judge_results_latest.json)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from llama_local import LocalServer

RUNS = Path(__file__).resolve().parent / "runs"

META_SYSTEM = (
    "Ты — старший QA-аналитик RAG-систем. Тебе дают результаты семантического "
    "судейства большого грид-теста (несколько баз знаний x несколько моделей x 100 "
    "вопросов), каждая строка — числовые оценки (rephrase/chunks/answer, 0..1) плюс "
    "однострочный комментарий проверяющего о самом слабом шаге. Твоя задача — "
    "прочитать ВСЕ комментарии и построить итоговый разбор:\n"
    "1. Топ-5 самых ТЯЖЁЛЫХ проблем (по влиянию на качество ответа абитуриенту), "
    "с примерами из комментариев и оценкой на каком шаге (rephrase/chunks/answer) "
    "они чаще всего возникают.\n"
    "2. Топ-5 самых ЧАСТЫХ проблем (по числу повторений в комментариях), даже если "
    "по отдельности не критичны.\n"
    "3. Есть ли системная разница между базами знаний (baseline/kb-relevant/kb-full) "
    "или моделями — на каком шаге конвейера они расходятся.\n"
    "4. Итоговая рекомендация: что чинить в первую очередь.\n"
    "Пиши по-русски, структурированным markdown, конкретно — не общими словами."
)


def build_prompt(data: dict) -> str:
    agg = data["aggregate"]
    lines = [f"ЧИСЛОВАЯ СВОДКА (n={agg['n_total']}, parse_fail={agg['n_parse_fail']}):"]
    lines.append("by_model: " + json.dumps(agg["by_model"], ensure_ascii=False))
    lines.append("by_base: " + json.dumps(agg["by_base"], ensure_ascii=False))
    lines.append("by_base_model: " + json.dumps(agg["by_base_model"], ensure_ascii=False))
    lines.append("\nКОММЕНТАРИИ ПРОВЕРЯЮЩЕГО (base | model | scores | comment):")
    for r in data["results"]:
        scores = f"r={r['rephrase_score']} c={r['chunks_score']} a={r['answer_score']}"
        lines.append(f"- [{r['base']} | {r['model']} | {scores}] {r.get('comment', '')}")
    return "\n".join(lines)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (RUNS / "judge_results_latest.json")
    data = json.loads(src.read_text(encoding="utf-8"))
    prompt = build_prompt(data)
    print(f"meta-prompt: {len(prompt)} chars (~{len(prompt)//3} tokens)")

    srv = LocalServer(np=1, ctx=120000, port=20098)
    ts = time.strftime("%Y%m%d_%H%M%S")
    srv.start(log_path=str(RUNS / f"llama_meta_np1_{ts}.log"))
    try:
        t0 = time.time()
        r = srv.chat(
            [{"role": "system", "content": META_SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=4000, temperature=0.3, timeout=1200,
        )
        print(f"meta-analysis generated in {time.time() - t0:.0f}s, "
              f"{r['tokens']} tokens, finish={r['finish']}")
    finally:
        srv.stop()

    out = RUNS / "META_ANALYSIS.md"
    out.write_text(f"# Meta-analysis of semantic judge results\n\nsource: {src.name}\n\n"
                    + r["content"], encoding="utf-8")
    print(f"-> {out}")
    print("\n" + r["content"])


if __name__ == "__main__":
    main()
