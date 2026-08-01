"""Phases G and J of the quant/top_k grid: generate answers, then judge them.

Grid: encoder backend {torch-fp16, gguf-q8_0, gguf-q4_k_m} x top_k {5,10,15,20}
      = 12 cells x 100 questions = 1200 answers, then 1200 judge calls.

Why these axes and not the originally proposed ones:
  - vector dtype (fp32/fp16/int8) is NOT here. Measured at retrieval level
    (grid_retrieval.py): quantised vectors are SLOWER to search (dense 10-14 ms vs
    1.6 ms for IndexFlatIP), save ~17 MB of CPU RAM (the FAISS index is 22.9 MB and
    lives on the CPU, so nothing on VRAM), and still perturb the ranking. Nothing to
    trade, so nothing to judge. fp32 is used throughout.
  - rerank_batch_size is NOT here. Scores are byte-identical for bs 16/32/64 in fp16
    (runs/quant_vram.json); it is a VRAM/latency knob, measured there.
  - bitsandbytes int8 weights are NOT here: more VRAM AND 3-6x slower than fp16.

The encoder backend axis IS worth judging: on the full 5856-chunk corpus the backends
disagree much more than the isolated fidelity probe suggested — top-10 overlap vs
torch-fp16 is 0.92 (q8_0) and 0.83 (q4_k_m). Whether that costs answer quality is
exactly the open question.

Retrieval is NOT redone here — grid_retrieval.py already wrote top-20 reranked chunks
per question per backend, and each cell just slices them to top_k. So this phase never
loads an encoder, which is what keeps the judge's VRAM uncontended.

Judge protocol follows semantic_judge_run.py (same model, np, reasoning-on, temperature)
with one deliberate change: there is no rephrase step in this pipeline (conversational
mode is off, retrieval runs on the raw question), so rephrase_score is dropped and only
chunks_score / answer_score are asked for.

    .venv\\Scripts\\python.exe experiments-rag-params\\grid_generate_judge.py generate
    .venv\\Scripts\\python.exe experiments-rag-params\\grid_generate_judge.py judge
    .venv\\Scripts\\python.exe experiments-rag-params\\grid_generate_judge.py all
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from llama_local import LocalServer, render_chatml  # noqa: E402
from rag.config import DEFAULT, QWOPUS_4B  # noqa: E402
from rag.generate import build_messages  # noqa: E402

BACKENDS = ["torch-fp16", "gguf-q8_0", "gguf-q4_k_m"]
TOP_KS = [5, 10, 15, 20]
VEC_DTYPE = "fp32"

GEN_PORT, JUDGE_PORT = 20098, 20099
# llama-server splits -c across slots, so what matters is ctx/np, NOT ctx.
# Measured prompt sizes over the real retrieval output (chars / ~2.6 for Russian):
#     top_k=5   mean 1702 tok / max 2298
#     top_k=10  mean 3395 tok / max 4421
#     top_k=15  mean 5069 tok / max 6457
#     top_k=20  mean 6769 tok / max 8722
# A first attempt at ctx=16384, np=4 gave slot_ctx=4096 — fine for k=5 (which is why
# it showed truncated=0) but it would have silently truncated k=10/15/20, i.e. the
# whole point of the sweep. Sized for max(top_k=20) + answer + headroom.
GEN_NP = 4
GEN_SLOT_CTX = 12288
GEN_CTX = GEN_SLOT_CTX * GEN_NP
ANSWERS_PATH = RUNS / "grid_answers.json"
JUDGE_PATH = RUNS / "grid_judge_results.json"
JUDGE_PARTIAL = RUNS / "grid_judge_partial.json"

JUDGE_SYSTEM = (
    "Ты — строгий аудитор качества RAG-конвейера приёмной комиссии юридического вуза "
    "МВД (голосовой ассистент для абитуриентов). Тебе даны: вопрос абитуриента, "
    "фрагменты документов, найденные RAG-поиском, финальный ответ модели и эталонный "
    "ответ (reference) для сверки фактов — эталон короче и суше, это не образец стиля, "
    "а источник истины.\n\n"
    "Оцени 2 шага конвейера НЕЗАВИСИМО:\n"
    "1. chunks_score (0..1): есть ли среди фрагментов фактический материал, достаточный "
    "чтобы ПРАВИЛЬНО ответить на вопрос (recall); низкий балл если нужного факта в "
    "чанках вообще нет, даже если модель как-то выкрутилась.\n"
    "2. answer_score (0..1): отвечает ли финальный ответ точно на вопрос, без выдуманных "
    "фактов, согласуясь с эталоном по сути (не по формулировке).\n\n"
    "Думай пошагово сколько нужно. В САМОМ КОНЦЕ ответа выведи РОВНО ОДИН JSON-объект и "
    "больше НИЧЕГО после него, в формате:\n"
    '{"chunks_score": 0.0, "answer_score": 0.0, "comment": "..."}\n'
    "comment — одно предложение по-русски: какой шаг слабее и почему (конкретно), или "
    "что отработало хорошо если всё ок."
)
JSON_RE = re.compile(r"\{[^{}]*\"chunks_score\"[^{}]*\}", re.DOTALL)


def cell_key(backend: str, top_k: int, qid: str) -> str:
    return f"{backend}|k{top_k}|{qid}"


def load_retrieval(backend: str) -> list[dict]:
    p = RUNS / f"grid_retrieval_{backend}_{VEC_DTYPE}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p.name} — run grid_retrieval.py first")
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


# --------------------------------------------------------------------------- #
# phase G: generation
# --------------------------------------------------------------------------- #

def cmd_generate() -> None:
    retrieval = {b: load_retrieval(b) for b in BACKENDS}
    n_q = len(retrieval[BACKENDS[0]])
    todo = [(b, k) for b in BACKENDS for k in TOP_KS]
    print(f"[G] {len(todo)} cells x {n_q} questions = {len(todo) * n_q} answers", flush=True)

    done: dict[str, dict] = {}
    if ANSWERS_PATH.exists():
        done = {r["key"]: r for r in json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))}
        print(f"[G] resuming: {len(done)} answers already generated", flush=True)

    # Guard: llama-server truncates silently, so verify every cell's worst-case prompt
    # fits one slot BEFORE spending an hour generating quietly-truncated answers.
    worst = 0
    for backend, top_k in todo:
        for r in retrieval[backend]:
            chars = sum(len(c["text"]) for c in r["chunks"][:top_k])
            worst = max(worst, chars)
    est_tok = worst / 2.6 + 400          # +400 for system prompt, template, answer
    print(f"[G] slot_ctx={GEN_SLOT_CTX} (ctx {GEN_CTX} / np {GEN_NP}); "
          f"worst-case prompt ~{est_tok:.0f} tok", flush=True)
    if est_tok > GEN_SLOT_CTX:
        raise SystemExit(
            f"slot context {GEN_SLOT_CTX} too small for the worst-case prompt "
            f"(~{est_tok:.0f} tok). Raise GEN_SLOT_CTX or lower GEN_NP.")

    cfg = DEFAULT
    srv = LocalServer(np=GEN_NP, ctx=GEN_CTX, port=GEN_PORT, model=str(QWOPUS_4B))
    ts = time.strftime("%Y%m%d_%H%M%S")
    srv.start(log_path=str(RUNS / f"llama_grid_gen_{ts}.log"))
    t_start = time.time()
    n_new = 0
    try:
        for backend, top_k in todo:
            rows = retrieval[backend]
            pending = [r for r in rows if cell_key(backend, top_k, r["id"]) not in done]
            if not pending:
                print(f"[G] {backend} k={top_k}: cached", flush=True)
                continue

            def one(r):
                chunks = r["chunks"][:top_k]
                messages = build_messages(r["question"], chunks, cfg)
                t0 = time.perf_counter()
                # closed-<think> prefill is the ONLY route that suppresses this model's
                # reasoning (docs/streaming-research-findings.md §4.3)
                text = srv.complete_no_think(messages, max_tokens=cfg.max_tokens,
                                             temperature=cfg.temperature)
                return {
                    "key": cell_key(backend, top_k, r["id"]),
                    "backend": backend, "top_k": top_k, "id": r["id"],
                    "question": r["question"], "reference": r["reference"],
                    "topic": r.get("topic", ""),
                    "answer": text.strip(),
                    "gen_s": round(time.perf_counter() - t0, 2),
                }

            with ThreadPoolExecutor(max_workers=GEN_NP) as ex:
                futs = [ex.submit(one, r) for r in pending]
                for f in as_completed(futs):
                    row = f.result()
                    done[row["key"]] = row
                    n_new += 1
            ANSWERS_PATH.write_text(json.dumps(list(done.values()), ensure_ascii=False),
                                    encoding="utf-8")
            el = time.time() - t_start
            print(f"[G] {backend} k={top_k}: {len(pending)} done "
                  f"({el / 60:.1f}m elapsed, {n_new} new total)", flush=True)
    finally:
        srv.stop()
    print(f"[G] wrote {len(done)} answers -> {ANSWERS_PATH.name}")


# --------------------------------------------------------------------------- #
# phase J: judge
# --------------------------------------------------------------------------- #

def judge_prompt(unit: dict, chunks: list[dict]) -> list[dict]:
    ctx = "\n\n".join(f"[{i+1}] ({c['source']}, п. {c.get('point') or '?'})\n{c['text']}"
                      for i, c in enumerate(chunks))
    user = (f"ВОПРОС АБИТУРИЕНТА: {unit['question']}\n\n"
            f"ФРАГМЕНТЫ ИЗ RAG (top-{unit['top_k']}):\n{ctx}\n\n"
            f"ОТВЕТ МОДЕЛИ: {unit['answer']}\n\n"
            f"ЭТАЛОН (reference): {unit['reference']}\n\n"
            "Оцени chunks / answer и выдай JSON по формату из системного сообщения.")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def extract_json(text: str) -> dict | None:
    m = JSON_RE.search(text) or re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    for cand in (m.group(0), m.group(0).replace("'", '"')):
        try:
            return json.loads(cand)
        except Exception:  # noqa: BLE001
            continue
    return None


def cmd_judge() -> None:
    answers = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    chunks_by = {b: {r["id"]: r["chunks"] for r in load_retrieval(b)} for b in BACKENDS}

    done: dict[str, dict] = {}
    if JUDGE_PARTIAL.exists():
        done = {r["key"]: r for r in json.loads(JUDGE_PARTIAL.read_text(encoding="utf-8"))}
        print(f"[J] resuming: {len(done)} already judged", flush=True)
    todo = [a for a in answers if a["key"] not in done]
    print(f"[J] judging {len(todo)}/{len(answers)} units, reasoning ON, np=5", flush=True)

    srv = LocalServer(np=5, ctx=50000, port=JUDGE_PORT)   # 9B judge, default model
    ts = time.strftime("%Y%m%d_%H%M%S")
    srv.start(log_path=str(RUNS / f"llama_grid_judge_{ts}.log"))
    t0 = time.time()
    n = 0
    try:
        def one(a):
            chunks = chunks_by[a["backend"]][a["id"]][: a["top_k"]]
            r = srv.chat(judge_prompt(a, chunks), max_tokens=1200, temperature=0.2)
            p = extract_json(r["content"])
            return {**{k: a[k] for k in ("key", "backend", "top_k", "id", "topic")},
                    "chunks_score": p.get("chunks_score") if p else None,
                    "answer_score": p.get("answer_score") if p else None,
                    "comment": p.get("comment") if p else f"[PARSE FAIL] {r['content'][:200]}",
                    "judge_gen_s": round(r["gen_s"], 1)}

        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(one, a) for a in todo]
            for f in as_completed(futs):
                row = f.result()
                done[row["key"]] = row
                n += 1
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    eta = (len(todo) - n) / (n / el) if n else 0
                    print(f"[J] {n}/{len(todo)}  ({el/60:.1f}m, ETA {eta/60:.1f}m)", flush=True)
                    JUDGE_PARTIAL.write_text(
                        json.dumps(list(done.values()), ensure_ascii=False), encoding="utf-8")
    finally:
        srv.stop()

    results = list(done.values())
    agg = aggregate(results)
    JUDGE_PATH.write_text(json.dumps({"aggregate": agg, "results": results},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    if JUDGE_PARTIAL.exists():
        JUDGE_PARTIAL.unlink()
    report(agg)


def aggregate(results: list[dict]) -> dict:
    def avg(vals):
        v = [x for x in vals if isinstance(x, (int, float))]
        return round(sum(v) / len(v), 3) if v else None

    cells = {}
    for r in results:
        cells.setdefault((r["backend"], r["top_k"]), []).append(r)
    by_cell = {f"{b}|k{k}": {"n": len(rs),
                             "chunks": avg(x["chunks_score"] for x in rs),
                             "answer": avg(x["answer_score"] for x in rs)}
               for (b, k), rs in sorted(cells.items())}
    by_backend, by_k = {}, {}
    for r in results:
        by_backend.setdefault(r["backend"], []).append(r)
        by_k.setdefault(r["top_k"], []).append(r)
    return {
        "n_total": len(results),
        "n_parse_fail": sum(1 for r in results if r["answer_score"] is None),
        "by_cell": by_cell,
        "by_backend": {b: {"n": len(rs), "chunks": avg(x["chunks_score"] for x in rs),
                           "answer": avg(x["answer_score"] for x in rs)}
                       for b, rs in by_backend.items()},
        "by_top_k": {str(k): {"n": len(rs), "chunks": avg(x["chunks_score"] for x in rs),
                              "answer": avg(x["answer_score"] for x in rs)}
                     for k, rs in sorted(by_k.items())},
    }


def report(agg: dict) -> None:
    print(f"\n=== ИТОГ (n={agg['n_total']}, парсинг не удался: {agg['n_parse_fail']}) ===")
    print("\nпо top_k:")
    for k, s in agg["by_top_k"].items():
        print(f"  top_k={k:3s}  n={s['n']:4d}  chunks={s['chunks']}  answer={s['answer']}")
    print("\nпо бэкенду:")
    for b, s in agg["by_backend"].items():
        print(f"  {b:14s} n={s['n']:4d}  chunks={s['chunks']}  answer={s['answer']}")
    print("\nпо ячейкам:")
    for c, s in agg["by_cell"].items():
        print(f"  {c:24s} n={s['n']:4d}  chunks={s['chunks']}  answer={s['answer']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "generate":
        cmd_generate()
    elif cmd == "judge":
        cmd_judge()
    elif cmd == "report":
        report(json.loads(JUDGE_PATH.read_text(encoding="utf-8"))["aggregate"])
    else:
        cmd_generate()
        cmd_judge()
