"""Comprehensive semantic judge over the model x knowledge-base grid.

For every (base, model) pair, judges each of the 100 questions on 3 pipeline
steps separately: rephrase quality, retrieved-chunks quality, final-answer
quality (0..1 each) + a 1-sentence comment pointing at the weakest step.
Bases: baseline-raw / kb-relevant / kb-full. Models: qwen3.5-9b, qwen3.6-27b,
deepseek-v4-flash (OpenRouter, already answered in runs/models_*.json) PLUS
local-2B on baseline-raw only (runs/compare_merged.json, 1st repeat).

Judge model: local Qwen3.5-9B via llama-server, np=5 / ctx=50000 (10000/slot),
REASONING LEFT ON (long chain-of-thought allowed) -- this is a quality judge,
not the production voice path.

Phase A (fast, no-think): rephrase_canonical for the 100 distinct questions.
Phase B (local, no LLM): retrieval-only search per (base, question) -> top-5
  chunk texts, by rebuilding each of the 3 indexes in turn.
Phase C (slow, reasoning-on): 1000 judge calls, 5 parallel workers.

Usage: uv run python experiments-rag-params/semantic_judge_run.py
Needs: rag.rephrase's REPHRASE_SYSTEM/FEWSHOT (imported, not re-run through the
prod LlamaServer -- driven here through LocalServer.complete_no_think instead).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag.config import DEFAULT
from rag.index import Indexes
from rag.pipeline import Pipeline
from rag.rephrase import REPHRASE_SYSTEM, FEWSHOT, _extract_question
from llama_local import LocalServer

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(__file__).resolve().parent / "runs"
POOL = Path(__file__).resolve().parent / "eval_set_100.json"
MODELS_FILES = {
    "baseline-raw": "models_baseline-raw_20260719_121735.json",
    "kb-relevant": "models_kb-relevant_20260719_122813.json",
    "kb-full": "models_kb-full_20260719_123733.json",
}
COMPARE_MERGED = RUNS / "compare_merged.json"
OR_MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.6-27b", "deepseek/deepseek-v4-flash"]
LOCAL_2B_MODEL = "local-qwen3.5-2b"

JUDGE_SYSTEM = (
    "Ты — строгий аудитор качества RAG-конвейера приёмной комиссии юридического "
    "вуза МВД (голосовой ассистент для абитуриентов). Тебе даны 4 артефакта одного "
    "запроса: исходный вопрос абитуриента, канонический rephrase этого вопроса, "
    "фрагменты документов, найденные RAG-поиском по rephrase, и финальный ответ "
    "модели на исходный вопрос. Также дан эталонный ответ (reference) для сверки "
    "фактов — эталон короче и суше, это не образец стиля, а источник истины.\n\n"
    "Оцени 3 шага конвейера НЕЗАВИСИМО:\n"
    "1. rephrase_score (0..1): сохранил ли rephrase суть и предмет вопроса, термины, "
    "не подменил ли тему.\n"
    "2. chunks_score (0..1): есть ли среди фрагментов фактический материал, "
    "достаточный чтобы ПРАВИЛЬНО ответить на вопрос (recall); низкий балл если "
    "нужного факта в чанках вообще нет, даже если модель как-то выкрутилась.\n"
    "3. answer_score (0..1): отвечает ли финальный ответ точно на вопрос, без "
    "выдуманных фактов, согласуясь с эталоном по сути (не по формулировке).\n\n"
    "Думай пошагово сколько нужно. В САМОМ КОНЦЕ ответа выведи РОВНО ОДИН JSON-"
    "объект и больше НИЧЕГО после него, в формате:\n"
    '{"rephrase_score": 0.0, "chunks_score": 0.0, "answer_score": 0.0, '
    '"comment": "..."}\n'
    "comment — одно предложение по-русски: какой шаг слабее всего и почему "
    "(конкретно, без общих слов), или что именно отработало хорошо если всё ок."
)

JSON_RE = re.compile(r"\{[^{}]*\"rephrase_score\"[^{}]*\}", re.DOTALL)


def rebuild_index(label: str):
    env_args = dict(cwd=str(REPO))
    if label == "baseline-raw":
        subprocess.run(["uv", "run", "python", "-m", "rag.ingest"], check=True, **env_args)
    elif label == "kb-relevant":
        subprocess.run(["uv", "run", "python", "experiments-rag-params/build_wiki_index.py"],
                        check=True, **env_args)
    elif label == "kb-full":
        subprocess.run(["uv", "run", "python", "experiments-rag-params/build_wiki_index.py",
                         "E:/voice-agent/enrollment-assistant/data/npa/knowledge-base-full/wiki"],
                        check=True, **env_args)
    subprocess.run(["uv", "run", "python", "-m", "rag.index"], check=True, **env_args)


def build_dataset(srv: LocalServer) -> list[dict]:
    pool = json.load(open(POOL, encoding="utf-8"))
    merged2b = json.load(open(COMPARE_MERGED, encoding="utf-8"))["rows"]
    merged2b_by_id = {r["id"]: r for r in merged2b}

    # ---- Phase A: rephrase, dedup by question text (base-independent) ----
    print(f"[A] rephrase: {len(pool)} questions, no-think, np={srv.np} ...")
    rephrase_cache: dict[str, str] = {}
    t0 = time.time()

    def do_rephrase(q):
        messages = [{"role": "system", "content": REPHRASE_SYSTEM}, *FEWSHOT,
                    {"role": "user", "content": q}]
        raw = srv.complete_no_think(messages, max_tokens=150)
        return q, (_extract_question(raw) or q)

    with ThreadPoolExecutor(max_workers=srv.np) as ex:
        futs = [ex.submit(do_rephrase, q["question"]) for q in pool]
        for i, f in enumerate(as_completed(futs)):
            q, canon = f.result()
            rephrase_cache[q] = canon
            if (i + 1) % 20 == 0:
                print(f"    rephrase {i + 1}/{len(pool)}  ({time.time() - t0:.0f}s)")
    print(f"[A] done in {time.time() - t0:.0f}s")

    # ---- Phase B: retrieval-only per base (local, no LLM) ----
    units = []
    for label in ["baseline-raw", "kb-relevant", "kb-full"]:
        print(f"[B] rebuilding index for {label} ...")
        rebuild_index(label)
        idx = Indexes()
        pipe = Pipeline(idx, server=None, cfg=DEFAULT)
        or_data = json.load(open(RUNS / MODELS_FILES[label], encoding="utf-8"))
        or_rows = {r["id"]: r for r in or_data["rows"]}
        print(f"[B] retrieval for {label}: {len(pool)} questions, {len(idx.chunks)} chunks")
        for q in pool:
            top, _ = pipe.search(q["question"])
            chunks = [{"source": c["source"], "point": c.get("point"), "text": c["text"]}
                      for c in top]
            row = or_rows[q["id"]]
            base_unit = dict(id=q["id"], base=label, question=q["question"],
                              topic=q.get("topic", ""), reference=q.get("reference", ""),
                              rephrase=rephrase_cache.get(q["question"], q["question"]),
                              chunks=chunks)
            for m in OR_MODELS:
                ans = row["answers"][m]
                units.append({**base_unit, "model": m, "answer": ans})
            if label == "baseline-raw":
                r2b = merged2b_by_id.get(q["id"])
                if r2b:
                    units.append({**base_unit, "model": LOCAL_2B_MODEL, "answer": r2b["baseline"]})
    print(f"[B] dataset built: {len(units)} judge units")
    return units


def extract_json(text: str) -> dict | None:
    m = JSON_RE.search(text)
    if not m:
        m2 = re.search(r"\{.*\}", text, re.DOTALL)  # loose fallback
        if not m2:
            return None
        cand = m2.group(0)
    else:
        cand = m.group(0)
    try:
        return json.loads(cand)
    except Exception:
        cand2 = cand.replace("'", '"')
        try:
            return json.loads(cand2)
        except Exception:
            return None


def judge_prompt(u: dict) -> list[dict]:
    ctx = "\n\n".join(f"[{i+1}] ({c['source']}, п. {c.get('point') or '?'})\n{c['text']}"
                       for i, c in enumerate(u["chunks"]))
    user = (
        f"ИСХОДНЫЙ ВОПРОС: {u['question']}\n\n"
        f"REPHRASE (канонический вопрос для поиска): {u['rephrase']}\n\n"
        f"ФРАГМЕНТЫ ИЗ RAG (top-5):\n{ctx}\n\n"
        f"ФИНАЛЬНЫЙ ОТВЕТ МОДЕЛИ ({u['model']}): {u['answer']}\n\n"
        f"ЭТАЛОН (reference): {u['reference']}\n\n"
        "Оцени rephrase / chunks / answer и выдай JSON по формату из системного сообщения."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def unit_key(u: dict) -> str:
    return f"{u['base']}|{u['model']}|{u['id']}"


PARTIAL_PATH = RUNS / "judge_results_partial.json"


def run_judge(srv: LocalServer, units: list[dict]) -> list[dict]:
    # Resume support: skip units already judged in a prior (interrupted) run.
    done_map: dict[str, dict] = {}
    if PARTIAL_PATH.exists():
        prev = json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
        done_map = {unit_key(r): r for r in prev}
        print(f"[C] resuming: {len(done_map)} units already judged in a prior run")

    results = [None] * len(units)
    todo = []
    for i, u in enumerate(units):
        cached = done_map.get(unit_key(u))
        if cached:
            results[i] = cached
        else:
            todo.append((i, u))

    print(f"[C] judging {len(todo)}/{len(units)} remaining units, reasoning ON, "
          f"np={srv.np}, max_tokens=1200 ...")
    t0 = time.time()
    done = 0
    lock_results = []  # completed-this-run, for periodic checkpoint writes

    def do_one(i, u):
        r = srv.chat(judge_prompt(u), max_tokens=1200, temperature=0.2)
        parsed = extract_json(r["content"])
        out = dict(u)
        del out["chunks"]  # keep results file small; chunks live in judge_dataset.json
        out.update({
            "rephrase_score": parsed.get("rephrase_score") if parsed else None,
            "chunks_score": parsed.get("chunks_score") if parsed else None,
            "answer_score": parsed.get("answer_score") if parsed else None,
            "comment": parsed.get("comment") if parsed else f"[PARSE FAIL] {r['content'][:200]}",
            "judge_tokens": r["tokens"], "judge_gen_s": round(r["gen_s"], 1),
            "judge_finish": r["finish"],
        })
        return i, out

    with ThreadPoolExecutor(max_workers=srv.np) as ex:
        futs = [ex.submit(do_one, i, u) for i, u in todo]
        for f in as_completed(futs):
            i, out = f.result()
            results[i] = out
            lock_results.append(out)
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(f"    judged {done}/{len(todo)} new  ({el/60:.1f}m elapsed, "
                      f"ETA {eta/60:.1f}m)")
                # checkpoint: everything resolved so far (resumed + this run)
                PARTIAL_PATH.write_text(
                    json.dumps([r for r in results if r is not None], ensure_ascii=False),
                    encoding="utf-8")
    return results


def aggregate(results: list[dict]) -> dict:
    def avg(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    by_model, by_base, by_pair = {}, {}, {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
        by_base.setdefault(r["base"], []).append(r)
        by_pair.setdefault((r["base"], r["model"]), []).append(r)
    n_parse_fail = sum(1 for r in results if r["rephrase_score"] is None)

    def summarize(rows):
        return {"n": len(rows),
                "rephrase": avg(r["rephrase_score"] for r in rows),
                "chunks": avg(r["chunks_score"] for r in rows),
                "answer": avg(r["answer_score"] for r in rows)}

    return {
        "n_total": len(results), "n_parse_fail": n_parse_fail,
        "by_model": {m: summarize(rs) for m, rs in by_model.items()},
        "by_base": {b: summarize(rs) for b, rs in by_base.items()},
        "by_base_model": {f"{b}|{m}": summarize(rs) for (b, m), rs in by_pair.items()},
    }


NEW_DATASET_PATH = RUNS / "judge_dataset_new_latest.json"


def build_new_model_units(srv: LocalServer, specs: list[tuple[str, Path]]) -> list[dict]:
    """Build baseline-raw-only judge units for one or more NEW local models, each
    already answered offline (see experiments-rag-params/new_model_answers.py) into
    a compare_<label>_<ts>.json file (rows[i]["answers"][0] is used). Mirrors the
    LOCAL_2B_MODEL slice of build_dataset() -- same rephrase + retrieval, just for
    whichever models are being added this run. Assumes the baseline-raw index is
    the one currently built in rag/artifacts (caller's job to rebuild_index first).
    Loads Indexes()/Pipeline (embed+rerank models) -- MUST run in a separate
    process from the judge LocalServer (see cmd_new_dataset / cmd_judge_add),
    same VRAM-contention reason as the cmd_dataset/cmd_judge split above."""
    pool = json.load(open(POOL, encoding="utf-8"))
    answers_by_label = {}
    for label, path in specs:
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        answers_by_label[label] = {r["id"]: r["answers"][0] for r in rows}

    print(f"[A'] rephrase (new models): {len(pool)} questions, no-think, np={srv.np} ...")
    rephrase_cache: dict[str, str] = {}

    def do_rephrase(q):
        messages = [{"role": "system", "content": REPHRASE_SYSTEM}, *FEWSHOT,
                    {"role": "user", "content": q}]
        raw = srv.complete_no_think(messages, max_tokens=150)
        return q, (_extract_question(raw) or q)

    with ThreadPoolExecutor(max_workers=srv.np) as ex:
        futs = [ex.submit(do_rephrase, q["question"]) for q in pool]
        for f in as_completed(futs):
            q, canon = f.result()
            rephrase_cache[q] = canon

    idx = Indexes()
    pipe = Pipeline(idx, server=None, cfg=DEFAULT)
    print(f"[B'] retrieval (new models) on baseline-raw: {len(pool)} questions, "
          f"{len(idx.chunks)} chunks")
    units = []
    for q in pool:
        top, _ = pipe.search(q["question"])
        chunks = [{"source": c["source"], "point": c.get("point"), "text": c["text"]}
                  for c in top]
        base_unit = dict(id=q["id"], base="baseline-raw", question=q["question"],
                          topic=q.get("topic", ""), reference=q.get("reference", ""),
                          rephrase=rephrase_cache.get(q["question"], q["question"]),
                          chunks=chunks)
        for label, _ in specs:
            ans = answers_by_label[label].get(q["id"])
            if ans is None:
                continue
            units.append({**base_unit, "model": label, "answer": ans})
    print(f"[B'] new-model dataset built: {len(units)} judge units")
    return units


def cmd_new_dataset(specs: list[tuple[str, Path]]):
    """Phase 1 of judge-add, run in its OWN process (spawned by cmd_judge_add):
    rephrase (small server) + retrieval (embed/rerank models loaded HERE only).
    Writes judge_dataset_new_latest.json and exits -- exiting frees the VRAM
    those models hold, same trick as cmd_dataset()."""
    RUNS.mkdir(exist_ok=True)
    srv = LocalServer(np=5, ctx=50000, port=20099)
    srv.start(log_path=str(RUNS / f"llama_newdataset_np5_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    try:
        units = build_new_model_units(srv, specs)
    finally:
        srv.stop()
    NEW_DATASET_PATH.write_text(json.dumps(units, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[new-dataset] wrote {len(units)} units -> {NEW_DATASET_PATH}")


def cmd_judge_add(specs: list[tuple[str, Path]]):
    """Judge one or more NEW local models (baseline-raw only, same protocol as
    LOCAL_2B_MODEL) and MERGE the results into judge_results_latest.json instead
    of overwriting it -- so the file keeps accumulating a comprehensive record
    across every model evaluated so far, old runs included.

    Two-process split (this orchestrator process never loads Indexes()/Pipeline
    itself -- it only shells out for that phase and for rebuild_index -- so the
    judge LocalServer below always gets full, uncontended VRAM):
      1. subprocess: rebuild_index("baseline-raw") + rephrase + retrieval, exits.
      2. here: judge LocalServer (np=5, ctx=50000) loaded fresh, judges the new
         units, then merges with the existing judge_results_latest.json.

    Usage: uv run python experiments-rag-params/semantic_judge_run.py judge-add \
        label1=runs/compare_label1_TS.json [label2=runs/compare_label2_TS.json ...]
    """
    RUNS.mkdir(exist_ok=True)
    print("[judge-add] rebuilding baseline-raw index (retrieval must match what "
          "these models actually saw) ...")
    rebuild_index("baseline-raw")

    print("[judge-add] building new-model dataset in an isolated subprocess "
          "(embed/rerank models must not share VRAM with the judge server) ...")
    specs_args = [f"{label}={path}" for label, path in specs]
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "new-dataset", *specs_args],
                    check=True, cwd=str(REPO))
    units = json.loads(NEW_DATASET_PATH.read_text(encoding="utf-8"))

    ts = time.strftime("%Y%m%d_%H%M%S")
    srv = LocalServer(np=5, ctx=50000, port=20099)
    srv.start(log_path=str(RUNS / f"llama_judgeadd_np5_{ts}.log"))
    try:
        new_results = run_judge(srv, units)
    finally:
        srv.stop()

    latest_path = RUNS / "judge_results_latest.json"
    prior_results = []
    if latest_path.exists():
        prior_results = json.loads(latest_path.read_text(encoding="utf-8"))["results"]
    prior_keys = {unit_key(r) for r in prior_results}
    new_keys = {unit_key(r) for r in new_results}
    overlap = prior_keys & new_keys
    if overlap:
        print(f"[judge-add] {len(overlap)} unit(s) already present in judge_results_latest.json "
              f"-- new results for these keys replace the old ones")
    merged = [r for r in prior_results if unit_key(r) not in new_keys] + new_results

    agg = aggregate(merged)
    out_path = RUNS / f"judge_results_{ts}.json"
    payload = {"meta": {"ts": ts, "n": len(merged),
                         "note": f"cumulative: {len(prior_results)} prior + "
                                 f"{len(new_results)} added this run "
                                 f"({', '.join(l for l, _ in specs)})"},
               "aggregate": agg, "results": merged}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()

    print(f"\n=== AGGREGATE (cumulative, n={len(merged)}) ===")
    print(f"parse failures: {agg['n_parse_fail']}/{agg['n_total']}")
    print("\nby model:")
    for m, s in agg["by_model"].items():
        print(f"  {m:28s} n={s['n']:4d}  rephrase={s['rephrase']}  chunks={s['chunks']}  answer={s['answer']}")
    print("\nby base:")
    for b, s in agg["by_base"].items():
        print(f"  {b:14s} n={s['n']:4d}  rephrase={s['rephrase']}  chunks={s['chunks']}  answer={s['answer']}")
    print("\nby base x model:")
    for k, s in sorted(agg["by_base_model"].items()):
        print(f"  {k:42s} n={s['n']:4d}  rephrase={s['rephrase']}  chunks={s['chunks']}  answer={s['answer']}")
    print(f"\n-> {out_path.name} (judge_results_latest.json updated in place)")


def cmd_dataset():
    """Phase A+B only: rephrase (small server) + retrieval (embed/rerank models
    loaded in THIS process). Writes judge_dataset_latest.json and exits -- exiting
    the process is what actually frees the embed/rerank models' VRAM, which is the
    point of splitting this from the judge phase (see cmd_judge)."""
    RUNS.mkdir(exist_ok=True)
    srv = LocalServer(np=5, ctx=50000, port=20099)
    srv.start(log_path=str(RUNS / f"llama_rephrase_np5_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    try:
        units = build_dataset(srv)
    finally:
        srv.stop()
    out = RUNS / "judge_dataset_latest.json"
    out.write_text(json.dumps(units, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[dataset] wrote {len(units)} units -> {out}")


def cmd_judge():
    """Phase C only: fresh process, fresh llama-server, FULL VRAM available (no
    embed/rerank models were ever loaded here) -- this is what fixes the observed
    1.7 tok/s/slot collapse (VRAM contention with the retrieval models loaded in
    the same process as the dataset-build phase, ~7.7GB llama-server vs an 8GB
    card leaves ~0 headroom for a second CUDA context)."""
    RUNS.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    units = json.load(open(RUNS / "judge_dataset_latest.json", encoding="utf-8"))
    srv = LocalServer(np=5, ctx=50000, port=20099)
    srv.start(log_path=str(RUNS / f"llama_judge_np5_{ts}.log"))
    try:
        results = run_judge(srv, units)
    finally:
        srv.stop()

    out_path = RUNS / f"judge_results_{ts}.json"
    agg = aggregate(results)
    out_path.write_text(json.dumps({"meta": {"ts": ts, "n": len(results)},
                                     "aggregate": agg, "results": results},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    (RUNS / "judge_results_latest.json").write_text(
        json.dumps({"meta": {"ts": ts, "n": len(results)}, "aggregate": agg, "results": results},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()  # done -- don't let a future run "resume" from stale data

    print("\n=== AGGREGATE (rephrase / chunks / answer, 0..1) ===")
    print(f"parse failures: {agg['n_parse_fail']}/{agg['n_total']}")
    print("\nby model:")
    for m, s in agg["by_model"].items():
        print(f"  {m:28s} n={s['n']:4d}  rephrase={s['rephrase']}  chunks={s['chunks']}  answer={s['answer']}")
    print("\nby base:")
    for b, s in agg["by_base"].items():
        print(f"  {b:14s} n={s['n']:4d}  rephrase={s['rephrase']}  chunks={s['chunks']}  answer={s['answer']}")
    print("\nby base x model:")
    for k, s in sorted(agg["by_base_model"].items()):
        print(f"  {k:42s} n={s['n']:4d}  rephrase={s['rephrase']}  chunks={s['chunks']}  answer={s['answer']}")
    print(f"\n-> {out_path.name}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "dataset":
        cmd_dataset()
    elif cmd == "judge":
        cmd_judge()
    elif cmd in ("judge-add", "new-dataset"):
        specs = []
        for arg in sys.argv[2:]:
            label, _, path = arg.partition("=")
            if not path:
                raise SystemExit(f"bad spec {arg!r}, expected label=path/to/compare_*.json")
            specs.append((label, Path(path)))
        if not specs:
            raise SystemExit(f"{cmd} needs at least one label=path spec")
        cmd_judge_add(specs) if cmd == "judge-add" else cmd_new_dataset(specs)
    else:
        cmd_dataset()
        cmd_judge()
