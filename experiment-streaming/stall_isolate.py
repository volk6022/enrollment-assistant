"""Isolate the one-off ~1 s event-loop stall seen in streaming_poc.py PART A.

In the contention matrix the FIRST combo (`stt solo`) showed loop_max = 1018 ms, while
every later combo — including all three components at once — stayed at 1.7-34 ms. A
one-second stall that happens once and never again smells like initialisation, not
steady-state contention, but "smells like" is not a measurement.

Two candidates were introduced in that run and are tested separately here:
  1. ResourceSampler — calls `nvidia-smi` via subprocess every 250 ms from a thread.
     The first NVML handshake is known to be slow, and process spawning on Windows is
     not free.
  2. Whisper's first CUDA call on a FRESH worker thread — a per-thread CUDA context
     may get created under the GIL even though steady-state inference does not hold it.

Four arms, each probing the loop the same way gil_probe.py does:
  A. sampler only, no STT          -> blames nvidia-smi if it stalls
  B. STT only, no sampler          -> blames the CUDA-context-on-new-thread path
  C. STT + sampler (as in the POC) -> should reproduce the original stall
  D. STT + sampler, second round   -> confirms it is one-off, not periodic

Run with nothing else on the GPU except llama-server.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gil_probe import Probe, TIGHT_TICK, TIMER_TICK, load_wav_f32  # noqa: E402
from streaming_poc import ResourceSampler, SAMPLE_WAV  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs" / "stall_isolate.json"
ROUNDS = 3


async def arm(label: str, *, use_stt: bool, use_sampler: bool, model, audio,
              pool: ThreadPoolExecutor | None) -> dict:
    tight, timer = Probe("t", TIGHT_TICK), Probe("m", TIMER_TICK)
    tt, tm = asyncio.create_task(tight.run()), asyncio.create_task(timer.run())
    await asyncio.sleep(0.15)

    sampler = ResourceSampler() if use_sampler else None
    if sampler:
        sampler.start()

    per_call = []
    if use_stt:
        loop = asyncio.get_running_loop()

        def once():
            segs, _ = model.transcribe(audio, language="ru", beam_size=5, vad_filter=True)
            return " ".join(s.text.strip() for s in segs)

        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            await loop.run_in_executor(pool, once)
            per_call.append(round(time.perf_counter() - t0, 3))
    else:
        await asyncio.sleep(ROUNDS * 1.0)

    res = sampler.stop() if sampler else {}
    tight.stop(); timer.stop()
    await asyncio.gather(tt, tm)
    out = {"label": label, "stt_calls_s": per_call,
           "tight_max_ms": tight.stats().get("gap_ms_max"),
           "tight_p99_ms": tight.stats().get("gap_ms_p99"),
           "timer_max_ms": timer.stats().get("gap_ms_max"),
           "sampler": res}
    print(f"{label:34s} loop_max={out['tight_max_ms']:9.3f}ms  "
          f"p99={out['tight_p99_ms']:7.3f}ms  stt={per_call}", flush=True)
    return out


async def main() -> None:
    from faster_whisper import WhisperModel
    print("loading whisper (cuda)...", flush=True)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
    audio, _ = load_wav_f32(SAMPLE_WAV)
    # warm in the MAIN thread — exactly what streaming_poc.py load() does
    list(model.transcribe(audio, language="ru", beam_size=5)[0])
    print("warm done\n", flush=True)

    rows = []
    # A: sampler alone — no STT, no worker thread
    rows.append(await arm("A sampler only", use_stt=False, use_sampler=True,
                          model=model, audio=audio, pool=None))

    # B: STT on a FRESH pool, no sampler
    poolB = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sttB")
    rows.append(await arm("B stt only (fresh thread)", use_stt=True, use_sampler=False,
                          model=model, audio=audio, pool=poolB))
    poolB.shutdown(wait=True)

    # C: STT on ANOTHER fresh pool + sampler — reproduces the POC's first combo
    poolC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sttC")
    rows.append(await arm("C stt + sampler (fresh thread)", use_stt=True, use_sampler=True,
                          model=model, audio=audio, pool=poolC))
    # D: same pool again — thread already initialised
    rows.append(await arm("D stt + sampler (same thread)", use_stt=True, use_sampler=True,
                          model=model, audio=audio, pool=poolC))
    poolC.shutdown(wait=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
