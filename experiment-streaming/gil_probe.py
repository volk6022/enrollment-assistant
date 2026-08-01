"""GIL-contention probe: can Whisper / Silero run in a thread without stalling asyncio?

The streaming rewrite hinges on one question per component: when we call it from a
worker thread (`loop.run_in_executor`), does the native code RELEASE the GIL so the
asyncio event loop keeps servicing the websocket, or does it HOLD the GIL for the whole
call and freeze everything?

Method — two probes running on the event loop while a workload runs in a thread pool:

  probe_tight   `await asyncio.sleep(0)` in a tight loop, timestamping every iteration.
                The MAX GAP between iterations is (near enough) the longest continuous
                GIL hold by the worker. This is the discriminating metric:

                  gap <=  ~20 ms  -> GIL is released / handed over regularly. Fine.
                                     (CPython's switch interval is 5 ms, so even a
                                     pure-Python hog yields at ~5-15 ms.)
                  gap >= ~100 ms  -> native call holds the GIL for its duration.
                                     BLOCKER: must move to its own process.

  probe_timer   `await asyncio.sleep(0.02)` — a realistic 20 ms scheduled heartbeat, the
                way a real streaming loop paces itself. Reports how late each tick fired.
                (Windows timer granularity is ~15 ms, so baseline lateness is nonzero;
                always read this against the `noop` baseline, never in absolute terms.)

Reference workloads bracket the measurement so the numbers are interpretable:
  noop     idle           -> floor: what the probes read with nothing running
  pyloop   pure Python    -> ceiling for a GIL-holding-but-yielding workload (~5-15 ms)
  npmatmul numpy BLAS     -> known GIL-RELEASING native code (should look like noop)

Usage (from the repo root, with the root .venv):
    .venv\\Scripts\\python.exe experiment-streaming\\gil_probe.py --workload noop
    .venv\\Scripts\\python.exe experiment-streaming\\gil_probe.py --workload whisper_cpu
    .venv\\Scripts\\python.exe experiment-streaming\\gil_probe.py --all --json out.json

Model loading happens BEFORE measurement starts (loading is a one-off at boot and would
otherwise dominate); only steady-state inference is measured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SAMPLE_WAV = REPO / "experiment-stt" / "test_data" / "sample_01.wav"
TTS_TEXT = (
    "Для поступления нужен паспорт, документ об образовании, "
    "медицинская справка и четыре фотографии размером три на четыре."
)

TIGHT_TICK = 0.0
TIMER_TICK = 0.020


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #

class Probe:
    """Records inter-iteration gaps of a coroutine spinning on the event loop."""

    def __init__(self, name: str, tick: float) -> None:
        self.name = name
        self.tick = tick
        self.gaps: list[float] = []
        self._stop = False

    async def run(self) -> None:
        prev = time.perf_counter()
        while not self._stop:
            await asyncio.sleep(self.tick)
            now = time.perf_counter()
            self.gaps.append(now - prev)
            prev = now

    def stop(self) -> None:
        self._stop = True

    def stats(self) -> dict:
        if not self.gaps:
            return {"ticks": 0}
        g = sorted(self.gaps)
        ms = lambda x: round(x * 1000, 3)  # noqa: E731

        def pct(p: float) -> float:
            return g[min(len(g) - 1, int(len(g) * p))]

        # "lateness" is only meaningful for the timer probe
        late = [max(0.0, x - self.tick) for x in self.gaps]
        return {
            "ticks": len(g),
            "gap_ms_p50": ms(pct(0.50)),
            "gap_ms_p95": ms(pct(0.95)),
            "gap_ms_p99": ms(pct(0.99)),
            "gap_ms_max": ms(g[-1]),
            "blocked_ms_total": ms(sum(late)),
            # how many ticks were delayed past a 50 ms "user would notice" line
            "ticks_over_50ms": sum(1 for x in g if x > 0.050),
        }


# --------------------------------------------------------------------------- #
# workloads — each returns a zero-arg callable, already warmed up
# --------------------------------------------------------------------------- #

def load_wav_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    if sr != 16000:
        import soxr
        pcm = soxr.resample(pcm, sr, 16000).astype(np.float32)
        sr = 16000
    return pcm.astype(np.float32), sr


def make_noop(_args):
    def fn():
        time.sleep(0.5)
    return fn, {"note": "idle baseline"}


def make_pyloop(_args):
    def fn():
        # pure-Python CPU burn: holds the GIL but CPython force-yields at the
        # switch interval, so this is the reference for "contended but yielding"
        t0 = time.perf_counter()
        x = 0
        while time.perf_counter() - t0 < 0.5:
            for _ in range(10000):
                x += 1
    return fn, {"note": "pure-Python GIL hog reference",
                "switch_interval_ms": sys.getswitchinterval() * 1000}


def make_npmatmul(_args):
    a = np.random.rand(1500, 1500).astype(np.float32)
    b = np.random.rand(1500, 1500).astype(np.float32)
    a @ b  # warm BLAS

    def fn():
        _ = a @ b
    return fn, {"note": "numpy BLAS reference (known to release the GIL)"}


def _make_whisper(args, device: str):
    from faster_whisper import WhisperModel
    kwargs = {"device": device, "compute_type": "int8" if device == "cpu" else "int8"}
    if args.cpu_threads:
        kwargs["cpu_threads"] = args.cpu_threads
    model = WhisperModel(args.whisper_model, **kwargs)
    audio, _sr = load_wav_f32(SAMPLE_WAV)

    def run_once():
        segs, _info = model.transcribe(audio, language="ru", beam_size=5, vad_filter=True)
        return " ".join(s.text.strip() for s in segs)

    warm = run_once()  # exclude model load + first-call kernel compile

    def fn():
        run_once()

    return fn, {
        "model": args.whisper_model,
        "device": device,
        "compute_type": kwargs["compute_type"],
        "cpu_threads": kwargs.get("cpu_threads", "default"),
        "audio_sec": round(len(audio) / 16000, 2),
        "warm_transcript": warm[:120],
    }


def make_whisper_cpu(args):
    return _make_whisper(args, "cpu")


def make_whisper_cuda(args):
    return _make_whisper(args, "cuda")


def _make_silero(args, device: str):
    import torch
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                              language="ru", speaker="v5_5_ru", trust_repo=True)
    model.to(device)

    def run_once():
        with torch.no_grad():
            wav = model.apply_tts(text=TTS_TEXT, speaker="baya", sample_rate=48000,
                                  put_accent=True, put_yo=True)
        return wav.squeeze().cpu().numpy()

    w = run_once()  # warm JIT (first call is ~10-13 s)

    def fn():
        run_once()

    return fn, {
        "version": "v5_5_ru", "speaker": "baya", "sample_rate": 48000,
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "audio_sec": round(len(w) / 48000, 2),
        "chars": len(TTS_TEXT),
    }


def make_silero_cpu(args):
    return _make_silero(args, "cpu")


def make_silero_cuda(args):
    return _make_silero(args, "cuda")


def make_dsp(_args):
    # the Silero post-chain (soxr + scipy de-esser) on 5 s of 48 kHz audio
    import soxr
    from scipy.signal import butter, sosfilt
    audio = np.random.rand(48000 * 5).astype(np.float32) * 0.5
    sos = butter(4, 5500.0 / 24000.0, btype="high", output="sos")

    def fn():
        d = soxr.resample(audio, 48000, 20000)
        u = soxr.resample(d, 20000, 48000).astype(np.float32)
        hf = sosfilt(sos, u).astype(np.float32)
        _ = u - hf + hf * 0.9
    return fn, {"note": "gateway TTS post-chain (soxr + scipy)"}


WORKLOADS = {
    "noop": make_noop,
    "pyloop": make_pyloop,
    "npmatmul": make_npmatmul,
    "whisper_cpu": make_whisper_cpu,
    "whisper_cuda": make_whisper_cuda,
    "silero_cpu": make_silero_cpu,
    "silero_cuda": make_silero_cuda,
    "dsp": make_dsp,
}


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def time_solo(fn, reps: int) -> list[float]:
    """Duration of `fn` with NO probes running.

    The tight probe spins `asyncio.sleep(0)` flat out: it burns a core and thrashes
    GIL acquire/release, which inflates the workload's own duration. So throughput is
    always taken from this probe-free pass; the probed pass is only for GIL detection.
    """
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


async def measure(fn, meta: dict, *, reps: int, threads: int, tight_probe: bool = True) -> dict:
    """Run `fn` `reps` times across `threads` worker threads while probing the loop."""
    tight = Probe("tight", TIGHT_TICK)
    timer = Probe("timer", TIMER_TICK)
    t_tight = asyncio.create_task(tight.run()) if tight_probe else None
    t_timer = asyncio.create_task(timer.run())
    await asyncio.sleep(0.15)  # let the probes settle before the load starts

    durations: list[float] = []
    lock = threading.Lock()

    def wrapped():
        t0 = time.perf_counter()
        fn()
        d = time.perf_counter() - t0
        with lock:
            durations.append(d)

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=threads)
    t0 = time.perf_counter()
    await asyncio.gather(*[loop.run_in_executor(pool, wrapped) for _ in range(reps)])
    wall = time.perf_counter() - t0
    pool.shutdown(wait=True)

    tight.stop()
    timer.stop()
    await asyncio.gather(*[t for t in (t_tight, t_timer) if t is not None])

    return {
        "meta": meta,
        "reps": reps,
        "threads": threads,
        "wall_sec": round(wall, 3),
        "call_sec_mean": round(statistics.fmean(durations), 3) if durations else None,
        "call_sec_max": round(max(durations), 3) if durations else None,
        "probe_tight": tight.stats() if tight_probe else {"skipped": True},
        "probe_timer": timer.stats(),
    }


def verdict(res: dict) -> str:
    if res["probe_tight"].get("skipped"):
        return "n/a (tight probe skipped)"
    mx = res["probe_tight"].get("gap_ms_max", 0)
    if mx < 45:
        return "RELEASES GIL — safe to run in a thread"
    if mx < 100:
        return "MOSTLY OK — brief holds, tolerable in a thread"
    return "HOLDS GIL — must be moved to a separate process"


def fmt(name: str, res: dict) -> str:
    t, m = res["probe_tight"], res["probe_timer"]
    solo = res.get("solo_sec_mean")
    slow = res.get("probe_slowdown_x")
    return "\n".join([
        f"=== {name} ===",
        f"  meta            {json.dumps(res['meta'], ensure_ascii=False)}",
        f"  reps/threads    {res['reps']} / {res['threads']}",
        f"  SOLO (no probe) mean {solo}s   <- use this for throughput/RTF",
        f"  probed          wall {res['wall_sec']}s  per-call mean {res['call_sec_mean']}s"
        f"  max {res['call_sec_max']}s  (slowdown {slow}x from the probe itself)",
        f"  tight probe     ticks={t.get('ticks')}  p50={t.get('gap_ms_p50')}ms"
        f"  p99={t.get('gap_ms_p99')}ms  MAX={t.get('gap_ms_max')}ms",
        f"  timer probe     ticks={m.get('ticks')}  p50={m.get('gap_ms_p50')}ms"
        f"  p99={m.get('gap_ms_p99')}ms  MAX={m.get('gap_ms_max')}ms"
        f"  over50ms={m.get('ticks_over_50ms')}",
        f"  VERDICT         {verdict(res)}",
    ])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=sorted(WORKLOADS))
    ap.add_argument("--all", action="store_true", help="run every workload in sequence")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1,
                    help=">1 also measures native parallel scaling")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="pin ct2/torch intra-op threads (0 = library default)")
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--no-tight", action="store_true",
                    help="skip the tight probe (measure realistic latency only)")
    ap.add_argument("--solo-reps", type=int, default=2,
                    help="probe-free timing reps for the throughput number")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    names = sorted(WORKLOADS) if args.all else [args.workload]
    if not names or names == [None]:
        ap.error("pass --workload NAME or --all")

    out: dict[str, dict] = {}
    for name in names:
        print(f"\n--- preparing {name} (loading/warming, not measured) ---", flush=True)
        try:
            fn, meta = WORKLOADS[name](args)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIPPED {name}: {type(exc).__name__}: {exc}", flush=True)
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        solo = time_solo(fn, args.solo_reps)
        res = await measure(fn, meta, reps=args.reps, threads=args.threads,
                            tight_probe=not args.no_tight)
        solo_mean = statistics.fmean(solo)
        res["solo_sec_all"] = [round(x, 3) for x in solo]
        res["solo_sec_mean"] = round(solo_mean, 3)
        res["probe_slowdown_x"] = (round(res["call_sec_mean"] / solo_mean, 2)
                                   if res["call_sec_mean"] and solo_mean else None)
        out[name] = res
        print(fmt(name, res), flush=True)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
