"""POC: Whisper (GPU) + Silero (CPU) + LLM (llama-server) running CONCURRENTLY.

`gil_probe.py` measured each component in isolation and found none of them holds the
GIL. That is necessary but not sufficient: the real question for the streaming rewrite
is what happens when all three run AT THE SAME TIME inside one asyncio process, which
is exactly the steady state of a live dialogue —

    the user is still speaking      -> whisper transcribes partials on the GPU
    the prompt is being kept warm   -> prefill updates go to llama-server
    the answer is being generated   -> tokens stream back
    the answer is being spoken      -> Silero synthesises sentence by sentence

...and often several of those overlap (barge-in, interruption, thinking-out-loud).

Two parts:

  PART A — contention matrix.
    Run each component solo, then in every pair, then all three, and report for each
    combination: per-component latency, its slowdown vs solo, and the event-loop probe.
    This isolates *which* pairing hurts, rather than just "it got slower".

  PART B — one realistic turn, end to end, on a real wav.
    Simulated mic at wall-clock pace -> partial STT -> incremental prefill -> answer
    stream -> sentence-wise TTS. Reports the timeline and the latency that actually
    matters: silence detected -> first audio ready to play.

Both parts sample VRAM and RSS throughout so memory can be planned rather than guessed.

Requires llama-server on :20055 (Qwopus3.5-4B Q4_K_M, -np 2).
Run:  .venv\\Scripts\\python.exe experiment-streaming\\streaming_poc.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

from gil_probe import Probe, TIGHT_TICK, TIMER_TICK, load_wav_f32  # noqa: E402

BASE = "http://127.0.0.1:20055"
SAMPLE_WAV = REPO / "experiment-stt" / "test_data" / "sample_01.wav"
OUT = Path(__file__).resolve().parent / "runs" / "streaming_poc.json"

NOTHINK = {"role": "assistant", "content": "<think>\n\n</think>\n\n"}
SYSTEM = ("Ты — помощница приёмной комиссии института. Отвечай кратко, 1–3 предложения, "
          "только по существу заданного вопроса.")
CONTEXT = "\n".join(
    f"[Документ {i}] Пункт {i}. Для поступления кандидат представляет заявление, "
    f"документ об образовании, медицинское заключение и результаты вступительных "
    f"испытаний. Документы принимаются с 20 июня по 31 июля."
    for i in range(1, 21)
)
TTS_SENTENCE = ("Для поступления нужны заявление, документ об образовании и "
                "медицинское заключение.")


# --------------------------------------------------------------------------- #
# resource sampling
# --------------------------------------------------------------------------- #

def vram_used_mb() -> float:
    """Total GPU memory in use, including the llama-server process (torch can't see that)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1.0


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:  # noqa: BLE001
        return -1.0


class ResourceSampler:
    """Samples VRAM/RSS on a background thread (nvidia-smi is a subprocess, so it must
    not run on the event loop)."""

    def __init__(self, period: float = 0.25) -> None:
        self.period = period
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        t0 = time.perf_counter()

        def loop():
            while not self._stop.is_set():
                self.samples.append({"t": round(time.perf_counter() - t0, 2),
                                     "vram_mb": vram_used_mb(), "rss_mb": round(rss_mb(), 1)})
                self._stop.wait(self.period)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
        if not self.samples:
            return {}
        v = [s["vram_mb"] for s in self.samples if s["vram_mb"] >= 0]
        r = [s["rss_mb"] for s in self.samples if s["rss_mb"] >= 0]
        return {"vram_mb_min": min(v) if v else None, "vram_mb_max": max(v) if v else None,
                "rss_mb_min": min(r) if r else None, "rss_mb_max": max(r) if r else None,
                "n_samples": len(self.samples)}


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #

@dataclass
class Components:
    whisper: object = None
    silero: object = None
    torch: object = None
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def load(self) -> dict:
        info = {}
        t0 = time.perf_counter()
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
        self.audio, _ = load_wav_f32(SAMPLE_WAV)
        list(self.whisper.transcribe(self.audio, language="ru", beam_size=5)[0])  # warm
        info["whisper_load_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        import torch
        self.torch = torch
        model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                                  language="ru", speaker="v5_5_ru", trust_repo=True)
        model.to("cpu")
        self.silero = model
        self.tts_once()  # warm the JIT (first call is ~10 s)
        info["silero_load_s"] = round(time.perf_counter() - t0, 2)
        return info

    # --- blocking calls, always used via run_in_executor ---
    def stt_once(self) -> str:
        segs, _ = self.whisper.transcribe(self.audio, language="ru", beam_size=5,
                                          vad_filter=True)
        return " ".join(s.text.strip() for s in segs)

    def tts_once(self) -> float:
        with self.torch.no_grad():
            wav = self.silero.apply_tts(text=TTS_SENTENCE, speaker="baya",
                                        sample_rate=48000, put_accent=True, put_yo=True)
        return len(wav) / 48000


async def llm_stream(client: httpx.AsyncClient, n_predict: int = 120) -> dict:
    """One answer generation with reasoning suppressed (the only route that works —
    see docs/streaming-research-findings.md §4.3)."""
    t0 = time.perf_counter()
    ttft = None
    gaps: list[float] = []
    prev = t0
    acc = ""
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Контекст:\n{CONTEXT}\n\n"
                                                     f"Вопрос: какие документы нужны?"},
                         NOTHINK],
            "temperature": 0.0, "max_tokens": n_predict, "stream": True}
    async with client.stream("POST", f"{BASE}/v1/chat/completions", json=body,
                             timeout=240.0) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            p = line[6:].strip()
            if p == "[DONE]":
                break
            d = (json.loads(p).get("choices") or [{}])[0].get("delta") or {}
            tok = d.get("content") or ""
            if not tok:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            else:
                gaps.append(now - prev)
            prev = now
            acc += tok
    return {"ttft_s": round(ttft, 3) if ttft else None,
            "total_s": round(time.perf_counter() - t0, 3),
            "inter_token_ms_p50": round(statistics.median(gaps) * 1000, 2) if gaps else None,
            "inter_token_ms_max": round(max(gaps) * 1000, 2) if gaps else None,
            "chars": len(acc)}


# --------------------------------------------------------------------------- #
# PART A — contention matrix
# --------------------------------------------------------------------------- #

async def run_combo(comp: Components, client: httpx.AsyncClient, *,
                    use_stt: bool, use_tts: bool, use_llm: bool,
                    rounds: int = 4) -> dict:
    """Run the selected components concurrently for `rounds` iterations each."""
    tight, timer = Probe("t", TIGHT_TICK), Probe("m", TIMER_TICK)
    tt, tm = asyncio.create_task(tight.run()), asyncio.create_task(timer.run())
    await asyncio.sleep(0.15)

    sampler = ResourceSampler()
    sampler.start()

    loop = asyncio.get_running_loop()
    # separate pools: STT and TTS must not queue behind each other
    stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
    tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
    lat: dict[str, list[float]] = {"stt": [], "tts": [], "llm": []}

    async def stt_loop():
        for _ in range(rounds):
            t0 = time.perf_counter()
            await loop.run_in_executor(stt_pool, comp.stt_once)
            lat["stt"].append(time.perf_counter() - t0)

    async def tts_loop():
        for _ in range(rounds):
            t0 = time.perf_counter()
            await loop.run_in_executor(tts_pool, comp.tts_once)
            lat["tts"].append(time.perf_counter() - t0)

    async def llm_loop():
        for _ in range(rounds):
            t0 = time.perf_counter()
            await llm_stream(client)
            lat["llm"].append(time.perf_counter() - t0)

    tasks = []
    if use_stt:
        tasks.append(stt_loop())
    if use_tts:
        tasks.append(tts_loop())
    if use_llm:
        tasks.append(llm_loop())

    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0

    stt_pool.shutdown(wait=True)
    tts_pool.shutdown(wait=True)
    res = sampler.stop()
    tight.stop(); timer.stop()
    await asyncio.gather(tt, tm)

    out = {"wall_s": round(wall, 3), "resources": res,
           "tight_max_ms": tight.stats().get("gap_ms_max"),
           "tight_p99_ms": tight.stats().get("gap_ms_p99"),
           "timer_max_ms": timer.stats().get("gap_ms_max")}
    for k, v in lat.items():
        if v:
            out[f"{k}_s_mean"] = round(statistics.fmean(v), 3)
            out[f"{k}_s_max"] = round(max(v), 3)
    return out


async def part_a(comp: Components, client: httpx.AsyncClient) -> dict:
    combos = [
        ("stt solo", True, False, False),
        ("tts solo", False, True, False),
        ("llm solo", False, False, True),
        ("stt+tts", True, True, False),
        ("stt+llm", True, False, True),
        ("tts+llm", False, True, True),
        ("stt+tts+llm (ALL)", True, True, True),
    ]
    results = {}
    for label, s, t, l in combos:
        r = await run_combo(comp, client, use_stt=s, use_tts=t, use_llm=l)
        results[label] = r
        parts = [f"{k.split('_')[0]}={r[k]}s" for k in
                 ("stt_s_mean", "tts_s_mean", "llm_s_mean") if k in r]
        print(f"  {label:20s} wall={r['wall_s']:6.2f}s  {'  '.join(parts):48s}"
              f"  loop_max={r['tight_max_ms']}ms  vram_max={r['resources'].get('vram_mb_max')}MB",
              flush=True)
        await asyncio.sleep(0.5)  # let things settle between combos

    # slowdown vs solo
    solo = {"stt": results["stt solo"].get("stt_s_mean"),
            "tts": results["tts solo"].get("tts_s_mean"),
            "llm": results["llm solo"].get("llm_s_mean")}
    for label, r in results.items():
        r["slowdown_vs_solo"] = {
            k: round(r[f"{k}_s_mean"] / solo[k], 2)
            for k in ("stt", "tts", "llm")
            if f"{k}_s_mean" in r and solo.get(k)
        }
    return results


# --------------------------------------------------------------------------- #
# PART B — one realistic turn
# --------------------------------------------------------------------------- #

async def part_b(comp: Components, client: httpx.AsyncClient) -> dict:
    """Simulated mic at wall-clock pace -> partial STT -> prefill -> answer -> TTS.

    The point is the timeline, especially: how long after the user goes silent does the
    first speakable audio exist?
    """
    tight, timer = Probe("t", TIGHT_TICK), Probe("m", TIMER_TICK)
    tt, tm = asyncio.create_task(tight.run()), asyncio.create_task(timer.run())
    sampler = ResourceSampler(); sampler.start()

    loop = asyncio.get_running_loop()
    stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
    tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    audio = comp.audio
    sr = 16000
    FRAME = int(0.02 * sr)          # 20 ms frames, as a real mic would deliver
    PARTIAL_EVERY = 0.5             # cap partial STT rate (GPU is shared)
    SILENCE_HOLD = 0.5              # end-of-speech trigger from the plan

    # --- shared state, and the rule for each field ---
    # `ring` is appended by the mic task (event loop) and READ by the stt thread.
    # Python list append/slice are atomic under the GIL, and the stt thread only ever
    # reads a snapshot taken on the loop, so no lock is needed. Anything that needs a
    # read-modify-write across threads WOULD need one.
    ring: list[np.ndarray] = []
    ring_lock = threading.Lock()
    timeline: list[dict] = []
    t_start = time.perf_counter()

    def mark(event: str, **kw):
        timeline.append({"t": round(time.perf_counter() - t_start, 3), "event": event, **kw})

    partials: list[str] = []
    speech_done = asyncio.Event()

    async def mic_task():
        """Feed the wav in real time, then hold silence to trigger end-of-speech."""
        mark("mic_start")
        for i in range(0, len(audio), FRAME):
            with ring_lock:
                ring.append(audio[i:i + FRAME])
            await asyncio.sleep(0.02)
        mark("speech_end_real")
        await asyncio.sleep(SILENCE_HOLD)
        mark("silence_detected")
        speech_done.set()

    async def partial_stt_task():
        """Re-transcribe the accumulated buffer at a capped rate while speech continues."""
        while not speech_done.is_set():
            await asyncio.sleep(PARTIAL_EVERY)
            with ring_lock:
                if not ring:
                    continue
                snapshot = np.concatenate(ring)      # snapshot taken ON THE LOOP
            t0 = time.perf_counter()
            text = await loop.run_in_executor(       # thread only sees the snapshot
                stt_pool, lambda a=snapshot: " ".join(
                    s.text.strip() for s in
                    comp.whisper.transcribe(a, language="ru", beam_size=5,
                                            vad_filter=True)[0]))
            partials.append(text)
            mark("partial_stt", ms=round((time.perf_counter() - t0) * 1000),
                 audio_s=round(len(snapshot) / sr, 2), text=text[:60])
            # keep the prompt warm — cheap (~60 ms) thanks to cache_prompt
            if text:
                t1 = time.perf_counter()
                await client.post(f"{BASE}/completion", json={
                    "prompt": f"{SYSTEM}\n\nКонтекст:\n{CONTEXT}\n\n"
                              f"Реплика абитуриента (в процессе): {text}",
                    "n_predict": 1, "cache_prompt": True, "temperature": 0.0},
                    timeout=60.0)
                mark("prefill_update", ms=round((time.perf_counter() - t1) * 1000))

    mic = asyncio.create_task(mic_task())
    stt = asyncio.create_task(partial_stt_task())
    await speech_done.wait()
    await asyncio.gather(mic, stt)

    t_silence = time.perf_counter()

    # final STT on the full buffer
    with ring_lock:
        full = np.concatenate(ring)
    t0 = time.perf_counter()
    final_text = await loop.run_in_executor(
        stt_pool, lambda a=full: " ".join(
            s.text.strip() for s in
            comp.whisper.transcribe(a, language="ru", beam_size=5, vad_filter=True)[0]))
    mark("final_stt", ms=round((time.perf_counter() - t0) * 1000), text=final_text[:80])

    # answer stream; synthesise each sentence as soon as it is complete
    first_audio_at = None
    sent_buf = ""
    n_sent = 0
    t0 = time.perf_counter()
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Контекст:\n{CONTEXT}\n\n"
                                                     f"Вопрос: {final_text}"},
                         NOTHINK],
            "temperature": 0.0, "max_tokens": 150, "stream": True}
    async with client.stream("POST", f"{BASE}/v1/chat/completions", json=body,
                             timeout=240.0) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            p = line[6:].strip()
            if p == "[DONE]":
                break
            d = (json.loads(p).get("choices") or [{}])[0].get("delta") or {}
            tok = d.get("content") or ""
            if not tok:
                continue
            if first_audio_at is None and n_sent == 0:
                mark("llm_first_token", ms=round((time.perf_counter() - t0) * 1000))
            sent_buf += tok
            if any(sent_buf.rstrip().endswith(c) for c in ".!?"):
                sentence = sent_buf.strip()
                sent_buf = ""
                ts = time.perf_counter()
                dur = await loop.run_in_executor(
                    tts_pool, lambda s=sentence: _tts_text(comp, s))
                n_sent += 1
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                    mark("FIRST_AUDIO_READY",
                         since_silence_ms=round((first_audio_at - t_silence) * 1000))
                mark("tts_sentence", n=n_sent, ms=round((time.perf_counter() - ts) * 1000),
                     audio_s=round(dur, 2), text=sentence[:60])
    mark("answer_complete", ms=round((time.perf_counter() - t0) * 1000))

    stt_pool.shutdown(wait=True); tts_pool.shutdown(wait=True)
    res = sampler.stop()
    tight.stop(); timer.stop()
    await asyncio.gather(tt, tm)

    return {
        "timeline": timeline,
        "n_partials": len(partials),
        "final_transcript": final_text,
        "latency_silence_to_first_audio_ms":
            round((first_audio_at - t_silence) * 1000) if first_audio_at else None,
        "tight_max_ms": tight.stats().get("gap_ms_max"),
        "timer_max_ms": timer.stats().get("gap_ms_max"),
        "resources": res,
    }


def _tts_text(comp: Components, text: str) -> float:
    with comp.torch.no_grad():
        wav = comp.silero.apply_tts(text=text, speaker="baya", sample_rate=48000,
                                    put_accent=True, put_yo=True)
    return len(wav) / 48000


# --------------------------------------------------------------------------- #

async def main() -> None:
    out: dict = {}
    print(f"VRAM before loading: {vram_used_mb()} MB")
    comp = Components()
    out["load"] = comp.load()
    print(f"loaded: {out['load']}   VRAM now: {vram_used_mb()} MB   RSS: {rss_mb():.0f} MB\n")
    out["vram_after_load_mb"] = vram_used_mb()

    async with httpx.AsyncClient() as client:
        try:
            h = await client.get(f"{BASE}/health", timeout=5)
            print(f"llama-server: {h.text.strip()}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"llama-server NOT reachable at {BASE}: {exc}")
            return

        print("=== PART A: contention matrix ===")
        out["part_a"] = await part_a(comp, client)

        print("\n=== PART B: one realistic turn ===")
        out["part_b"] = await part_b(comp, client)
        for e in out["part_b"]["timeline"]:
            extra = {k: v for k, v in e.items() if k not in ("t", "event")}
            print(f"  {e['t']:7.3f}s  {e['event']:22s} {extra}")
        print(f"\n  silence -> first audio: "
              f"{out['part_b']['latency_silence_to_first_audio_ms']} ms")
        print(f"  loop max gap: {out['part_b']['tight_max_ms']} ms")
        print(f"  resources: {out['part_b']['resources']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
