"""Is faster-whisper on CPU salvageable for streaming, or must it go on the GPU?

Baseline measured by gil_probe.py on this box (RTX 3060 Ti 8 GB / 12 logical cores):
    large-v3-turbo int8 CUDA -> 0.31 s for 2.65 s of audio (RTF 0.12)
    large-v3-turbo int8 CPU  -> 23.0 s for 2.65 s of audio (RTF 8.7)   <- 74x slower

74x is far outside what "CPU is a bit slower" would explain, so sweep the knobs that
plausibly cause it: intra-op thread count, quantisation kernel, beam width, model size.
Target for streaming: RTF < ~0.3 on a chunk, i.e. transcription keeps up with speech.

Run:  .venv\\Scripts\\python.exe experiment-streaming\\whisper_cpu_sweep.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from gil_probe import SAMPLE_WAV, load_wav_f32  # same audio as the GIL probe

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "runs" / "whisper_cpu_sweep.json"

# (label, model, compute_type, cpu_threads, beam_size)
GRID = [
    ("turbo/int8/th=0(default)/beam5", "large-v3-turbo", "int8", 0, 5),
    ("turbo/int8/th=6/beam5", "large-v3-turbo", "int8", 6, 5),
    ("turbo/int8/th=12/beam5", "large-v3-turbo", "int8", 12, 5),
    ("turbo/int8/th=12/beam1", "large-v3-turbo", "int8", 12, 1),
    ("turbo/float32/th=12/beam1", "large-v3-turbo", "float32", 12, 1),
    ("turbo/int8_float32/th=12/beam1", "large-v3-turbo", "int8_float32", 12, 1),
    ("small/int8/th=12/beam1", "small", "int8", 12, 1),
    ("base/int8/th=12/beam1", "base", "int8", 12, 1),
]

REPS = 2


def main() -> None:
    from faster_whisper import WhisperModel

    audio, _sr = load_wav_f32(SAMPLE_WAV)
    dur = len(audio) / 16000
    print(f"audio: {dur:.2f}s from {SAMPLE_WAV.name}\n")

    results = []
    for label, model_size, compute, threads, beam in GRID:
        try:
            kw = {"device": "cpu", "compute_type": compute}
            if threads:
                kw["cpu_threads"] = threads
            t_load = time.perf_counter()
            model = WhisperModel(model_size, **kw)
            load_s = time.perf_counter() - t_load

            def run():
                segs, _ = model.transcribe(audio, language="ru", beam_size=beam,
                                           vad_filter=True)
                return " ".join(s.text.strip() for s in segs)

            text = run()  # warm
            times = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                run()
                times.append(time.perf_counter() - t0)
            mean = sum(times) / len(times)
            row = {
                "label": label, "model": model_size, "compute_type": compute,
                "cpu_threads": threads or "default", "beam_size": beam,
                "load_s": round(load_s, 2),
                "infer_s": round(mean, 3), "rtf": round(mean / dur, 3),
                "transcript": text,
            }
            print(f"{label:34s} {mean:7.2f}s  RTF {mean/dur:6.2f}  load {load_s:5.1f}s")
            print(f"{'':34s} -> {text}")
        except Exception as exc:  # noqa: BLE001
            row = {"label": label, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{label:34s} FAILED: {type(exc).__name__}: {exc}")
        results.append(row)
        del model

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"audio_sec": dur, "reps": REPS, "grid": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
