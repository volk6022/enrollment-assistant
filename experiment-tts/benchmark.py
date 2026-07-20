"""Main TTS benchmark runner.

Usage:
    python benchmark.py --models all --device cpu --save-audio --output results/run.json

Flags:
    --models    Comma-separated list of model keys, or "all".
                Available: silero-xenia, silero-aidar, silero-baya, silero-kseniya,
                           piper-irinia, piper-ruslan, kokoro
    --device    "cpu" or "cuda" (default: cpu)
    --output    Path for the JSON results file (default: results/benchmark_{run_id}.json)
    --save-audio  If set, save a WAV file per sentence to audio_samples/
    --warmup    Number of warm-up sentences before timing (default: 1)
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tabulate import tabulate  # type: ignore[import]

from audio_utils import save_wav_from_bytes
from metrics import ram_mb, vram_peak_mb, vram_peak_reset, vram_snapshot
from models.base import TTSModel, TTSResult
from test_sentences import STREAMING_SEQUENCE, TEST_SENTENCES

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_model_registry(device: str) -> dict[str, "TTSModel"]:
    """Lazily import and instantiate all models. Missing packages are reported, not crashed."""
    registry: dict[str, TTSModel] = {}

    # Silero voices
    try:
        from models.silero import SileroTTSModel
        for speaker in ("xenia", "aidar", "baya", "kseniya"):
            key = f"silero-{speaker}"
            registry[key] = SileroTTSModel(speaker=speaker, sample_rate=24000, device=device)
    except ImportError as exc:
        print(f"[WARN] Silero models unavailable: {exc}")

    # Piper voices
    try:
        from models.piper import PiperTTSModel
        registry["piper-irina"] = PiperTTSModel(voice_name="ru_RU-irina-medium")
        registry["piper-ruslan"] = PiperTTSModel(voice_name="ru_RU-ruslan-medium")
    except ImportError as exc:
        print(f"[WARN] Piper models unavailable: {exc}")

    # Kokoro (English reference)
    try:
        from models.kokoro import KokoroTTSModel
        registry["kokoro"] = KokoroTTSModel(voice="af_heart", device=device)
    except ImportError as exc:
        print(f"[WARN] Kokoro model unavailable: {exc}")

    return registry


# ---------------------------------------------------------------------------
# Per-model benchmark logic
# ---------------------------------------------------------------------------

def _run_model(
    model: TTSModel,
    sentences: list[str],
    streaming_sequence: list[str],
    device: str,
    save_audio: bool,
    audio_dir: Path,
    warmup_count: int,
) -> dict:
    """Load, warm up, benchmark, and unload *model*. Return a result dict."""
    result: dict = {
        "model": model.name,
        "device": device,
        "load_time_s": None,
        "vram_after_load_mb": None,
        "ram_after_load_mb": None,
        "sentences": [],
        "streaming_test": None,
        "avg_rtf": None,
        "avg_ttfa_s": None,
        "error": None,
    }

    # -- Load --
    print(f"\n[{model.name}] Loading...", flush=True)
    vram_peak_reset()
    vram_before = vram_snapshot()
    t_load = time.perf_counter()
    try:
        model.load()
    except (ImportError, FileNotFoundError) as exc:
        result["error"] = str(exc)
        print(f"[{model.name}] SKIP — {exc}")
        return result
    load_time = time.perf_counter() - t_load

    result["load_time_s"] = round(load_time, 3)
    result["vram_after_load_mb"] = round(vram_peak_mb(), 1)
    result["ram_after_load_mb"] = round(ram_mb(), 1)
    print(f"[{model.name}] Loaded in {load_time:.2f}s  "
          f"VRAM={result['vram_after_load_mb']:.0f} MB  "
          f"RAM={result['ram_after_load_mb']:.0f} MB")

    # -- Warm-up --
    print(f"[{model.name}] Warming up ({warmup_count} sentence(s))...", flush=True)
    for i in range(min(warmup_count, len(sentences))):
        try:
            model.synthesize(sentences[i])
        except Exception as exc:
            print(f"[{model.name}] Warm-up error: {exc}")
            break

    # -- Sentence benchmarks --
    print(f"[{model.name}] Benchmarking {len(sentences)} sentences...")
    sentence_results: list[dict] = []
    for idx, text in enumerate(sentences):
        try:
            tts_result: TTSResult = model.synthesize(text)
        except Exception as exc:
            print(f"[{model.name}]   [{idx:02d}] ERROR: {exc}")
            sentence_results.append({"text": text, "error": str(exc)})
            continue

        audio_file = None
        if save_audio:
            wav_path = audio_dir / f"{model.name}_{idx:02d}.wav"
            save_wav_from_bytes(tts_result.audio_data, wav_path, tts_result.sample_rate)
            audio_file = str(wav_path)

        row = {
            "text": text[:80] + ("…" if len(text) > 80 else ""),
            "audio_duration_s": round(tts_result.audio_duration_s, 3),
            "synthesis_time_s": round(tts_result.synthesis_time_s, 3),
            "ttfa_s": round(tts_result.ttfa_s, 3),
            "rtf": round(tts_result.rtf, 4),
            "vram_peak_mb": round(tts_result.vram_peak_mb, 1),
            "ram_mb": round(tts_result.ram_mb, 1),
            "sample_rate": tts_result.sample_rate,
            "audio_file": audio_file,
        }
        sentence_results.append(row)
        print(
            f"[{model.name}]   [{idx:02d}] rtf={tts_result.rtf:.3f}  "
            f"ttfa={tts_result.ttfa_s * 1000:.0f} ms  "
            f"dur={tts_result.audio_duration_s:.2f}s"
        )
    result["sentences"] = sentence_results

    # -- Streaming test (LLM∥TTS simulation) --
    print(f"[{model.name}] Streaming test ({len(streaming_sequence)} sentences)...")
    streaming_result = _run_streaming_test(model, streaming_sequence, save_audio, audio_dir)
    result["streaming_test"] = streaming_result

    # -- Aggregate stats --
    valid = [s for s in sentence_results if "error" not in s]
    if valid:
        result["avg_rtf"] = round(sum(s["rtf"] for s in valid) / len(valid), 4)
        result["avg_ttfa_s"] = round(sum(s["ttfa_s"] for s in valid) / len(valid), 4)

    # -- Unload --
    model.unload()
    print(f"[{model.name}] Unloaded.")
    return result


def _run_streaming_test(
    model: TTSModel,
    sequence: list[str],
    save_audio: bool,
    audio_dir: Path,
) -> dict:
    """Simulate LLM∥TTS by feeding sentences one at a time and measuring TTFA per sentence."""
    ttfa_list: list[float] = []
    total_audio_duration = 0.0
    t_stream_start = time.perf_counter()

    for idx, text in enumerate(sequence):
        if model.supports_streaming:
            # True streaming: time to first yielded chunk.
            gen = model.synthesize_streaming(text)
            t0 = time.perf_counter()
            try:
                first_chunk = next(gen)
                ttfa = time.perf_counter() - t0
                remaining = list(gen)
                all_bytes = first_chunk + b"".join(remaining)
            except StopIteration:
                ttfa = time.perf_counter() - t0
                all_bytes = b""
        else:
            # Fake streaming: synthesize() is the atomic unit; TTFA = full synthesis time.
            t0 = time.perf_counter()
            try:
                tts_result = model.synthesize(text)
                ttfa = tts_result.synthesis_time_s
                all_bytes = tts_result.audio_data
            except Exception as exc:
                print(f"      Streaming sentence [{idx}] error: {exc}")
                continue

        import numpy as np
        audio_np = np.frombuffer(all_bytes, dtype=np.int16)
        # Use stored sample_rate from the model; fall back to a reasonable default.
        sr = getattr(model, "sample_rate", 24000)
        dur = len(audio_np) / sr if len(audio_np) > 0 else 0.0

        ttfa_list.append(ttfa)
        total_audio_duration += dur

        if save_audio and all_bytes:
            wav_path = audio_dir / f"{model.name}_stream_{idx:02d}.wav"
            save_wav_from_bytes(all_bytes, wav_path, sr)

        print(
            f"      stream[{idx}]: ttfa={ttfa * 1000:.0f} ms  dur={dur:.2f}s  "
            f"rtf={ttfa / dur:.3f}" if dur > 0 else f"      stream[{idx}]: ttfa={ttfa * 1000:.0f} ms  dur=0"
        )

    total_time = time.perf_counter() - t_stream_start
    overall_rtf = total_time / total_audio_duration if total_audio_duration > 0 else 0.0

    return {
        "sentences": len(sequence),
        "total_time_s": round(total_time, 3),
        "avg_ttfa_s": round(sum(ttfa_list) / len(ttfa_list), 4) if ttfa_list else None,
        "total_audio_duration_s": round(total_audio_duration, 3),
        "overall_rtf": round(overall_rtf, 4),
    }


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_summary(all_results: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    headers = ["Model", "Load(s)", "VRAM(MB)", "RAM(MB)", "avg RTF", "avg TTFA(ms)", "Stream TTFA(ms)", "Status"]
    rows = []
    for r in all_results:
        if r.get("error"):
            rows.append([r["model"], "-", "-", "-", "-", "-", "-", f"ERROR: {r['error'][:40]}"])
            continue
        st = r.get("streaming_test") or {}
        rows.append([
            r["model"],
            f"{r['load_time_s']:.2f}" if r["load_time_s"] is not None else "-",
            f"{r['vram_after_load_mb']:.0f}" if r["vram_after_load_mb"] is not None else "-",
            f"{r['ram_after_load_mb']:.0f}" if r["ram_after_load_mb"] is not None else "-",
            f"{r['avg_rtf']:.4f}" if r["avg_rtf"] is not None else "-",
            f"{r['avg_ttfa_s'] * 1000:.1f}" if r["avg_ttfa_s"] is not None else "-",
            f"{st['avg_ttfa_s'] * 1000:.1f}" if st.get("avg_ttfa_s") is not None else "-",
            "OK",
        ])

    print("\n" + "=" * 80)
    print("TTS BENCHMARK SUMMARY")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Russian TTS models for latency, VRAM, and RTF."
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            'Comma-separated model keys or "all". '
            "Available: silero-xenia, silero-aidar, silero-baya, silero-kseniya, "
            "piper-irinia, piper-ruslan, kokoro"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device for models that support GPU (default: cpu).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the JSON results file (default: results/benchmark_<run_id>.json).",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save WAV files to audio_samples/ for manual quality evaluation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up synthesis calls before timing (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full benchmark suite."""
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")

    # Resolve output path.
    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / "results" / f"benchmark_{run_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(__file__).parent / "audio_samples"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Build registry and select models.
    registry = _build_model_registry(args.device)
    if args.models.strip().lower() == "all":
        selected_keys = list(registry.keys())
    else:
        selected_keys = [k.strip() for k in args.models.split(",")]

    unknown = [k for k in selected_keys if k not in registry]
    if unknown:
        print(f"[WARN] Unknown model keys (will be skipped): {unknown}")
        selected_keys = [k for k in selected_keys if k in registry]

    print(f"\nTTS Benchmark — run_id={run_id}  device={args.device}")
    print(f"Models: {selected_keys}")
    print(f"Sentences: {len(TEST_SENTENCES)}  |  Streaming sequence: {len(STREAMING_SEQUENCE)}")
    print(f"Save audio: {args.save_audio}  |  Output: {output_path}\n")

    all_results: list[dict] = []
    for key in selected_keys:
        model = registry[key]
        model_result = _run_model(
            model=model,
            sentences=TEST_SENTENCES,
            streaming_sequence=STREAMING_SEQUENCE,
            device=args.device,
            save_audio=args.save_audio,
            audio_dir=audio_dir,
            warmup_count=args.warmup,
        )
        all_results.append(model_result)

    # Print summary table.
    _print_summary(all_results)

    # Persist JSON.
    output: dict = {
        "run_id": run_id,
        "device": args.device,
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
