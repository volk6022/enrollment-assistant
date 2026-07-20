"""Main benchmark runner for STT model comparison.

Usage:
    python benchmark.py --models all --device cuda --data-dir ./test_data
    python benchmark.py --models faster-whisper-small,vosk-ru-small --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tabulate import tabulate

from metrics import wer, cer, vram_snapshot, ram_mb
from models.base import STTModel, STTResult


# ---------------------------------------------------------------------------
# Model registry — maps CLI name to (class, kwargs)
# ---------------------------------------------------------------------------

def _build_model_registry(device: str) -> dict[str, STTModel]:
    """Construct all available model instances for the given device."""
    from models.faster_whisper import FasterWhisperModel
    from models.vosk_model import VoskModel

    _cuda = device == "cuda"

    registry: dict[str, STTModel] = {
        # faster-whisper variants
        "faster-whisper-tiny": FasterWhisperModel(
            model_size="tiny", device=device, compute_type="int8"
        ),
        "faster-whisper-base": FasterWhisperModel(
            model_size="base", device=device, compute_type="int8"
        ),
        "faster-whisper-small": FasterWhisperModel(
            model_size="small", device=device, compute_type="int8"
        ),
        "faster-whisper-large-v3-turbo": FasterWhisperModel(
            model_size="large-v3-turbo", device=device, compute_type="int8"
        ),
        # Vosk (CPU-only models)
        "vosk-ru-small": VoskModel(model_size="small"),
        "vosk-ru-large": VoskModel(model_size="large"),  # ~1.5 GB — marked large
        # NOTE: qwen3-asr dropped — HF checkpoint uses the legacy `thinker.*`
        #   namespace, transformers 5.13 expects `model.language_model.*`; all
        #   weights load as UNEXPECTED/MISSING → random init → garbage output.
        #   faster-whisper-large-v3-turbo already wins (WER ~0.15). See qwen3_asr.py.
        # NOTE: silero-stt dropped — Silero's open STT models cover only
        #   en/de/es/ua; Russian is enterprise-only (not in models.yml).
    }
    return registry


# ---------------------------------------------------------------------------
# Test data loading
# ---------------------------------------------------------------------------

def _collect_test_pairs(data_dir: Path) -> list[tuple[Path, str]]:
    """Return [(wav_path, reference_text), ...] pairs from data_dir.

    Pairs a .wav file with a same-name .txt file. Files with no matching
    transcript are skipped with a warning.
    """
    pairs: list[tuple[Path, str]] = []
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        print(f"[warn] No .wav files found in {data_dir}", file=sys.stderr)
        return pairs

    for wav in wav_files:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"[warn] No transcript for {wav.name} — skipping", file=sys.stderr)
            continue
        reference = txt.read_text(encoding="utf-8").strip()
        pairs.append((wav, reference))

    return pairs


# ---------------------------------------------------------------------------
# Per-model benchmark loop
# ---------------------------------------------------------------------------

def _run_model(
    model: STTModel,
    pairs: list[tuple[Path, str]],
    device: str,
) -> dict[str, Any]:
    """Load model, run all audio samples, unload, return structured results."""
    print(f"\n--- {model.name} ---")

    vram_before_load = vram_snapshot()
    ram_before_load = ram_mb()

    t_load_start = time.perf_counter()
    try:
        model.load()
    except (ImportError, FileNotFoundError) as exc:
        print(f"[skip] {model.name}: {exc}", file=sys.stderr)
        return {
            "model": model.name,
            "device": device,
            "error": str(exc),
            "samples": [],
        }
    load_time_s = time.perf_counter() - t_load_start

    vram_after_load = vram_snapshot()
    ram_after_load = ram_mb()

    print(
        f"  Loaded in {load_time_s:.2f}s | "
        f"VRAM: {vram_after_load:.1f} MB (+{vram_after_load - vram_before_load:.1f}) | "
        f"RAM: {ram_after_load:.1f} MB (+{ram_after_load - ram_before_load:.1f})"
    )

    sample_results: list[dict[str, Any]] = []
    wer_values: list[float] = []
    cer_values: list[float] = []
    rtf_values: list[float] = []

    for wav_path, reference in pairs:
        print(f"  transcribing {wav_path.name} ...", end=" ", flush=True)
        try:
            result: STTResult = model.transcribe(wav_path)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            sample_results.append({
                "file": wav_path.name,
                "error": str(exc),
            })
            continue

        sample_wer = wer(reference, result.text)
        sample_cer = cer(reference, result.text)
        wer_values.append(sample_wer)
        cer_values.append(sample_cer)
        rtf_values.append(result.rtf)

        print(
            f"RTF={result.rtf:.3f} WER={sample_wer:.3f} CER={sample_cer:.3f} "
            f"({result.duration_s:.2f}s / {result.audio_duration_s:.2f}s audio)"
        )

        sample_results.append({
            "file": wav_path.name,
            "audio_duration_s": round(result.audio_duration_s, 3),
            "inference_time_s": round(result.duration_s, 3),
            "rtf": round(result.rtf, 4),
            "vram_peak_inference_mb": round(result.vram_peak_mb, 2),
            "vram_delta_mb": round(result.vram_delta_mb, 2),
            "wer": round(sample_wer, 4),
            "cer": round(sample_cer, 4),
            "hypothesis": result.text,
            "reference": reference,
        })

    model.unload()

    avg_wer = sum(wer_values) / len(wer_values) if wer_values else None
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else None
    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else None

    return {
        "model": model.name,
        "device": device,
        "load_time_s": round(load_time_s, 3),
        "vram_after_load_mb": round(vram_after_load, 2),
        "ram_after_load_mb": round(ram_after_load, 2),
        "samples": sample_results,
        "avg_rtf": round(avg_rtf, 4) if avg_rtf is not None else None,
        "avg_wer": round(avg_wer, 4) if avg_wer is not None else None,
        "avg_cer": round(avg_cer, 4) if avg_cer is not None else None,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_results: list[dict[str, Any]]) -> None:
    """Print a formatted summary table of all benchmark results."""
    headers = ["Model", "Device", "Load(s)", "VRAM(MB)", "Avg RTF", "Avg WER", "Avg CER", "Status"]
    rows = []
    for r in all_results:
        if "error" in r:
            rows.append([r["model"], r["device"], "-", "-", "-", "-", "-", f"ERROR: {r['error'][:40]}"])
        else:
            rows.append([
                r["model"],
                r["device"],
                r.get("load_time_s", "-"),
                r.get("vram_after_load_mb", "-"),
                r.get("avg_rtf", "-"),
                r.get("avg_wer", "-"),
                r.get("avg_cer", "-"),
                "OK" if r["samples"] else "no data",
            ])

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=".4f"))


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="STT model benchmark harness for Russian speech"
    )
    parser.add_argument(
        "--models",
        default="all",
        help='Comma-separated model names, or "all". '
             'Example: faster-whisper-small,vosk-ru-small',
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("test_data"),
        help="Directory containing WAV files and matching .txt transcripts",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Target device for GPU models (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path. Default: results/results_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full benchmark suite."""
    import torch

    args = parse_args()

    # Resolve output path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    output_path: Path = args.output or Path("results") / f"results_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect GPU name
    gpu_name: str | None = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print("CUDA not available — running on CPU only")
        if args.device == "cuda":
            print("[warn] --device cuda requested but CUDA unavailable; falling back to cpu")
            args.device = "cpu"

    # Build registry and filter by --models
    registry = _build_model_registry(args.device)

    if args.models.strip().lower() == "all":
        selected_names = list(registry.keys())
    else:
        selected_names = [n.strip() for n in args.models.split(",")]
        unknown = [n for n in selected_names if n not in registry]
        if unknown:
            print(f"[error] Unknown model(s): {unknown}", file=sys.stderr)
            print(f"Available: {list(registry.keys())}", file=sys.stderr)
            sys.exit(1)

    selected_models = [registry[n] for n in selected_names]

    # Collect test audio pairs
    pairs = _collect_test_pairs(args.data_dir)
    if not pairs:
        print(
            f"[warn] No audio/transcript pairs found in {args.data_dir}. "
            "Benchmark will run without WER scoring.",
            file=sys.stderr,
        )

    print(f"\nBenchmarking {len(selected_models)} model(s) on {len(pairs)} audio file(s).")

    # Run benchmark
    all_results: list[dict[str, Any]] = []
    for model in selected_models:
        result = _run_model(model, pairs, args.device)
        all_results.append(result)

    # Assemble output
    output_data: dict[str, Any] = {
        "run_id": run_id,
        "device": args.device,
        "gpu_name": gpu_name,
        "results": all_results,
    }

    # Save JSON
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    _print_summary(all_results)


if __name__ == "__main__":
    main()
