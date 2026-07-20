"""Full TTS enhancement grid — every voice × accent × denoise × stitch combo.

Produces a ready-to-audition experiment grid so you can pick the right pipeline
by *listening*, not guessing. For each cell it records per-stage latency (accent /
synthesis / denoise), RTF, TTFA, and clipping diagnostics, and writes audio to
`grid_audio/<cell>/` plus a JSON + Markdown summary under `results/`.

Axes
----
  voice   : silero-{xenia,aidar,baya,kseniya}, piper-{irina,ruslan}
  accent  : raw | ruaccent           (RUAccent `+` stress preprocessing)
  denoise : none | dfn               (DeepFilterNet post-filter)
  stitch  : concat | crossfade       (streaming seam handling; audio-only)

Each voice is loaded ONCE (amortises Silero's ~10 s first-call JIT); the accent ×
denoise × stitch sweep runs against the loaded model. Missing optional packages
(ruaccent / deepfilternet) don't crash — those cells are marked "skipped".

Usage
-----
    python experiment_grid.py                       # everything
    python experiment_grid.py --smoke               # 1 silero + 1 piper, quick
    python experiment_grid.py --voices silero-xenia,piper-ruslan
    python experiment_grid.py --accent raw,ruaccent --denoise none,dfn
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tabulate import tabulate

from audio_utils import save_wav
from enhance import (
    AccentProcessor,
    Denoiser,
    clipping_stats,
    concat,
    crossfade,
    deesser,
    high_shelf,
    lowpass,
    pcm_bytes_to_f32,
    peak_normalize,
    soft_limit,
    soxr_smooth,
)
from test_sentences import STREAMING_SEQUENCE, TEST_SENTENCES

# Representative sentences to audition per cell: short / medium / long.
REPRESENTATIVE_IDX = [0, 6, 12]

# Parametric post mode: soxr_deess_s2_18k / soxr_deess_s3_22k, etc.
# level 2 = de-ess (5.5k/0.06/x5); level 3 = deeper (5k/0.05/x6); via = soxr cutoff.
_SOXR_DEESS_RE = re.compile(r"soxr_deess_s([23])_(\d+)k")


# ---------------------------------------------------------------------------
# Voice registry
# ---------------------------------------------------------------------------

def build_voice_registry(
    device: str,
    engines: list[str],
    silero_version: str = "v5_5_ru",
    silero_sr: int = 24000,
) -> dict[str, object]:
    """voice_key → TTSModel instance (not yet loaded).

    Silero voices use *silero_version* / *silero_sr*; Piper is sample-rate-fixed.
    """
    reg: dict[str, object] = {}
    if "silero" in engines:
        try:
            from models.silero import SileroTTSModel
            for spk in ("xenia", "aidar", "baya", "kseniya"):
                m = SileroTTSModel(
                    speaker=spk, sample_rate=silero_sr, device=device, version=silero_version
                )
                reg[m.name] = m  # e.g. silero-v55-aidar-48k
        except ImportError as exc:
            print(f"[WARN] Silero unavailable: {exc}")
    if "piper" in engines:
        try:
            from models.piper import PiperTTSModel
            reg["piper-irina"] = PiperTTSModel(voice_name="ru_RU-irina-medium")
            reg["piper-ruslan"] = PiperTTSModel(voice_name="ru_RU-ruslan-medium")
        except ImportError as exc:
            print(f"[WARN] Piper unavailable: {exc}")
    return reg


# ---------------------------------------------------------------------------
# One synthesis, with timed accent-pre and denoise-post stages
# ---------------------------------------------------------------------------

def synth_f32(
    model,
    text: str,
    accentor: AccentProcessor | None,
    post_mode: str,
    denoiser: Denoiser | None,
) -> dict:
    """Synthesize *text*; return float32 audio + per-stage timings & diagnostics.

    post_mode: "none" | "norm" (peak-normalize -1 dBFS) | "lp10"/"lp8" (Butterworth
    low-pass smoothing at 10/8 kHz) | "soxr" (soxr anti-alias round-trip) |
    "dfn" (DeepFilterNet, Linux only). Smoothing modes also peak-normalize.
    """
    t_acc0 = time.perf_counter()
    used_text = accentor.process(text) if accentor is not None else text
    accent_ms = (time.perf_counter() - t_acc0) * 1000.0

    t_syn0 = time.perf_counter()
    res = model.synthesize(used_text)
    synth_ms = (time.perf_counter() - t_syn0) * 1000.0

    audio = pcm_bytes_to_f32(res.audio_data)
    sr = res.sample_rate

    denoise_ms = 0.0
    t_dn0 = time.perf_counter()
    if post_mode == "norm":
        audio = peak_normalize(audio, -1.0)
    elif post_mode == "lp10":
        audio = peak_normalize(lowpass(audio, sr, 10000.0), -1.0)
    elif post_mode == "lp8":
        audio = peak_normalize(lowpass(audio, sr, 8000.0), -1.0)
    elif post_mode == "soxr":
        audio = peak_normalize(soxr_smooth(audio, sr, via_hz=20000), -1.0)
    elif post_mode == "deess":                       # HF-spike compressor only
        audio = peak_normalize(deesser(audio, sr), -1.0)
    elif post_mode == "shelf":                       # gentle top tilt-down
        audio = peak_normalize(high_shelf(audio, sr, 7000.0, -5.0), -1.0)
    elif post_mode == "deess_shelf":                 # de-ess then shelf (Ivan's HF-stab case)
        audio = peak_normalize(high_shelf(deesser(audio, sr), sr, 7000.0, -4.0), -1.0)
    elif post_mode == "soxr_deess":                  # anti-alias + de-ess (Ivan's pick)
        audio = peak_normalize(deesser(soxr_smooth(audio, sr, 20000), sr), -1.0)
    elif post_mode == "soxr_deess_s1":               # ladder: mild extra smoothing
        a = soxr_smooth(audio, sr, 18000)
        audio = peak_normalize(deesser(a, sr), -1.0)
    elif post_mode == "soxr_deess_s2":               # ladder: medium
        a = soxr_smooth(audio, sr, 16000)
        audio = peak_normalize(deesser(a, sr, 5500.0, 0.06, 5.0), -1.0)
    elif _SOXR_DEESS_RE.fullmatch(post_mode):        # soxr_deess_s{2,3}_{via}k, any via
        _m = _SOXR_DEESS_RE.fullmatch(post_mode)
        _level, _via = _m.group(1), int(_m.group(2)) * 1000
        a = soxr_smooth(audio, sr, _via)
        _params = (5500.0, 0.06, 5.0) if _level == "2" else (5000.0, 0.05, 6.0)
        audio = peak_normalize(deesser(a, sr, *_params), -1.0)
    elif post_mode == "soxr_deess_s3":               # ladder: strong (dullest)
        a = soxr_smooth(audio, sr, 14000)
        audio = peak_normalize(deesser(a, sr, 5000.0, 0.05, 6.0), -1.0)
    elif post_mode == "soxr_deess_shelf":            # anti-alias + de-ess + gentle top tilt
        a = deesser(soxr_smooth(audio, sr, 18000), sr)
        audio = peak_normalize(high_shelf(a, sr, 8000.0, -3.0), -1.0)
    elif post_mode == "limit":                       # loudness "cap"
        audio = soft_limit(audio, gain_db=3.0, ceiling=0.95)
    elif post_mode == "deess_limit":                 # tame HF, then cap loud
        audio = soft_limit(deesser(audio, sr), gain_db=3.0, ceiling=0.95)
    elif post_mode == "dfn" and denoiser is not None:
        audio = denoiser.process(audio, sr)
    denoise_ms = (time.perf_counter() - t_dn0) * 1000.0 if post_mode != "none" else 0.0

    dur = len(audio) / sr if sr else 0.0
    clip = clipping_stats(audio)
    return {
        "audio": audio,
        "sample_rate": sr,
        "text_used": used_text,
        "accent_ms": round(accent_ms, 1),
        "synth_ms": round(synth_ms, 1),
        "denoise_ms": round(denoise_ms, 1),
        # TTFA in a naive pipeline = accent + synth + denoise (all before playback).
        "ttfa_ms": round(accent_ms + synth_ms + denoise_ms, 1),
        "audio_duration_s": round(dur, 3),
        "rtf": round((accent_ms + synth_ms + denoise_ms) / 1000.0 / dur, 4) if dur > 0 else None,
        "clip_peak": round(clip.peak, 4),
        "clipped_pct": round(clip.clipped_pct, 3),
        "dc_offset": round(clip.dc_offset, 5),
    }


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def run_grid(args) -> dict:
    """Execute the full grid and return a results dict."""
    device = args.device
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent
    audio_root = out_dir / "grid_audio"
    audio_root.mkdir(exist_ok=True)

    registry = build_voice_registry(
        device, args.engines, silero_version=args.version, silero_sr=args.sr
    )
    voices = args.voices or list(registry.keys())
    accents = args.accent            # subset of {"raw", "ruaccent"}
    denoises = args.post             # subset of {"none","norm","lp10","lp8","soxr","dfn"}

    # Shared enhancers (loaded lazily, reused across cells).
    accentor = AccentProcessor()
    denoiser = Denoiser()
    ruaccent_ok = accentor.available
    dfn_ok = denoiser.available
    if "ruaccent" in accents and not ruaccent_ok:
        print("[WARN] ruaccent not installed — 'ruaccent' cells will be skipped.")
    if "dfn" in denoises and not dfn_ok:
        print("[WARN] deepfilternet not installed — 'dfn' cells will be skipped.")

    rep_sentences = [TEST_SENTENCES[i] for i in REPRESENTATIVE_IDX]
    cells: list[dict] = []

    for vkey in voices:
        if vkey not in registry:
            print(f"[WARN] unknown voice {vkey!r} — skipping")
            continue
        model = registry[vkey]
        print(f"\n=== loading {vkey} ===", flush=True)
        try:
            model.load()
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {vkey}: {exc}")
            continue
        # Warm up (Silero JIT ~10 s; it recompiles across input-length buckets,
        # so warm short/medium/long to absorb it before timing) — untimed.
        try:
            for wtext in (TEST_SENTENCES[0], TEST_SENTENCES[6], TEST_SENTENCES[12]):
                model.synthesize(wtext)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] warmup failed for {vkey}: {exc}")

        for accent in accents:
            if accent == "ruaccent" and not ruaccent_ok:
                continue
            acc = accentor if accent == "ruaccent" else None
            for denoise in denoises:
                if denoise == "dfn" and not dfn_ok:
                    continue
                dn = denoiser if denoise == "dfn" else None
                cell_id = f"{vkey}__{accent}__{denoise}"  # denoise ∈ none|norm|dfn
                cell_dir = audio_root / cell_id
                cell_dir.mkdir(exist_ok=True)
                print(f"  cell {cell_id}", flush=True)

                # --- representative sentences ---
                reps: list[dict] = []
                for idx, text in zip(REPRESENTATIVE_IDX, rep_sentences):
                    r = synth_f32(model, text, acc, denoise, dn)
                    save_wav(r["audio"], cell_dir / f"rep_{idx:02d}.wav", r["sample_rate"])
                    reps.append({k: v for k, v in r.items() if k != "audio"})
                    print(
                        f"    rep[{idx:02d}] ttfa={r['ttfa_ms']:.0f}ms "
                        f"(acc {r['accent_ms']:.0f} + syn {r['synth_ms']:.0f} + dn {r['denoise_ms']:.0f}) "
                        f"rtf={r['rtf']} peak={r['clip_peak']}"
                    )

                # --- streaming stitch (concat vs crossfade) ---
                stream_chunks: list[np.ndarray] = []
                stream_ttfas: list[float] = []
                sr = 24000
                for text in STREAMING_SEQUENCE:
                    r = synth_f32(model, text, acc, denoise, dn)
                    stream_chunks.append(r["audio"])
                    stream_ttfas.append(r["ttfa_ms"])
                    sr = r["sample_rate"]

                stitch_info = {}
                for stitch in args.stitch:
                    stitched = (
                        crossfade(stream_chunks, sr, ms=args.crossfade_ms)
                        if stitch == "crossfade"
                        else concat(stream_chunks)
                    )
                    save_wav(stitched, cell_dir / f"stream_{stitch}.wav", sr)
                    stitch_info[stitch] = {
                        "duration_s": round(len(stitched) / sr, 3),
                        "file": str((cell_dir / f"stream_{stitch}.wav").relative_to(out_dir)),
                    }

                cells.append({
                    "cell": cell_id,
                    "voice": vkey,
                    "accent": accent,
                    "denoise": denoise,
                    "avg_ttfa_ms": round(sum(x["ttfa_ms"] for x in reps) / len(reps), 1),
                    "avg_rtf": round(
                        sum(x["rtf"] for x in reps if x["rtf"]) / len(reps), 4
                    ),
                    "avg_synth_ms": round(sum(x["synth_ms"] for x in reps) / len(reps), 1),
                    "avg_accent_ms": round(sum(x["accent_ms"] for x in reps) / len(reps), 1),
                    "avg_denoise_ms": round(sum(x["denoise_ms"] for x in reps) / len(reps), 1),
                    "max_clip_peak": round(max(x["clip_peak"] for x in reps), 4),
                    "stream_avg_ttfa_ms": round(sum(stream_ttfas) / len(stream_ttfas), 1),
                    "representative": reps,
                    "streaming": stitch_info,
                    "audio_dir": str(cell_dir.relative_to(out_dir)),
                })

        try:
            model.unload()
        except Exception:  # noqa: BLE001
            pass

    return {
        "run_id": run_id,
        "device": device,
        "silero_version": args.version,
        "silero_sr": args.sr,
        "ruaccent_available": ruaccent_ok,
        "deepfilternet_available": dfn_ok,
        "crossfade_ms": args.crossfade_ms,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_reports(result: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write JSON + Markdown summary; return their paths."""
    run_id = result["run_id"]
    res_dir = out_dir / "results"
    res_dir.mkdir(exist_ok=True)
    json_path = res_dir / f"grid_{run_id}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    headers = ["cell", "voice", "accent", "denoise",
               "TTFA ms", "RTF", "synth ms", "acc ms", "dn ms", "peak"]
    rows = [
        [c["cell"], c["voice"], c["accent"], c["denoise"],
         c["avg_ttfa_ms"], c["avg_rtf"], c["avg_synth_ms"],
         c["avg_accent_ms"], c["avg_denoise_ms"], c["max_clip_peak"]]
        for c in sorted(result["cells"], key=lambda c: c["avg_ttfa_ms"])
    ]
    table = tabulate(rows, headers=headers, tablefmt="github")

    md = [
        f"# TTS enhancement grid — {run_id}",
        "",
        f"- device: `{result['device']}`  |  Silero: `{result.get('silero_version')}@"
        f"{result.get('silero_sr')}`  |  crossfade: {result['crossfade_ms']} ms",
        f"- RUAccent available: **{result['ruaccent_available']}**  |  "
        f"DeepFilterNet available: **{result['deepfilternet_available']}**",
        f"- cells: **{len(result['cells'])}**  |  audio under `grid_audio/<cell>/`",
        "",
        "Sorted by TTFA (accent + synthesis + denoise, all pre-playback). "
        "`peak` ≥ 1.0 flags a clipping/quantisation artefact to fix upstream of any denoiser.",
        "",
        table,
        "",
        "## How to audition",
        "- `grid_audio/<cell>/rep_00|06|12.wav` — short / medium / long single sentences.",
        "- `grid_audio/<cell>/stream_concat.wav` vs `stream_crossfade.wav` — listen at "
        "sentence seams for clicks; crossfade should remove them.",
        "- Compare `__raw__` vs `__ruaccent__` cells of the same voice for stress correctness.",
        "- Compare post modes on the same voice for the metallic buzz: "
        "`__none` → `__norm` → `__lp10` → `__lp8` → `__soxr`. lp8 smooths most (dullest); "
        "pick the lightest one that kills the metal.",
    ]
    md_path = res_dir / f"grid_{run_id}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTS enhancement experiment grid.")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--engines", type=lambda s: [x.strip() for x in s.split(",")],
                   default=["silero", "piper"], help="silero,piper")
    p.add_argument("--version", default="v5_5_ru",
                   help="Silero bundle version (e.g. v4_ru, v5_ru, v5_5_ru).")
    p.add_argument("--sr", type=int, default=24000, choices=[8000, 24000, 48000],
                   help="Silero output sample rate (Piper is fixed).")
    p.add_argument("--voices", type=lambda s: [x.strip() for x in s.split(",")], default=None,
                   help="Comma-separated voice keys (default: all in selected engines).")
    p.add_argument("--accent", type=lambda s: [x.strip() for x in s.split(",")],
                   default=["raw", "ruaccent"], help="raw,ruaccent")
    p.add_argument("--post", type=lambda s: [x.strip() for x in s.split(",")],
                   default=["none", "norm", "lp10", "lp8", "soxr"],
                   help="none,norm,lp10,lp8,soxr,dfn (dfn Linux-only)")
    p.add_argument("--stitch", type=lambda s: [x.strip() for x in s.split(",")],
                   default=["concat", "crossfade"], help="concat,crossfade")
    p.add_argument("--crossfade-ms", type=float, default=20.0)
    p.add_argument("--smoke", action="store_true",
                   help="Quick run: 1 silero + 1 piper voice only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke and args.voices is None:
        args.voices = [f"silero-{args.version.replace('_ru','').replace('_','')}-xenia-{args.sr//1000}k",
                       "piper-ruslan"]

    out_dir = Path(__file__).parent
    print(f"TTS grid — device={args.device}  engines={args.engines}  "
          f"silero={args.version}@{args.sr}  voices={args.voices or 'ALL'}  "
          f"accent={args.accent}  post={args.post}  stitch={args.stitch}")
    result = run_grid(args)

    json_path, md_path = write_reports(result, out_dir)
    print("\n" + "=" * 78)
    print(f"cells: {len(result['cells'])}")
    print(f"JSON : {json_path}")
    print(f"MD   : {md_path}")


if __name__ == "__main__":
    main()
