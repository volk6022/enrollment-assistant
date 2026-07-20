"""Silero version × sample-rate shoot-out — targets the metallic "robo-buzz".

Ivan's ear flagged a pervasive metallic hum on Silero (v4). That artefact is
vocoder quantisation/aliasing, NOT background hiss — a denoiser won't touch it.
The two real levers are the model **version** (we were pinned to the old v4_ru)
and the output **sample rate** (we ran 24 kHz; 48 kHz aliases less). This renders
every version × sample-rate × voice on short, real-length answers so the buzz can
be judged by ear.

Each version bundle is loaded once; speaker and sample_rate are per-call, so the
whole sweep is 3 model loads. Audio → grid_audio/silero_versions/<cell>/.

    python silero_version_compare.py                 # all versions/voices/SRs
    python silero_version_compare.py --versions v4_ru,v5_5_ru --voices xenia,aidar
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from audio_utils import save_wav
from enhance import clipping_stats

# Short, real-length answers (~2–3 sentences is Ivan's typical reply).
ANSWERS = [
    "Добрый день! Документы принимаются до тридцать первого июля.",
    "Минимальный проходной балл в прошлом году составил двести восемьдесят четыре балла. "
    "Подавайте заявление через приёмную комиссию или портал госуслуг.",
    "Вступительные испытания начинаются второго августа. Если остались вопросы, звоните нам в рабочее время.",
]

DEFAULT_VERSIONS = ["v4_ru", "v5_ru", "v5_5_ru"]
DEFAULT_VOICES = ["aidar", "baya", "kseniya", "xenia"]
DEFAULT_SRS = [24000, 48000]


def _hf_roughness(audio: np.ndarray) -> float:
    """Crude proxy for the 'staircase' buzz: RMS energy in the top octave.

    A cleaner vocoder puts less energy in the very top of the band. Not a ground
    truth — just a number to sort by; trust the audio. Returns HF/total RMS ratio.
    """
    if len(audio) < 16:
        return 0.0
    # First difference emphasises high frequencies; ratio of its RMS to signal RMS.
    d = np.diff(audio)
    sig = float(np.sqrt(np.mean(audio**2))) + 1e-9
    return round(float(np.sqrt(np.mean(d**2))) / sig, 4)


def render(versions, voices, srs) -> dict:
    import torch
    import warnings
    warnings.filterwarnings("ignore")

    out_dir = Path(__file__).parent
    audio_root = out_dir / "grid_audio" / "silero_versions"
    audio_root.mkdir(parents=True, exist_ok=True)

    cells = []
    for ver in versions:
        print(f"\n=== {ver} ===", flush=True)
        try:
            model, _ = torch.hub.load(
                "snakers4/silero-models", "silero_tts",
                language="ru", speaker=ver, trust_repo=True,
            )
            model.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {ver}: {exc!r}")
            continue

        for voice in voices:
            for sr in srs:
                cell = f"{ver}__{voice}__{sr // 1000}k"
                cell_dir = audio_root / cell
                cell_dir.mkdir(exist_ok=True)
                peaks, roughs, times = [], [], []
                ok = True
                for i, text in enumerate(ANSWERS):
                    try:
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            wav = model.apply_tts(
                                text=text, speaker=voice, sample_rate=sr,
                                put_accent=True, put_yo=True,
                            )
                        dt = time.perf_counter() - t0
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [SKIP] {cell} sent{i}: {exc!r}")
                        ok = False
                        break
                    a = wav.squeeze().cpu().numpy().astype(np.float32)
                    save_wav(a, cell_dir / f"ans_{i}.wav", sr)
                    peaks.append(clipping_stats(a).peak)
                    roughs.append(_hf_roughness(a))
                    times.append(dt)
                if not ok:
                    continue
                cells.append({
                    "cell": cell, "version": ver, "voice": voice, "sample_rate": sr,
                    "max_peak": round(max(peaks), 4),
                    "avg_hf_roughness": round(sum(roughs) / len(roughs), 4),
                    "avg_synth_s": round(sum(times) / len(times), 3),
                    "audio_dir": str(cell_dir.relative_to(out_dir)),
                })
                print(f"  {cell:28s} peak={max(peaks):.3f} "
                      f"hf_rough={sum(roughs)/len(roughs):.3f} synth={sum(times)/len(times):.2f}s")
    return {"cells": cells}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--versions", type=lambda s: s.split(","), default=DEFAULT_VERSIONS)
    p.add_argument("--voices", type=lambda s: s.split(","), default=DEFAULT_VOICES)
    p.add_argument("--srs", type=lambda s: [int(x) for x in s.split(",")], default=DEFAULT_SRS)
    args = p.parse_args()

    result = render(args.versions, args.voices, args.srs)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    res_path = Path(__file__).parent / "results" / f"silero_versions_{run_id}.json"
    res_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncells: {len(result['cells'])}  ->  {res_path}")
    print("Audio: grid_audio/silero_versions/<version>__<voice>__<sr>/ans_0..2.wav")
    print("Lower avg_hf_roughness ≈ smoother (less staircase buzz) — but TRUST YOUR EARS.")


if __name__ == "__main__":
    main()
