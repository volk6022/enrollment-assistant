"""Empirically verify which SSML features Silero v5_5_ru actually honours.

Not "does it error" — "does it CHANGE the audio in the intended way":
  * rate  → speech duration (slow → longer, fast → shorter)
  * break → duration (adds the pause length)
  * pitch → mean F0 (librosa.yin) over voiced frames
Also XML-validates every string intonation_ssml.py emits for its examples.

Renders audio to grid_audio/ssml_verify/ so results can be confirmed by ear.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from audio_utils import save_wav
from intonation_ssml import sentence_to_full_ssml

BASE = "Документы принимаются до тридцать первого июля"
VOICE = "baya"
SR = 48000

# (label, ssml_text or None for plain text). None → call apply_tts(text=...).
CASES = [
    ("plain_text", None),
    ("ssml_plain", f"<speak><p><s>{BASE}.</s></p></speak>"),
    ("rate_slow", f'<speak><prosody rate="slow">{BASE}.</prosody></speak>'),
    ("rate_x-slow", f'<speak><prosody rate="x-slow">{BASE}.</prosody></speak>'),
    ("rate_fast", f'<speak><prosody rate="fast">{BASE}.</prosody></speak>'),
    ("rate_x-fast", f'<speak><prosody rate="x-fast">{BASE}.</prosody></speak>'),
    ("pitch_high", f'<speak><prosody pitch="high">{BASE}.</prosody></speak>'),
    ("pitch_x-high", f'<speak><prosody pitch="x-high">{BASE}.</prosody></speak>'),
    ("pitch_low", f'<speak><prosody pitch="low">{BASE}.</prosody></speak>'),
    ("pitch_x-low", f'<speak><prosody pitch="x-low">{BASE}.</prosody></speak>'),
    ("break_800", f'<speak>Документы принимаются<break time="800ms"/>до тридцать первого июля.</speak>'),
    ("break_strength", f'<speak>Документы принимаются<break strength="strong"/>до тридцать первого июля.</speak>'),
    ("question_native", "<speak><p><s>Документы уже приняли?</s></p></speak>"),
    ("exc_combo", f'<speak><prosody pitch="x-high" rate="fast">{BASE}!</prosody></speak>'),
]


def mean_f0(audio: np.ndarray, sr: int) -> float:
    """Mean F0 over voiced frames via librosa.yin — a real pitch number."""
    try:
        import librosa
        f0 = librosa.yin(audio, fmin=70, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        # keep plausible voiced range only
        f0 = f0[(f0 > 70) & (f0 < 400)]
        return round(float(np.median(f0)), 1) if len(f0) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> None:
    import torch
    import warnings
    warnings.filterwarnings("ignore")

    out = Path(__file__).parent / "grid_audio" / "ssml_verify"
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. XML validity of the module's own output ---
    print("=== XML validity of intonation_ssml.py output ===")
    for s in [
        "Ты серьёзно? [q] Мы же [emp]точно договаривались на пятницу.",
        "Погоди... [pause:long] я не понимаю, что произошло.",
        "Это невероятно!",
        "Ты уже поел?",
        "Обычное повествовательное предложение без маркеров.",
    ]:
        ssml = sentence_to_full_ssml(s)
        try:
            ET.fromstring(ssml)
            print(f"  OK   {s[:40]!r:44} -> {ssml}")
        except ET.ParseError as exc:
            print(f"  BAD  {s[:40]!r:44} -> {exc}: {ssml}")

    # --- 2. Load model ---
    print(f"\n=== loading v5_5_ru ({VOICE} @ {SR}) ===")
    model, _ = torch.hub.load(
        "snakers4/silero-models", "silero_tts",
        language="ru", speaker="v5_5_ru", trust_repo=True,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    # warmup
    with torch.no_grad():
        model.apply_tts(text="прогрев", speaker=VOICE, sample_rate=SR)

    # --- 3. Feature probes ---
    print("\n=== SSML feature probes (baya @ 48k) ===")
    print(f"{'case':18} {'status':8} {'dur_s':>7} {'F0_Hz':>7}  note")
    base_dur = base_f0 = None
    rows = []
    for label, ssml in CASES:
        try:
            with torch.no_grad():
                if ssml is None:
                    wav = model.apply_tts(text=BASE + ".", speaker=VOICE, sample_rate=SR)
                else:
                    wav = model.apply_tts(ssml_text=ssml, speaker=VOICE, sample_rate=SR)
            a = wav.squeeze().cpu().numpy().astype(np.float32)
            dur = round(len(a) / SR, 3)
            f0 = mean_f0(a, SR)
            save_wav(a, out / f"{label}.wav", SR)
            status = "OK"
        except Exception as exc:  # noqa: BLE001
            print(f"{label:18} {'ERROR':8} {'-':>7} {'-':>7}  {type(exc).__name__}: {str(exc)[:60]}")
            rows.append((label, "ERROR", None, None))
            continue

        if label == "ssml_plain":
            base_dur, base_f0 = dur, f0
        note = ""
        if base_dur and label.startswith(("rate", "break", "exc")):
            note = f"Δdur={dur - base_dur:+.2f}s vs ssml_plain"
        elif base_f0 and label.startswith("pitch"):
            note = f"ΔF0={f0 - base_f0:+.0f}Hz vs ssml_plain"
        print(f"{label:18} {status:8} {dur:>7} {f0:>7}  {note}")
        rows.append((label, status, dur, f0))

    print(f"\nAudio: grid_audio/ssml_verify/<case>.wav")
    print("Read: rate works if Δdur has the right sign; pitch works if ΔF0 moves; "
          "break works if Δdur ≈ pause length. If a tag's numbers match ssml_plain, "
          "Silero silently ignored it.")


if __name__ == "__main__":
    main()
