"""Load test: NFR-04 event loop stall budget under concurrent STT + TTS.

`tasks.md` T-05 acceptance: "разрыв event loop < 50 мс под одновременной
нагрузкой STT+TTS". Method mirrors `experiment-streaming/gil_probe.py`'s
`Probe` (an `await asyncio.sleep(0)` loop running on the event loop; the max
gap between iterations is, near enough, the longest continuous stretch
anything else held the loop for) driven concurrently with
`experiment-streaming/streaming_poc.py`'s contention pattern (STT and TTS
pools hammered from `asyncio.gather`, each on its own single-worker pool).

This is a real-hardware test, not a mock: it loads faster-whisper on CUDA and
Silero on CPU for real and measures actual latency -- a mock GIL is not a
thing that can hold or release anything, and `docs/streaming-research-
findings.md` §1/§6 is exactly the record of that lesson being learned the
hard way for whisper/silero specifically. Skips cleanly if no CUDA device is
visible to torch (whisper is not viable on CPU -- see
`WhisperWorker.__init__`).
"""

from __future__ import annotations

import asyncio
import statistics
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from backend.audio.ring import AudioRing
from backend.stt.whisper_worker import WhisperWorker
from backend.tts.silero_worker import SileroWorker

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_WAV = REPO_ROOT / "experiment-stt" / "test_data" / "sample_01.wav"

_TIGHT_TICK_S = 0.0  # gil_probe.py: `asyncio.sleep(0)` -- the discriminating probe
_LOOP_STALL_BUDGET_MS = 50.0  # NFR-04
_ROUNDS = 6
_TTS_SENTENCE = (
    "Для поступления нужны заявление, документ об образовании и медицинская справка."
)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _silero_loadable() -> bool:
    """`torch.hub.load(..., "silero_tts", ...)` unconditionally imports
    `omegaconf` deep inside the cached hub entrypoint
    (`~/.cache/torch/hub/snakers4_silero-models_master/src/silero/silero.py`)
    to parse `models.yml`. It is not declared in `pyproject.toml` (only
    `silero-vad` is -- a different package, used by `backend.audio.vad`, not
    this test) and is missing from this environment as of T-05. Skip with a
    clear reason instead of letting the failure surface as an opaque
    `ModuleNotFoundError` from three frames inside `torch.hub`.
    """
    try:
        import omegaconf  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not _cuda_available(),
        reason=(
            "whisper requires CUDA (docs/streaming-research-findings.md §2: CPU RTF "
            "5.24 vs <0.3 required) -- no GPU visible to torch on this runner"
        ),
    ),
    pytest.mark.skipif(
        not _silero_loadable(),
        reason=(
            "silero_tts's torch.hub entrypoint requires 'omegaconf', which is not "
            "declared in pyproject.toml or installed in this environment -- add it "
            "next to the other T-05 additions there (see that file's comment on "
            "packages uv sync silently wiped before) rather than installing it "
            "ad hoc; do not add it from inside this task."
        ),
    ),
]


def _load_wav_pcm16_16k(path: Path) -> bytes:
    """`AudioRing`/whisper require 16kHz mono; the checked-in sample wavs are
    not (`sample_01.wav` is 24kHz), so this resamples the same way
    `experiment-streaming/gil_probe.py`'s `load_wav_f32` does.
    """
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000:
        import soxr

        pcm = soxr.resample(pcm, sample_rate, 16000).astype(np.float32)
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class LoopStallProbe:
    """`await asyncio.sleep(0)` in a tight loop; each gap between iterations
    is how long something else held the event loop for
    (`experiment-streaming/gil_probe.py`'s method, ported directly rather
    than reimplemented differently).
    """

    def __init__(self) -> None:
        self.gaps: list[float] = []
        self._stop = False

    async def run(self) -> None:
        prev = time.perf_counter()
        while not self._stop:
            await asyncio.sleep(_TIGHT_TICK_S)
            now = time.perf_counter()
            self.gaps.append(now - prev)
            prev = now

    def stop(self) -> None:
        self._stop = True

    def stats(self) -> dict[str, float]:
        gaps = sorted(self.gaps)

        def pct(p: float) -> float:
            return gaps[min(len(gaps) - 1, int(len(gaps) * p))]

        return {
            "p50_ms": pct(0.50) * 1000,
            "p99_ms": pct(0.99) * 1000,
            "max_ms": gaps[-1] * 1000 if gaps else 0.0,
            "n": float(len(gaps)),
        }


@pytest.mark.asyncio
async def test_event_loop_stall_under_concurrent_stt_tts() -> None:
    sample_rate = 16000
    pcm = _load_wav_pcm16_16k(SAMPLE_WAV)
    audio_ms = (len(pcm) // 2) * 1000 // sample_rate

    ring = AudioRing(sample_rate=sample_rate, capacity_seconds=60)
    ring.append(pcm, at_ms=0)

    stt = WhisperWorker(
        model_size="large-v3-turbo",
        device="cuda",
        compute_type="int8",
        language="ru",
        partial_max_hz=2.5,
        window_seconds=15.0,
        overlap_seconds=2.0,
        final_pass=False,
        # Warming with silence (the naive default) measurably does NOT avoid
        # the first-call CUDA stall against real speech -- see
        # `WhisperWorker`'s "Warmup content matters" note, discovered while
        # writing this exact test: silence-warmed, round 0 stalled the loop
        # 118ms; warmed with this same real clip, steady state from round 0.
        warmup_audio_pcm16=pcm,
    )
    tts = SileroWorker(speaker="xenia", native_sample_rate=48000, output_sample_rate=16000)
    await asyncio.gather(stt.start(), tts.start())

    probe = LoopStallProbe()
    probe_task = asyncio.create_task(probe.run())
    await asyncio.sleep(0.1)  # let the probe settle before load starts, per gil_probe.py

    stt_wall: list[float] = []
    tts_wall: list[float] = []

    async def stt_loop() -> None:
        for _ in range(_ROUNDS):
            task = None
            while task is None:
                task = stt.try_partial(ring, speech_start_ms=0, now_ms=audio_ms)
                if task is None:
                    await asyncio.sleep(0.05)
            t0 = time.perf_counter()
            result = await task
            stt_wall.append(time.perf_counter() - t0)
            assert result.text, "expected a real transcript from sample_01.wav, got none"

    async def tts_loop() -> None:
        for _ in range(_ROUNDS):
            t0 = time.perf_counter()
            result = await tts.synthesize(_TTS_SENTENCE)
            tts_wall.append(time.perf_counter() - t0)
            assert result.pcm16, "expected non-empty PCM16 output"

    try:
        await asyncio.gather(stt_loop(), tts_loop())
    finally:
        probe.stop()
        await probe_task
        stt.close()
        tts.close()

    stats = probe.stats()
    stt_rtf = statistics.fmean(stt_wall) / (audio_ms / 1000)

    print(
        f"\nloop stall (NFR-04, budget {_LOOP_STALL_BUDGET_MS:.0f}ms): "
        f"p50={stats['p50_ms']:.2f}ms p99={stats['p99_ms']:.2f}ms "
        f"max={stats['max_ms']:.2f}ms  n={int(stats['n'])}"
    )
    print(
        f"stt wall (NFR-03, RTF budget <0.3): mean={statistics.fmean(stt_wall):.3f}s "
        f"max={max(stt_wall):.3f}s  audio={audio_ms / 1000:.2f}s  RTF={stt_rtf:.3f}"
    )
    print(f"tts wall: mean={statistics.fmean(tts_wall):.3f}s max={max(tts_wall):.3f}s")

    assert stats["max_ms"] < _LOOP_STALL_BUDGET_MS, (
        f"NFR-04 violated: event loop stalled {stats['max_ms']:.1f}ms under "
        f"concurrent STT+TTS, budget is {_LOOP_STALL_BUDGET_MS:.0f}ms"
    )
    assert stt_rtf < 0.3, f"NFR-03 violated: STT RTF {stt_rtf:.3f} >= 0.3 budget"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
