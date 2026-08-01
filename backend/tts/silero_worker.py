"""Silero worker: dedicated single-worker CPU pool, `run_in_executor` only.

`plan.md` §3.4/§2, `docs/streaming-research-findings.md` §3: Silero stays on
CPU (RTF 0.066, VRAM freed for whisper). The pool exists to make
`run_in_executor` unavoidable -- the bug this fixes is that the current
gateway (`services/voice-gateway/app/main.py:95,197`) calls
`SileroTTSClient.synthesize` synchronously inside an `async def` handler and
blocks the event loop for the whole synthesis call. `SileroWorker.synthesize`
is a coroutine that always hands the blocking `apply_tts` call to its own
thread; nothing in this module is ever safe to call directly from the event
loop's own frame.

Own single-worker `ThreadPoolExecutor`, never shared with `WhisperWorker`'s
pool (`plan.md` §2: a shared pool means synthesis queues behind
transcription -- exactly the latency this rewrite exists to remove).

Text is normalized (`text_normalize.normalize_for_tts`, ported byte-for-byte
from `services/voice-gateway/app/text_normalize.py`) before it reaches
Silero -- bare digit runs and Latin-script terms are otherwise silently
dropped or garble the surrounding sentence (see that module's docstring).
Nothing else from the legacy gateway's post-chain (RUAccent stress marking,
soxr/de-esser DSP) is in scope here: `plan.md` §3.4 names only normalization
as a hard requirement for this worker.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from backend.tts.text_normalize import normalize_for_tts

_WARMUP_TEXTS = (
    "Прогрев.",
    "Короткий прогрев модели синтеза речи перед первым настоящим запросом.",
)


class SileroModelProtocol(Protocol):
    """Structural type for anything `SileroWorker` can drive -- only the one
    method actually used. Unit tests inject a scripted fake so the
    normalize -> synthesize -> resample pipeline can be verified without
    `torch.hub` or model weights.
    """

    def apply_tts(
        self,
        *,
        text: str,
        speaker: str,
        sample_rate: int,
        put_accent: bool,
        put_yo: bool,
    ) -> object: ...


@dataclass(slots=True, frozen=True)
class SynthesisResult:
    """`pcm16` is raw PCM16LE mono at `sample_rate` -- no WAV header, ready
    to frame onto the wire per `contracts/websocket.md` §3.1.
    """

    pcm16: bytes
    sample_rate: int
    wall_s: float
    audio_seconds: float


def _default_model_factory(
    *, model_variant: str, device: str, cache_dir: str | None
) -> Callable[[], SileroModelProtocol]:
    def factory() -> SileroModelProtocol:
        import torch

        if cache_dir:
            torch.hub.set_dir(cache_dir)
        model, _ = torch.hub.load(
            "snakers4/silero-models",
            "silero_tts",
            language="ru",
            speaker=model_variant,
            trust_repo=True,
        )
        model.to(device)
        return model

    return factory


class SileroWorker:
    """One instance per process, mirroring `WhisperWorker`. `model_variant`
    is the torch.hub model identifier -- `TTS_MODEL` in `.env.example`,
    which is the literal string `"v5_5_ru"` (NOT `"v5_5"`: torch.hub raises
    `AssertionError: Speaker not in the supported list` on the bare form,
    since the model name doubles as torch.hub's own `speaker=` kwarg --
    see `.env.example`'s comment on this). `speaker` is a different knob:
    the actual voice passed to `apply_tts` (`TTS_SPEAKER`, e.g. `"xenia"`;
    valid set is aidar/baya/kseniya/eugene/xenia). Do not confuse the two.
    """

    def __init__(
        self,
        *,
        speaker: str,
        native_sample_rate: int,
        model_variant: str = "v5_5_ru",
        output_sample_rate: int | None = None,
        device: str = "cpu",
        cache_dir: str | None = None,
        model_factory: Callable[[], SileroModelProtocol] | None = None,
    ) -> None:
        self._speaker = speaker
        self._native_sample_rate = native_sample_rate
        self._output_sample_rate = output_sample_rate or native_sample_rate
        self._model_factory = model_factory or _default_model_factory(
            model_variant=model_variant, device=device, cache_dir=cache_dir
        )
        self._model: SileroModelProtocol | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    @property
    def is_warm(self) -> bool:
        return self._model is not None

    async def start(self) -> None:
        """Loads the model and runs a couple of throwaway syntheses, both
        inside the worker thread -- `torch.hub.load` plus Silero's first-call
        JIT compile easily costs seconds; paying that during startup instead
        of on the first live sentence is the same reasoning `WhisperWorker`
        applies to CUDA init (`plan.md` §2 "Прогрев").
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._pool, self._load_and_warm)

    def _load_and_warm(self) -> None:
        self._model = self._model_factory()
        for text in _WARMUP_TEXTS:
            self._synthesize_blocking(text)

    async def synthesize(self, text: str) -> SynthesisResult:
        """Always goes through `run_in_executor` -- see module docstring for
        the bug in the legacy gateway this exists to not repeat.
        """
        if self._model is None:
            raise RuntimeError("SileroWorker.start() must be awaited before synthesize()")
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        pcm16, audio_seconds = await loop.run_in_executor(
            self._pool, self._synthesize_blocking, text
        )
        return SynthesisResult(
            pcm16=pcm16,
            sample_rate=self._output_sample_rate,
            wall_s=time.perf_counter() - t0,
            audio_seconds=audio_seconds,
        )

    def _synthesize_blocking(self, text: str) -> tuple[bytes, float]:
        """Runs on the pool's worker thread only."""
        normalized = normalize_for_tts(text)
        if not normalized.strip():
            return b"", 0.0
        assert self._model is not None
        import torch

        with torch.no_grad():
            wav = self._model.apply_tts(
                text=normalized,
                speaker=self._speaker,
                sample_rate=self._native_sample_rate,
                put_accent=True,
                put_yo=True,
            )
        audio = wav.squeeze().cpu().numpy().astype(np.float32)
        if self._output_sample_rate != self._native_sample_rate:
            import soxr

            audio = soxr.resample(
                audio, self._native_sample_rate, self._output_sample_rate
            ).astype(np.float32)
        audio_seconds = len(audio) / self._output_sample_rate
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return pcm16, audio_seconds

    def close(self) -> None:
        self._pool.shutdown(wait=True)
