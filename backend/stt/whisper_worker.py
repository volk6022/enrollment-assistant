"""Whisper worker: dedicated single-worker CUDA pool, throttled partials.

`plan.md` §2/§3.3, `docs/streaming-research-findings.md` §2/§6: this is the
one component that pays for every extra millisecond spent on it, because the
GPU serializes whisper runs against LLM generation and against itself (§2:
four concurrent whisper calls on this GPU took 1.202s against 0.302s for one
-- ×4, no parallelism gained). Two rules here are load-bearing, not stylistic:

1. **Own single-worker `ThreadPoolExecutor`, never shared with
   `SileroWorker`'s pool.** A shared pool means TTS queues behind STT --
   exactly the latency this rewrite exists to remove (`plan.md` §2).
2. **At most one whisper run in flight, throttled to `STT_PARTIAL_MAX_HZ`.**
   A tick that arrives while the previous run hasn't returned is DROPPED,
   not queued -- queuing would let partial transcriptions pile up behind a
   busy GPU and eventually run on stale audio anyway (`plan.md` §3.3). Both
   rules live in `try_partial`, not in the caller, so a caller that polls
   too eagerly can't defeat them.

**CPU is not a fallback.** Measured RTF for this exact model is 5.24 on CPU
against the <0.3 NFR-03 budget -- 45-75x slower than CUDA, and unfixable by
quantization, thread count or beam width
(`docs/streaming-research-findings.md` §2). `device="cpu"` is therefore
rejected at construction instead of being silently accepted and left to fail
NFR-03 in production.

**Warmup runs from inside the worker thread**, never the caller's thread:
`start()` awaits `run_in_executor(self._load_and_warm)`, so both model
construction and the first inference happen on the same thread identity the
pool will keep using for every real request afterwards.
`docs/streaming-research-findings.md` §6 ("Аномалия в 1 секунду") found the
first CUDA call from a *fresh* thread pays a one-off initialization cost
(128 ms - 1 s observed) that is PROCESS-scoped, not thread-scoped -- get that
paid during startup, not during the first live partial.

**Warmup content matters, not just warmup timing** -- found while building
this module, not in the original research doc, so recorded here: warming
with `WHISPER_SAMPLE_RATE` of digital silence (the obvious choice, and what
`services/voice-gateway/app/faster_whisper_stt.py` does) leaves a
118-121 ms stall on the FIRST call against real speech afterwards, measured
on this exact box -- silence makes the decoder emit almost no tokens, so
whatever CTranslate2 CUDA kernels the multi-step beam-search generation loop
needs for a real, multi-word utterance never actually get exercised (and
presumably JIT/autotuned) during warmup. Synthetic Gaussian noise doesn't
fix it either (measured: still 121.4 ms) -- only real speech-shaped audio
drove the stall down to the steady-state 1.5-5 ms this module needs to meet
NFR-04. `warmup_audio_pcm16` exists so a caller can pass real audio (e.g.
bytes read from `GREETING_AUDIO_PATH`, which every deployment already ships
per FR-01) instead of relying on the synthetic fallback, which is
best-effort only and NOT guaranteed to avoid the stall.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from backend.audio.ring import AudioRing

_WHISPER_SAMPLE_RATE = 16000
_WARMUP_SECONDS = 1.0


class WhisperSegmentProtocol(Protocol):
    """Structural shape of one faster-whisper `Segment` -- only what we use."""

    text: str


class WhisperModelProtocol(Protocol):
    """Structural type for anything `WhisperWorker` can drive. `WhisperWorker`
    never imports `faster_whisper` itself outside of the default factory --
    unit tests inject a scripted fake so the throttle/window/in-flight logic
    can be verified without CUDA or model weights.
    """

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str,
        beam_size: int,
        vad_filter: bool,
    ) -> tuple[Iterable[WhisperSegmentProtocol], object]: ...


@dataclass(slots=True, frozen=True)
class TranscriptionResult:
    """One whisper run's output, partial or final.

    `window_start_ms`/`window_end_ms` are the ring offsets actually read --
    callers report these back in `transcript.update` (`contracts/websocket.md`
    §3.3) rather than re-deriving them, since the sliding-window math that
    produced `window_start_ms` lives entirely in this module.
    """

    text: str
    window_start_ms: int
    window_end_ms: int
    wall_s: float
    is_final: bool = False


def _default_model_factory(
    *, model_size: str, device: str, compute_type: str
) -> Callable[[], WhisperModelProtocol]:
    def factory() -> WhisperModelProtocol:
        from faster_whisper import WhisperModel

        return WhisperModel(model_size, device=device, compute_type=compute_type)

    return factory


class WhisperWorker:
    """One instance per process (plan.md §9: "Вне области ... несколько
    одновременных пользователей" -- a single dialogue session at a time is
    the whole scope), owning its own model, pool and throttle state.
    """

    def __init__(
        self,
        *,
        model_size: str,
        device: str,
        compute_type: str,
        language: str,
        partial_max_hz: float,
        window_seconds: float,
        overlap_seconds: float,
        final_pass: bool,
        warmup_audio_pcm16: bytes | None = None,
        model_factory: Callable[[], WhisperModelProtocol] | None = None,
    ) -> None:
        if device == "cpu":
            raise ValueError(
                "STT_DEVICE=cpu is not viable for whisper: measured RTF 5.24 "
                "against the <0.3 NFR-03 budget, 45-75x slower than CUDA and "
                "unfixable by quantization/thread-count/beam-width tuning "
                "(docs/streaming-research-findings.md §2). Whisper must run "
                "on device='cuda'."
            )
        self._language = language
        self._min_interval_s = 1.0 / partial_max_hz
        self._window_ms = int(window_seconds * 1000)
        self._overlap_ms = int(overlap_seconds * 1000)
        self.final_pass_enabled = final_pass

        self._warmup_audio_pcm16 = warmup_audio_pcm16
        self._model_factory = model_factory or _default_model_factory(
            model_size=model_size, device=device, compute_type=compute_type
        )
        self._model: WhisperModelProtocol | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

        self._in_flight = False
        self._last_started_at: float | None = None

    @property
    def is_warm(self) -> bool:
        return self._model is not None

    async def start(self) -> None:
        """Loads the model and runs one throwaway transcription, both inside
        the worker thread -- see the module docstring for why this can't be
        `WhisperModel(...)` called directly on whatever thread constructs
        this object.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._pool, self._load_and_warm)

    def _load_and_warm(self) -> None:
        """See the module docstring's "Warmup content matters" note: silence
        does NOT exercise the same CUDA kernels a real multi-word decode
        does, so this prefers `_warmup_audio_pcm16` (real speech) when the
        caller supplied it, over the synthetic (silence) fallback that is
        known, measured, not to fully absorb the first-call cost.
        """
        self._model = self._model_factory()
        if self._warmup_audio_pcm16:
            audio = (
                np.frombuffer(self._warmup_audio_pcm16, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
        else:
            audio = np.zeros(int(_WHISPER_SAMPLE_RATE * _WARMUP_SECONDS), dtype=np.float32)
        list(
            self._model.transcribe(
                audio, language=self._language, beam_size=5, vad_filter=True
            )[0]
        )

    def window_start_ms(self, *, speech_start_ms: int, now_ms: int) -> int:
        """`plan.md` §3.3: the window is `STT_WINDOW_SECONDS` from
        `speech_start_ms` while the turn is shorter than that; once the turn
        runs longer, the window slides forward in `(window - overlap)`
        steps so consecutive runs still share `STT_OVERLAP_SECONDS` of
        audio. Whisper has no memory between calls, so a slide with no
        overlap risks losing a word that straddles the boundary.
        """
        elapsed_ms = now_ms - speech_start_ms
        if elapsed_ms <= self._window_ms:
            return speech_start_ms
        step_ms = max(self._window_ms - self._overlap_ms, 1)
        slides = 1 + (elapsed_ms - self._window_ms) // step_ms
        return speech_start_ms + slides * step_ms

    def try_partial(
        self, ring: AudioRing, *, speech_start_ms: int, now_ms: int
    ) -> asyncio.Task[TranscriptionResult] | None:
        """Non-blocking: starts a partial run and returns its `Task`, or
        returns `None` immediately -- without starting anything -- if
        throttled, already in flight, or there is no audio to read yet.

        Callers are expected to call this on every audio tick while VAD says
        the interlocutor is speaking (`plan.md` §3.3: "только при активном
        VAD" is the caller's job -- this method only enforces the GPU-side
        half of the rule, the Hz cap and the in-flight guard).
        """
        if self._model is None:
            raise RuntimeError("WhisperWorker.start() must be awaited before try_partial()")
        if self._in_flight:
            return None
        if self._last_started_at is not None:
            if time.monotonic() - self._last_started_at < self._min_interval_s:
                return None
        start_ms = self.window_start_ms(speech_start_ms=speech_start_ms, now_ms=now_ms)
        audio = ring.snapshot(start_ms, now_ms)
        if audio.size == 0:
            return None
        self._in_flight = True
        self._last_started_at = time.monotonic()
        loop = asyncio.get_running_loop()
        return loop.create_task(self._run(audio, start_ms, now_ms, is_final=False))

    async def run_final(
        self, ring: AudioRing, *, speech_start_ms: int, end_ms: int
    ) -> TranscriptionResult:
        """The full-turn pass gated by `STT_FINAL_PASS` (OQ-03, off by
        default: the last partial usually matches and this saves ~310ms of
        877ms, `docs/streaming-research-findings.md` §6). Callers only
        invoke this when `final_pass_enabled` is True. Not throttled by the
        Hz cap -- it fires once per turn, on an event, not on a timer -- but
        it still runs on the same single-worker pool, so it naturally queues
        behind whatever partial is still finishing rather than racing it.
        """
        if self._model is None:
            raise RuntimeError("WhisperWorker.start() must be awaited before run_final()")
        audio = ring.snapshot(speech_start_ms, end_ms)
        return await self._run(audio, speech_start_ms, end_ms, is_final=True)

    async def _run(
        self,
        audio_i16: np.ndarray,
        window_start_ms: int,
        window_end_ms: int,
        *,
        is_final: bool,
    ) -> TranscriptionResult:
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        try:
            text = await loop.run_in_executor(self._pool, self._transcribe_blocking, audio_i16)
        finally:
            self._in_flight = False
        return TranscriptionResult(
            text=text,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            wall_s=time.perf_counter() - t0,
            is_final=is_final,
        )

    def _transcribe_blocking(self, audio_i16: np.ndarray) -> str:
        """Runs on the pool's worker thread only. `audio_i16` is a snapshot
        copy from `AudioRing.snapshot()` -- this thread never touches the
        ring itself (plan.md §2 "мутируем на loop, в поток отдаём снимок").
        """
        assert self._model is not None
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio_f32, language=self._language, beam_size=5, vad_filter=True
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def close(self) -> None:
        self._pool.shutdown(wait=True)
