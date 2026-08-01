"""`DialogueSession` -- the T-09 orchestration: socket -> ring -> VAD -> STT ->
automaton -> RAG/LLM -> TTS -> socket, one instance per `/ws/dialogue`
connection. Protocol: `contracts/websocket.md`. Requirements: FR-01..FR-03,
FR-09..FR-17.

This module is glue, not a new layer -- every non-trivial decision already
lives in a wave-1/wave-2 module (`backend/audio`, `backend/stt`, `backend/tts`,
`backend/llm`, `backend/rag`, `backend/dialogue`). What follows documents the
project's known traps and exactly where this module addresses each one, so a
reviewer can check them off against real code instead of prose:

1. **Greeting cut is two `dialogue.qnt` steps, not one** (plan.md §7). The
   guard on `userInterruptsGreeting` needs `userSpeaking` already `True`,
   which only `userStartsSpeaking` can raise, and `DialogueMachine.step()`
   applies exactly one action per call. `_on_audio_frame` enqueues BOTH ticks
   back-to-back, synchronously, the instant `VadGate.feed()` returns a
   `SpeechStarted` for a chunk -- see `_enqueue_speech_start_ticks`. Nothing
   else can be interleaved between them because both `put_nowait()` calls
   happen without an `await` in between.
2. **`decide()` is called by automaton nodes, never by this module.**
   `_DecidingInterject`/`_DecidingBargeIn` (backend/dialogue/machine.py) own
   that call; this module owns only the cheap non-LLM gate -- reporting
   truthful `user_speaking`/`elapsed_ms`/`turn_ended` on every tick and
   calling `DialogueMachine.step()`. Grep of this file for `.decide(` other
   than inside `_RecordingDecisionClient` (a pass-through wrapper that exists
   only to capture `reason`/`understood` for telemetry and `ScenarioContext`,
   see below) should come back empty.
3. **`VadGate.notify_agent_speaking(bool)` is called on every playback
   start/stop** -- `_start_agent_audio` / `_stop_agent_audio`, wrapping
   greeting, every generated/fixed reply, and the farewell. Skipping this
   silently breaks barge-in detection (memory.md §3); it is centralized here
   so no code path that sends agent audio can forget it.
4. **`DialogueMemory.reset_transcript()` fires only from `session.reset`.**
   `_handle_session_reset` is the ONLY caller. Turn completion
   (`_finish_user_turn`) calls `add_user_turn`, never `reset_transcript`.
5. **Whisper is warmed with real speech, not silence.** `ensure_greeting_audio`
   renders the greeting once via `SileroWorker` (also FR-01's requirement) and
   `backend/app.py` passes that exact file's PCM16 bytes into
   `WhisperWorker(warmup_audio_pcm16=...)` -- this module doesn't warm Whisper
   itself, it only guarantees the audio exists before `WhisperWorker.start()`
   is awaited.
6. **Prefill is fed during speech, not after.** Every whisper partial that
   lands (`_on_partial_result`) is appended to `DialogueMemory`'s session-wide
   transcript buffer immediately (FR-10) -- by the time a turn ends and
   `stream_answer` starts, the transcript prefix is already what the LLM
   prompt's volatile tail will contain, so `cache_prompt` has almost nothing
   new to evaluate for that portion (llm.md §4.3).
7. **State mutates on the loop; workers get snapshots.** `AudioRing.append`/
   `snapshot` already enforce this (audio/ring.py); this module never passes
   `self._ring` itself into an executor, only reads returned dataclasses
   (`TranscriptionResult`, `SynthesisResult`) back from worker coroutines that
   already did the `run_in_executor` hop internally.
8. **Three separate single-worker pools.** `SessionDependencies` carries a
   dedicated `rag_executor` (`ThreadPoolExecutor(max_workers=1)`, built by
   `backend/app.py`) passed explicitly to every `RagPipeline.asearch(...,
   executor=rag_executor)` call in this module -- the default `None` would
   fall back to asyncio's shared pool (rag/pipeline.py's own docstring flags
   this). `WhisperWorker`/`SileroWorker` already own their pools internally.
9. **At most one `decide()` next to an active `stream_answer`.**
   `LlamaClient` enforces this with a semaphore and raises
   `DecisionInFlightError` instead of queuing (llm.md §7: a decision that
   lands late is worse than none). `_run_automaton_step` catches exactly that
   exception around every step that might trigger a decision and treats it as
   "no decision this tick" -- logged, not fatal; by construction (a single
   serial tick worker, see `_automaton_worker`) this should never actually
   fire, but the module doesn't assume its own concurrency model is bug-free.
10. **STT throttling lives in `WhisperWorker.try_partial`, not here.** This
    module calls it on every tick while a user turn is open and takes
    whatever it returns (a `Task` or `None`) at face value -- no separate
    timer, no "is it time yet" check duplicated here.

Two implementation choices the specs leave open, resolved here and flagged
again in the delivery report:

* **Fast path vs. automaton path.** `_on_audio_frame` (called directly from
  the `websocket.receive()` loop) only ever does cheap, non-blocking work:
  `AudioRing.append`, `VadGate.feed`, and handing the whisper worker a
  partial-transcription tick (itself non-blocking by design, trap #10). The
  automaton step -- which can legitimately block on an HTTP round-trip inside
  `decide()` (DecidingInterject/DecidingBargeIn) -- runs on a SEPARATE task
  (`_automaton_worker`) fed through an `asyncio.Queue`, so a slow LLM decision
  never stalls the socket's read loop and therefore never stalls
  `AudioRing.append` (FR-09: recording must not stop, including "moments when
  the agent is deciding").
* **`speech_left_ms` during streamed synthesis.** `dialogue.qnt`'s
  `draftReady` sets a single fixed `speechLeft` when Formulating -> Speaking
  fires; FR-11 requires sending audio sentence-by-sentence before the full
  answer (and thus its true total duration) is known. This module treats
  `speech_left_ms` as a live "queued-but-not-yet-played" counter: the FIRST
  synthesized sentence triggers the real `draftReady` action (with that
  sentence's duration); every later sentence tops the same counter up
  directly (`_extend_speech_left`) rather than re-invoking a `dialogue.qnt`
  action a second time for the same draft. `voiced_seconds` for the barge-in
  prompt and `delivered_ms` for `DialogueMemory.commit_agent_turn` are both
  derived the same way: `total_synthesized_ms - speech_left_ms` ("what the
  automaton's own real-time countdown says has actually played"), never raw
  bytes-sent, so a burst of buffered-but-unplayed audio never counts as heard.
"""
from __future__ import annotations

import asyncio
import json
import struct
import time
import wave
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from backend.audio.clock import SessionClock
from backend.audio.ring import AudioRing
from backend.audio.vad import Overlap, SileroVadModel, SpeechEnded, SpeechStarted, VadEvent, VadGate
from backend.config import Settings
from backend.dialogue.machine import AutomatonState, DecisionClient, DialogueMachine
from backend.dialogue.memory import DialogueMemory
from backend.dialogue.models import AgentState, DialogueState, DialogueTimers, Draft
from backend.dialogue.nodes import AutomatonInput, DeadlockError, DialogueThresholds
from backend.dialogue.scenarios import ActionKind, Scenario, ScenarioContext, ScenarioRegistry
from backend.llm.client import DecisionInFlightError, LlamaClient, LlamaServerError, Message
from backend.llm.prompts import build_answer_messages, build_intent_prompt
from backend.llm.schemas import IntentDecision
from backend.rag.pipeline import RagPipeline
from backend.stt.whisper_worker import TranscriptionResult, WhisperWorker
from backend.telemetry import get_logger
from backend.tts.silero_worker import SileroWorker, SynthesisResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Wire framing (contracts/websocket.md §2.1/§3.1) -- 4-byte LE uint32 prefix
# + raw PCM16LE mono. No base64 on this path (§1: "жжёт CPU на обоих концах").
# ---------------------------------------------------------------------------

_PREFIX = struct.Struct("<I")


def _pack_audio_frame(prefix: int, pcm16: bytes) -> bytes:
    return _PREFIX.pack(prefix & 0xFFFFFFFF) + pcm16


def _unpack_audio_frame(frame: bytes) -> tuple[int, bytes]:
    if len(frame) < _PREFIX.size:
        raise ValueError(
            f"audio frame shorter than the {_PREFIX.size}-byte offset_ms prefix "
            f"(got {len(frame)} bytes) -- contracts/websocket.md §2.1"
        )
    (prefix,) = _PREFIX.unpack(frame[: _PREFIX.size])
    return prefix, frame[_PREFIX.size :]


# ---------------------------------------------------------------------------
# Fixed Russian strings this module owns. Neither is specified anywhere in
# specs/001-streaming-dialogue/ -- see the delivery report's "недоописано"
# section. Kept short and deliberately unopinionated about admissions-office
# specifics (the real phone number/office details already live in
# `dialogue/scenarios.yaml`'s `substitutions`, not here).
# ---------------------------------------------------------------------------

_GREETING_TEXT = (
    "Здравствуйте! Это голосовой ассистент приёмной комиссии. "
    "Слушаю ваш вопрос."
)

_FAREWELL_TEXT = "Если у вас больше нет вопросов, на этом всё. Хорошего дня!"

_STREAMING_PLACEHOLDER_MS = 60_000
"""Arms `speech_left_ms` on the first sentence of a streamed reply -- see
`DialogueSession._finalize_speech_timer`'s docstring. Large enough that no
realistic RAG+LLM+TTS gap between sentences exhausts it before the next
sentence tops it up or the draft finishes and gets corrected down."""

_SYSTEM_PROMPT = (
    "Ты — голосовой ассистент приёмной комиссии института. Отвечаешь вслух, "
    "поэтому говори разговорным языком, короткими фразами, без списков и "
    "markdown-разметки. Не выдумывай факты: если ответа нет в предоставленном "
    "контексте, честно скажи, что не нашёл, и предложи обратиться в приёмную "
    "комиссию напрямую. Не повторяй вопрос собеседника перед ответом."
)


# ---------------------------------------------------------------------------
# Greeting asset (FR-01) -- also the whisper warmup source (plan.md §2:
# "тишина прогревом не является", `warmup_audio_pcm16`).
# ---------------------------------------------------------------------------


def read_wav_pcm16(path: Path) -> tuple[bytes, int]:
    """Returns `(pcm16_mono_bytes, sample_rate)` for a 16-bit mono WAV file."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {wav_file.getsampwidth() * 8}-bit")
        if wav_file.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {wav_file.getnchannels()} channels")
        pcm = wav_file.readframes(wav_file.getnframes())
        return pcm, wav_file.getframerate()


def _write_wav_pcm16(path: Path, pcm16: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)


async def ensure_greeting_audio(
    path: Path, tts_worker: SileroWorker, *, sample_rate: int
) -> None:
    """Renders `_GREETING_TEXT` to `path` via `tts_worker` if it doesn't exist
    yet. Idempotent -- safe to call on every process start. `tts_worker` must
    already be started (`SileroWorker.start()` awaited) since this calls
    `synthesize()`.

    Rendered at `sample_rate` (== `AUDIO_SAMPLE_RATE`, 16000), not
    `TTS_SAMPLE_RATE` (48000): this same file doubles as `WhisperWorker`'s
    `warmup_audio_pcm16`, which whisper expects at its own 16 kHz operating
    rate with no resampling step. A single 16 kHz greeting file serves both
    FR-01 playback and the warmup requirement without needing a resampler in
    the hot path for something this short.
    """
    if path.exists():
        return
    result = await tts_worker.synthesize(_GREETING_TEXT)
    pcm16 = result.pcm16
    if result.sample_rate != sample_rate:
        import numpy as np
        import soxr

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        resampled = soxr.resample(audio, result.sample_rate, sample_rate).astype(np.float32)
        pcm16 = (resampled * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
    _write_wav_pcm16(path, pcm16, sample_rate)
    await logger.ainfo("greeting_audio_generated", path=str(path), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Decision-call recorder -- wraps whatever `DecisionClient` this process was
# given (a real `LlamaClient` in production, `MOCK_LLM`'s stand-in, or a test
# fake) so `DialogueMachine`'s two decision nodes keep working completely
# unmodified (llm.md §1's `decide()` shape), while this module can still read
# back `reason`/`understood`/`interrupt` for `ScenarioContext` and for the
# `decisions` telemetry block (contracts/websocket.md §3.6) -- neither of
# which `DialogueMachine.step()` exposes on its own return value.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RecordedDecision:
    kind: str  # "interject" | "barge_in"
    at_ms: int
    result: bool
    reason: str
    latency_ms: float
    payload: dict[str, Any]


@dataclass
class _RecordingDecisionClient:
    inner: DecisionClient
    clock: SessionClock
    history: list[RecordedDecision] = field(default_factory=list)

    async def decide(
        self, prompt: str, schema: dict[str, Any], *, max_tokens: int = 256
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = await self.inner.decide(prompt, schema, max_tokens=max_tokens)
        latency_ms = (time.perf_counter() - started) * 1000
        if "interject" in schema["properties"]:
            self.history.append(
                RecordedDecision(
                    kind="interject",
                    at_ms=self.clock.now_ms(),
                    result=bool(result["interject"]),
                    reason=str(result.get("reason", "")),
                    latency_ms=latency_ms,
                    payload=result,
                )
            )
        elif "interrupt" in schema["properties"]:
            self.history.append(
                RecordedDecision(
                    kind="barge_in",
                    at_ms=self.clock.now_ms(),
                    result=bool(result["interrupt"]),
                    reason=str(result.get("reason", "")),
                    latency_ms=latency_ms,
                    payload=result,
                )
            )
        return result

    def pop_last(self, kind: str) -> RecordedDecision | None:
        for item in reversed(self.history):
            if item.kind == kind:
                return item
        return None


# ---------------------------------------------------------------------------
# Process-level singletons, assembled once by `backend/app.py`'s lifespan and
# handed to every `DialogueSession`. Everything here is either stateless or
# owns its own internal single-worker pool (plan.md §2/§9) -- nothing in this
# dataclass is mutated per-session; `DialogueSession` builds its own
# per-connection state (ring, VAD, memory, automaton) around it.
# ---------------------------------------------------------------------------


@dataclass
class SessionDependencies:
    settings: Settings
    llama_client: LlamaClient
    whisper_worker: WhisperWorker
    tts_worker: SileroWorker
    rag_pipeline: RagPipeline
    rag_executor: Executor
    scenario_registry: ScenarioRegistry
    thresholds: DialogueThresholds
    greeting_pcm16: bytes
    greeting_duration_ms: int


# ---------------------------------------------------------------------------
# Per-tick unit of work handed from the socket's fast read path to the serial
# automaton worker (see module docstring, "Fast path vs. automaton path").
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _Tick:
    user_speaking: bool
    elapsed_ms: int
    turn_ended: bool = False


def _rag_context_text(chunks: tuple[dict[str, object], ...]) -> str:
    if not chunks:
        return ""
    lines = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        source = chunk.get("source")
        if source:
            lines.append(f"[{source}] {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def _diff_suffix(previous: str, current: str) -> str:
    """Whisper re-decodes the whole open window every partial (plan.md §3.3
    -- there is no native streaming API), so consecutive partials are USUALLY
    `current == previous + <new words>`. `DialogueMemory.append_transcript`
    is tail-only (memory.md §6/§7: rewriting the middle would break llama-
    server's `cache_prompt` prefix match), so the common case just appends
    the suffix. When the window slides (`STT_WINDOW_SECONDS` exceeded,
    `WhisperWorker.window_start_ms`) or the decoder revises earlier words,
    `current` is no longer a superset of `previous`; there is no way to
    "unwrite" what's already in an append-only buffer, so this falls back to
    appending the whole new partial as a distinct fragment rather than
    silently dropping or duplicating words. This is an honest limitation,
    not a bug fix -- see the delivery report.
    """
    if current.startswith(previous):
        return current[len(previous) :]
    if not previous:
        return current
    return (" " if previous and not previous.endswith(" ") else "") + current


class DialogueSession:
    """One `/ws/dialogue` connection. Constructed fresh per accepted socket;
    `run()` drives it until disconnect or `session.ended`.
    """

    def __init__(self, *, websocket: WebSocket, session_id: str, deps: SessionDependencies) -> None:
        self._ws = websocket
        self._session_id = session_id
        self._deps = deps
        self._settings = deps.settings

        self._clock = SessionClock()
        self._ring = AudioRing(
            sample_rate=self._settings.audio.audio_sample_rate,
            capacity_seconds=self._settings.audio.audio_ring_seconds,
        )
        self._vad_model = SileroVadModel(sample_rate=self._settings.audio.audio_sample_rate)
        self._vad = VadGate(
            model=self._vad_model,
            sample_rate=self._settings.audio.audio_sample_rate,
            threshold=self._settings.audio.vad_threshold,
            preroll_ms=self._settings.audio.vad_preroll_ms,
            silence_end_ms=self._settings.audio.vad_silence_end_ms,
            overlap_threshold_ms=int(self._settings.dialogue.dialogue_barge_in_overlap_s * 1000),
        )
        self._memory = DialogueMemory(max_transcript_chars=self._settings.dialogue.transcript_buffer_chars)
        self._decision_recorder = _RecordingDecisionClient(inner=deps.llama_client, clock=self._clock)
        self._machine = DialogueMachine(decision_client=self._decision_recorder, thresholds=deps.thresholds)
        self._automaton = AutomatonState(
            dialogue=DialogueState(
                agent=AgentState.GREETING,
                draft=Draft.NO_DRAFT,
                timers=DialogueTimers(speech_left_ms=deps.greeting_duration_ms),
            )
        )

        self._user_speaking = False
        self._speech_start_ms: int | None = None
        # Set alongside `_speech_start_ms`, on the SAME audio-offset_ms axis
        # (memory.md §1's "единая ось времени") -- `_handle_turn_ended` uses
        # this for `run_final()`'s `end_ms` and `DialogueMemory.add_user_turn`
        # instead of `SessionClock.now_ms()`, which is real wall-clock time
        # and NOT guaranteed to track the client's self-reported `offset_ms`
        # axis closely enough for ring lookups (found while wiring T-09: a
        # naive `SessionClock.now_ms()` end_ms produced end_ms < start_ms
        # whenever server processing ran faster than the audio it was fed,
        # collapsing `AudioRing.snapshot()` to zero samples -- see report).
        self._speech_end_ms: int | None = None
        self._last_offset_ms: int | None = None
        self._last_partial_text = ""
        self._text_turn_pending = False

        self._tick_queue: asyncio.Queue[_Tick] = asyncio.Queue()
        # Two independent tasks can both want to advance the automaton: the
        # serial VAD-tick worker (`_automaton_worker`) and the answer task
        # (`_answer_task`, draftReady/speech-left top-ups as sentences are
        # synthesized). Both go through `_run_automaton_step`/
        # `_extend_speech_left`, and both hold this lock while touching
        # `self._automaton` -- without it the two tasks could interleave a
        # read-modify-write on the same `DialogueTimers` and lose an update.
        self._automaton_lock = asyncio.Lock()
        self._automaton_worker_task: asyncio.Task[None] | None = None
        self._pending_stt_task: asyncio.Task[TranscriptionResult] | None = None
        self._answer_task: asyncio.Task[None] | None = None
        self._agent_audio_seq = 0
        self._mute_tts = False
        self._closed = False
        self._send_lock = asyncio.Lock()

        # Current draft bookkeeping (FR-12/FR-16, module docstring's
        # "speech_left_ms during streamed synthesis").
        self._draft_text = ""
        self._draft_start_ms: int | None = None
        self._draft_total_synthesized_ms = 0
        self._draft_timer_credited_ms = 0
        self._draft_scenario: str | None = None
        self._draft_history_snapshot = ""
        self._pending_farewell: SynthesisResult | None = None

    # -- entry point ---------------------------------------------------

    async def run(self) -> None:
        await self._ws.accept()
        await self._send_json(
            {
                "type": "session.ready",
                "session_id": self._session_id,
                "t0_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        await logger.ainfo("ws_session_ready", session_id=self._session_id)

        self._automaton_worker_task = asyncio.create_task(self._automaton_worker())
        await self._begin_greeting()

        try:
            while True:
                message = await self._ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                raw_bytes = message.get("bytes")
                raw_text = message.get("text")
                if raw_bytes is not None:
                    await self._on_audio_frame(raw_bytes)
                elif raw_text is not None:
                    await self._on_control_message(raw_text)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001 -- last-resort session boundary, logged then torn down
            await logger.aexception("ws_session_crashed", session_id=self._session_id)
        finally:
            await self._teardown()

    async def _teardown(self) -> None:
        self._closed = True
        tasks = [
            t
            for t in (self._answer_task, self._automaton_worker_task, self._pending_stt_task)
            if t is not None
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            # `.cancel()` alone only REQUESTS cancellation -- it does not
            # wait for it. A task blocked inside `loop.run_in_executor()`
            # (whisper/silero's single-worker pools) can't be preempted
            # once the underlying thread has actually started running the
            # job (`concurrent.futures.Future.cancel()` returns False for a
            # running job); without awaiting here, this method would return
            # -- and a caller sharing these same worker pools across
            # sessions (this project's own acceptance-test harness, and
            # conceivably a fast reconnect in production) could start
            # competing with a thread that's still finishing the PREVIOUS
            # session's last decode/synthesis call. Awaiting with
            # `return_exceptions=True` drains that properly (CancelledError
            # is expected and discarded); this is what actually makes
            # `_teardown()` mean "this session's background work is done",
            # not just "asked to stop".
            await asyncio.gather(*tasks, return_exceptions=True)
        await logger.ainfo("ws_session_closed", session_id=self._session_id)

    # -- outbound framing / sends ---------------------------------------

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._closed or self._ws.client_state != WebSocketState.CONNECTED:
            return
        async with self._send_lock:
            await self._ws.send_json(payload)

    async def _send_audio(self, pcm16: bytes) -> None:
        if not pcm16 or self._closed or self._ws.client_state != WebSocketState.CONNECTED:
            return
        chunk_bytes = max(
            2,
            self._settings.audio.audio_chunk_size_ms
            * self._settings.audio.audio_sample_rate
            // 1000
            * 2,
        )
        async with self._send_lock:
            for offset in range(0, len(pcm16), chunk_bytes):
                self._agent_audio_seq += 1
                frame = _pack_audio_frame(self._agent_audio_seq, pcm16[offset : offset + chunk_bytes])
                await self._ws.send_bytes(frame)

    async def _flush_audio(self, reason: str) -> None:
        await self._send_json({"type": "audio.flush", "reason": reason})

    # -- greeting (FR-01/FR-02/FR-03) ------------------------------------

    async def _begin_greeting(self) -> None:
        self._vad.notify_agent_speaking(True)
        if not self._mute_tts:
            await self._send_audio(self._deps.greeting_pcm16)

    # -- fast path: raw audio frames --------------------------------------

    async def _on_audio_frame(self, frame: bytes) -> None:
        offset_ms, pcm = _unpack_audio_frame(frame)
        self._ring.append(pcm, offset_ms)

        chunk_ms = self._chunk_duration_ms(pcm)
        elapsed_ms = self._advance_elapsed_ms(offset_ms, chunk_ms)

        event = self._vad.feed(pcm, offset_ms)
        self._apply_vad_event(event, offset_ms)

        if isinstance(event, SpeechStarted):
            # Trap #1: both dialogue.qnt steps enqueued back-to-back, no
            # `await` between them -- see module docstring point 1.
            self._tick_queue.put_nowait(_Tick(user_speaking=True, elapsed_ms=0))
            self._tick_queue.put_nowait(_Tick(user_speaking=True, elapsed_ms=elapsed_ms))
        else:
            turn_ended = isinstance(event, SpeechEnded)
            self._tick_queue.put_nowait(
                _Tick(user_speaking=self._user_speaking, elapsed_ms=elapsed_ms, turn_ended=turn_ended)
            )

        self._maybe_try_partial_stt(offset_ms)

    def _chunk_duration_ms(self, pcm: bytes) -> int:
        samples = len(pcm) // 2
        return samples * 1000 // self._settings.audio.audio_sample_rate

    def _advance_elapsed_ms(self, offset_ms: int, chunk_ms: int) -> int:
        if self._last_offset_ms is None:
            delta = chunk_ms
        else:
            delta = offset_ms - self._last_offset_ms
            if delta <= 0:
                delta = chunk_ms
        self._last_offset_ms = offset_ms
        return delta

    def _apply_vad_event(self, event: VadEvent | None, offset_ms: int) -> None:
        if isinstance(event, SpeechStarted):
            self._user_speaking = True
            self._speech_start_ms = event.speech_start_ms
            self._last_partial_text = ""
        elif isinstance(event, SpeechEnded):
            self._user_speaking = False
            self._speech_end_ms = event.end_ms
        elif isinstance(event, Overlap):
            # Level unchanged (still speaking). dialogue.qnt's own
            # overlapGrows/overlapTriggersDecision (nodes.py), driven by the
            # automaton's own overlap_ms ticking while agent==Speaking, is
            # the mechanism this module relies on for barge-in STATE timing
            # -- see the module docstring's open design note on why two
            # overlap-threshold implementations exist in this codebase.
            # `VadGate`'s own `Overlap` event is logged here for telemetry
            # (it fires once per overlap episode, per FR-17's non-LLM gate)
            # but does not itself drive a `machine.step()` call.
            asyncio.create_task(
                logger.ainfo("vad_overlap_detected", session_id=self._session_id, duration_ms=event.duration_ms, at_ms=offset_ms)
            )

    def _maybe_try_partial_stt(self, now_ms: int) -> None:
        if not self._user_speaking or self._speech_start_ms is None:
            return
        if self._pending_stt_task is not None and not self._pending_stt_task.done():
            return
        task = self._deps.whisper_worker.try_partial(
            self._ring, speech_start_ms=self._speech_start_ms, now_ms=now_ms
        )
        if task is None:
            return
        self._pending_stt_task = task
        task.add_done_callback(self._on_partial_result_done)

    def _on_partial_result_done(self, task: asyncio.Task[TranscriptionResult]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            asyncio.create_task(logger.aerror("stt_partial_failed", error=str(exc)))
            return
        asyncio.create_task(self._on_partial_result(task.result()))

    async def _on_partial_result(self, result: TranscriptionResult) -> None:
        suffix = _diff_suffix(self._last_partial_text, result.text)
        self._last_partial_text = result.text
        if suffix:
            self._memory.append_transcript(suffix, result.window_start_ms, result.window_end_ms)
        await self._send_json(
            {
                "type": "transcript.update",
                "text": result.text,
                "is_final": False,
                "start_ms": result.window_start_ms,
                "end_ms": result.window_end_ms,
                "confidence": None,  # WhisperSegmentProtocol/TranscriptionResult carry
                # no per-utterance confidence today (stt/whisper_worker.py) -- see report.
            }
        )

    # -- control frames (contracts/websocket.md §2.2/§2.3) ---------------

    async def _on_control_message(self, raw_text: str) -> None:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            await self._send_json({"type": "error", "code": "bad_json", "text": "invalid JSON frame", "recoverable": True})
            return
        message_type = payload.get("type")
        if message_type == "user.text":
            await self._on_user_text(str(payload.get("text", "")))
        elif message_type == "session.reset":
            await self._on_session_reset()
        elif message_type == "session.config":
            if "mute_tts" in payload:
                self._mute_tts = bool(payload["mute_tts"])
        else:
            await self._send_json(
                {"type": "error", "code": "unknown_message_type", "text": str(message_type), "recoverable": True}
            )

    async def _on_user_text(self, text: str) -> None:
        """contracts/websocket.md §2.2: parity with the legacy "Отправить как
        текст" button. Treated as an already-finished user turn -- bypasses
        STT entirely, so `_speech_start_ms`/`_speech_end_ms` are synthesized
        from the current clock rather than a real VAD preroll, and
        `_handle_turn_ended` must NOT attempt a `run_final()` STT pass for
        this turn (`_text_turn_pending`) -- there is no corresponding audio
        in the ring, and running whisper over whatever silence happens to be
        there would overwrite the actual typed text with an empty result.
        """
        now_ms = self._clock.now_ms()
        self._speech_start_ms = now_ms
        self._speech_end_ms = now_ms
        self._text_turn_pending = True
        self._last_partial_text = text
        self._user_speaking = False
        self._tick_queue.put_nowait(_Tick(user_speaking=False, elapsed_ms=0, turn_ended=True))

    async def _on_session_reset(self) -> None:
        """FR: "Новая сессия" (contracts/websocket.md §2.3) -- fresh
        conversation on the same socket. `DialogueMemory.reset_transcript()`
        is called ONLY from here (trap #4); `turns` is also cleared since
        `session.reset` is meant to start an entirely new conversation, not
        just truncate the incremental-prefill buffer.
        """
        if self._answer_task is not None:
            self._answer_task.cancel()
            self._answer_task = None
        self._memory.reset_transcript()
        self._memory.turns.clear()
        self._last_partial_text = ""
        self._speech_start_ms = None
        self._speech_end_ms = None
        self._text_turn_pending = False
        self._user_speaking = False
        self._last_offset_ms = None
        self._automaton = AutomatonState(
            dialogue=DialogueState(
                agent=AgentState.GREETING,
                draft=Draft.NO_DRAFT,
                timers=DialogueTimers(speech_left_ms=self._deps.greeting_duration_ms),
            )
        )
        await logger.ainfo("ws_session_reset", session_id=self._session_id)
        await self._begin_greeting()

    # -- automaton worker (serial; may block on decide()) ----------------

    async def _automaton_worker(self) -> None:
        while True:
            tick = await self._tick_queue.get()
            try:
                await self._run_automaton_step(tick)
            except Exception:  # noqa: BLE001 -- one bad tick must not kill the worker
                await logger.aexception("automaton_tick_failed", session_id=self._session_id)

    async def _run_automaton_step(self, tick: _Tick, **extra: Any) -> None:
        # dialogue.qnt's Formulating has NO "nothing happened this tick"
        # no-op action -- unlike every other state (greetingPlays/idleTicks/
        # speechPlays/closingPlays all handle a quiet tick explicitly),
        # CANDIDATES_BY_STATE[Formulating] is exactly (draftAbandoned,
        # draftReady, userStartsSpeaking), none of which guard on "nothing
        # new happened" (nodes.py). Found while wiring T-09, not covered by
        # any existing test (test_state_machine.py's Formulating cases only
        # ever step with user_speaking=True or draft_ready=True): stepping
        # unconditionally on every incoming audio chunk -- this module's
        # general policy for every other state -- deadlocks
        # (`DeadlockError`) the instant a quiet chunk arrives while
        # Formulating. `AudioRing.append`/`VadGate.feed` already ran in
        # `_on_audio_frame` regardless (FR-09: recording never stops); this
        # only skips handing an uneventful tick to `DialogueMachine.step()`,
        # which dialogue.qnt itself never expected to receive one for this
        # state. See the delivery report.
        if (
            self._automaton.dialogue.agent is AgentState.FORMULATING
            and not tick.user_speaking
            and "draft_ready" not in extra
        ):
            return

        # idleHangup's dialogue.qnt effect needs `farewell_duration_ms` in
        # hand BEFORE the transition fires (it sets speech_left_ms from it
        # atomically) -- predict the crossing from the same pre-step state
        # the guard itself will check, and synthesize ahead of time.
        if "farewell_duration_ms" not in extra:
            farewell_ms = await self._synthesize_farewell_if_due(tick.elapsed_ms, tick.user_speaking)
            if farewell_ms is not None:
                extra["farewell_duration_ms"] = farewell_ms

        async with self._automaton_lock:
            previous_agent = self._automaton.dialogue.agent
            event = AutomatonInput(
                user_speaking=tick.user_speaking,
                elapsed_ms=tick.elapsed_ms,
                turn_ended=tick.turn_ended,
                dialogue_history=self._memory.serialize_history(
                    max_turns=self._settings.dialogue.dialogue_history_max_turns
                ),
                transcript_so_far=self._last_partial_text,
                interlocutor_transcript=self._last_partial_text,
                draft_answer=self._draft_text,
                # `_draft_timer_credited_ms`, NOT `_draft_total_synthesized_ms`:
                # `machine.py`'s DecidingBargeIn node computes voiced_seconds
                # as `(total_answer_ms - speech_left_ms) / 1000`, and
                # `speech_left_ms` was armed from the credited value (the
                # streaming placeholder while still synthesizing, the real
                # total once `_finalize_speech_timer` corrects it) -- mixing
                # in the separately-tracked synthesized-bytes total here
                # would make that subtraction meaningless while a draft is
                # still streaming. See `_finalize_speech_timer`'s docstring.
                total_answer_ms=self._draft_timer_credited_ms or None,
                **extra,
            )
            try:
                self._automaton = await self._machine.step(self._automaton, event)
            except DecisionInFlightError:
                # Trap #9: a second concurrent decide() is refused, not
                # queued. By construction (this lock serializes every
                # step()) this should not happen; if it somehow does, this
                # tick's transition simply doesn't fire -- never a crash.
                await logger.awarning("decide_in_flight_skipped", session_id=self._session_id)
                return
            except DeadlockError:
                await logger.aexception(
                    "dialogue_deadlock", session_id=self._session_id, agent=previous_agent.value
                )
                raise
            new_agent = self._automaton.dialogue.agent
        if new_agent is not previous_agent:
            await self._on_transition(previous_agent, new_agent, event)

    async def _step(self, *, user_speaking: bool, elapsed_ms: int, turn_ended: bool = False, **extra: Any) -> None:
        """Direct entry point for ticks this module raises itself (draft-
        ready, farewell) rather than ones queued from `_on_audio_frame` --
        always invoked from the single automaton-worker task, never from the
        socket's fast read path.
        """
        await self._run_automaton_step(_Tick(user_speaking=user_speaking, elapsed_ms=elapsed_ms, turn_ended=turn_ended), **extra)

    # -- state-transition reactions --------------------------------------

    async def _on_transition(self, previous: AgentState, current: AgentState, event: AutomatonInput) -> None:
        await self._send_json({"type": "state", "agent": current.value, "prev": previous.value, "at_ms": self._clock.now_ms()})
        if previous is AgentState.GREETING and current is AgentState.LISTENING:
            await self._on_greeting_ended(interrupted=event.user_speaking)
        elif previous is AgentState.LISTENING and current is AgentState.FORMULATING:
            self._answer_task = asyncio.create_task(self._handle_turn_ended())
        elif previous is AgentState.DECIDING_INTERJECT and current is AgentState.FORMULATING:
            self._answer_task = asyncio.create_task(self._handle_interject_accepted())
        elif previous is AgentState.FORMULATING and current is AgentState.LISTENING:
            await self._on_draft_abandoned()
        elif previous is AgentState.SPEAKING and current is AgentState.LISTENING:
            await self._on_speech_completed()
        elif previous is AgentState.DECIDING_BARGE_IN and current is AgentState.LISTENING:
            await self._on_barge_in_accepted()
        elif previous is AgentState.DECIDING_BARGE_IN and current is AgentState.SPEAKING:
            await self._on_barge_in_declined()
        elif previous is AgentState.LISTENING and current is AgentState.CLOSING:
            await self._on_closing_started()
        elif previous is AgentState.CLOSING and current is AgentState.LISTENING:
            await self._on_closing_interrupted()
        elif previous is AgentState.CLOSING and current is AgentState.ENDED:
            await self._on_session_ended()

    async def _on_greeting_ended(self, *, interrupted: bool) -> None:
        self._vad.notify_agent_speaking(False)
        if interrupted:
            await self._flush_audio(reason="greeting_cut")
            await logger.ainfo("greeting_cut", session_id=self._session_id, at_ms=self._clock.now_ms())
        else:
            await logger.ainfo("greeting_finished", session_id=self._session_id)

    # -- Listening -> Formulating: intent, scenario, RAG, answer ----------

    async def _handle_turn_ended(self) -> None:
        text = self._last_partial_text
        start_ms = self._speech_start_ms if self._speech_start_ms is not None else self._clock.now_ms()
        # Same offset_ms axis as `start_ms` -- see `_speech_end_ms`'s
        # docstring at its declaration for why `SessionClock.now_ms()` here
        # was a bug (it is real wall-clock time, unrelated to the audio
        # ring's own index space that `run_final()` reads from).
        end_ms = self._speech_end_ms if self._speech_end_ms is not None else self._last_offset_ms
        if end_ms is None or end_ms < start_ms:
            end_ms = self._clock.now_ms()
        # Consult the WORKER's own configured behaviour
        # (`final_pass_enabled`, set from `STT_FINAL_PASS` at construction),
        # not `self._settings` again -- `WhisperWorker` is the source of
        # truth for whether it will accept a `run_final()` call at all.
        # Also: attempt the final pass whenever it's enabled, regardless of
        # whether the last partial happens to be empty -- an empty partial
        # (the STT_PARTIAL_MAX_HZ throttle never got a completed decode in
        # before turn-end, e.g. a very short turn) is exactly the case the
        # final pass exists to correct, not a reason to skip it. Text-origin
        # turns (`user.text`, contracts/websocket.md §2.2) are the one
        # exception -- there is no corresponding audio in the ring at all,
        # so running whisper there would silently replace the typed text
        # with an empty transcription.
        text_turn = self._text_turn_pending
        self._text_turn_pending = False
        if self._deps.whisper_worker.final_pass_enabled and not text_turn:
            try:
                final = await self._deps.whisper_worker.run_final(
                    self._ring, speech_start_ms=start_ms, end_ms=end_ms
                )
                text = final.text
            except Exception:  # noqa: BLE001 -- fall back to the last partial rather than losing the turn
                await logger.aexception("stt_final_pass_failed", session_id=self._session_id)
        self._memory.add_user_turn(text=text, start_ms=start_ms, end_ms=end_ms)

        history = self._memory.serialize_history(max_turns=self._settings.dialogue.dialogue_history_max_turns)

        if not text.strip():
            scenario = self._deps.scenario_registry.match(AgentState.LISTENING, ScenarioContext(transcript=text))
            ctx = ScenarioContext(transcript=text)
        else:
            try:
                decision_dict = await self._deps.llama_client.decide(
                    build_intent_prompt(dialogue_history=history, transcript=text),
                    IntentDecision.SCHEMA,
                )
                decision = IntentDecision.model_validate(decision_dict)
                ctx = ScenarioContext(
                    intent=decision.intent,
                    confidence=decision.confidence,
                    intent_query=decision.query or text,
                    transcript=text,
                )
            except (LlamaServerError, DecisionInFlightError) as exc:
                await logger.aerror("intent_decision_failed", session_id=self._session_id, error=str(exc))
                ctx = ScenarioContext(error_type="server_error", transcript=text)
            scenario = self._deps.scenario_registry.match(AgentState.LISTENING, ctx)

        if scenario is None:
            await logger.awarning("no_scenario_matched", session_id=self._session_id, state="Listening")
            await self._speak_fixed_text("Простите, не смог разобрать вопрос. Повторите, пожалуйста.")
            await self._finish_draft_after_speaking(is_via_scenario=None)
            return

        self._draft_scenario = scenario.name
        self._draft_history_snapshot = history
        await self._run_scenario(scenario, ctx)

    async def _handle_interject_accepted(self) -> None:
        """FR-07 accepted: DecidingInterject -> Formulating already happened
        inside `machine.step()`; this module supplies the CONTENT (what to
        say) via the matching `dialogue/scenarios.yaml` entry -- the
        automaton only owns the state bookkeeping (module docstring's
        "content vs. state" split, plan.md §7).
        """
        decision = self._decision_recorder.pop_last("interject")
        ctx = ScenarioContext(interject_accepted=decision.result if decision else True, transcript=self._last_partial_text)
        scenario = self._deps.scenario_registry.match(AgentState.DECIDING_INTERJECT, ctx)
        if scenario is None:
            await logger.awarning("no_scenario_matched", session_id=self._session_id, state="DecidingInterject")
            return
        self._draft_scenario = scenario.name
        self._draft_history_snapshot = self._memory.serialize_history(
            max_turns=self._settings.dialogue.dialogue_history_max_turns
        )
        await self._run_scenario(scenario, ctx)

    async def _run_scenario(self, scenario: Scenario, ctx: ScenarioContext) -> None:
        registry = self._deps.scenario_registry
        rag_context = ""
        for action in scenario.actions:
            if self._closed:
                return
            if action.kind is ActionKind.NOOP:
                continue
            if action.kind is ActionKind.SAY:
                text = registry.render(action.text or "", ctx)
                await self._speak_fixed_text(text)
            elif action.kind is ActionKind.RAG_QUERY:
                query = registry.render(action.query or "", ctx)
                chunks, timings = await self._deps.rag_pipeline.asearch(query, executor=self._deps.rag_executor)
                rag_context = _rag_context_text(tuple(chunks))
                max_score = max((float(c.get("rerank_score", 0.0)) for c in chunks), default=0.0)
                ctx = replace(ctx, rag_count=len(chunks), rag_max_score=max_score, rag_answer=rag_context)
                await logger.ainfo(
                    "rag_query", session_id=self._session_id, query=query, count=len(chunks), timings_ms=timings
                )
                override = registry.match(AgentState.FORMULATING, ctx)
                if override is not None and override.name != scenario.name:
                    await self._run_scenario(override, ctx)
                    return
            elif action.kind is ActionKind.SAY_GENERATED:
                await self._speak_generated(action.instruction or "", use_rag=bool(action.use_rag), rag_context=rag_context)
            elif action.kind is ActionKind.COMMIT_PARTIAL:
                await self._commit_partial_draft()
            elif action.kind is ActionKind.RESUME_SPEAKING:
                await self._resume_speaking()

        await self._finish_draft_after_speaking(is_via_scenario=scenario.name)

    # -- speaking: fixed text / generated streaming -----------------------

    async def _speak_fixed_text(self, text: str) -> None:
        if not text.strip():
            return
        synthesis = await self._deps.tts_worker.synthesize(text)
        self._draft_text = (self._draft_text + " " + text).strip() if self._draft_text else text
        await self._start_or_extend_speaking(synthesis)

    async def _speak_generated(self, instruction: str, *, use_rag: bool, rag_context: str) -> None:
        history = self._draft_history_snapshot or self._memory.serialize_history(
            max_turns=self._settings.dialogue.dialogue_history_max_turns
        )
        messages: list[Message] = build_answer_messages(
            system_prompt=f"{_SYSTEM_PROMPT}\n\n{instruction.strip()}",
            rag_context=rag_context if use_rag else "",
            dialogue_history=history,
            transcript=self._last_partial_text,
        )
        try:
            async for sentence in self._deps.llama_client.stream_answer(
                messages,
                max_tokens=self._settings.llm.llm_max_tokens,
                temperature=self._settings.llm.llm_temperature,
            ):
                if self._closed:
                    return
                if self._automaton.dialogue.draft is Draft.DROPPED:
                    # FR-12: the user resumed speaking before any audio for
                    # this draft went out; `_on_draft_abandoned` already
                    # fired. Stop generating -- nothing further may be
                    # committed (dialogue.qnt's inv_dropped_never_committed).
                    return
                synthesis = await self._deps.tts_worker.synthesize(sentence)
                self._draft_text = (self._draft_text + " " + sentence).strip() if self._draft_text else sentence
                await self._start_or_extend_speaking(synthesis)
        except LlamaServerError as exc:
            await logger.aerror("stream_answer_failed", session_id=self._session_id, error=str(exc))
            if not self._draft_text:
                await self._speak_fixed_text("Простите, у меня сейчас сбой. Повторите, пожалуйста, вопрос.")

    async def _start_or_extend_speaking(self, synthesis: SynthesisResult) -> None:
        if self._automaton.dialogue.agent not in (AgentState.FORMULATING, AgentState.SPEAKING):
            # The draft that produced this chunk was abandoned or the
            # session reset while synthesis was in flight (`_on_draft_
            # abandoned`/`_on_session_reset` cancel `_answer_task`, but a
            # single `await tts_worker.synthesize()` already past its own
            # cancellation point can still return one last result) -- drop
            # it instead of sending audio for a reply nobody is voicing
            # anymore or driving a `draftReady` transition from the wrong
            # state.
            return
        chunk_ms = int(synthesis.audio_seconds * 1000)
        if self._automaton.dialogue.agent is AgentState.FORMULATING:
            self._vad.notify_agent_speaking(True)
            self._draft_start_ms = self._clock.now_ms()
            self._draft_total_synthesized_ms = chunk_ms
            # `_STREAMING_PLACEHOLDER_MS`, not `chunk_ms` -- see the
            # docstring on `_finalize_speech_timer` for why a scenario with
            # more than one voiced action (say -> rag_query -> say_generated,
            # e.g. `answer_question`) cannot safely arm `speech_left_ms` with
            # only the FIRST chunk's duration.
            self._draft_timer_credited_ms = _STREAMING_PLACEHOLDER_MS
            await self._step(
                user_speaking=self._user_speaking,
                elapsed_ms=0,
                draft_ready=True,
                answer_duration_ms=_STREAMING_PLACEHOLDER_MS,
            )
        else:
            await self._extend_speech_left(chunk_ms)
            self._draft_timer_credited_ms += chunk_ms
            self._draft_total_synthesized_ms += chunk_ms
        if not self._mute_tts:
            await self._send_audio(synthesis.pcm16)

    async def _extend_speech_left(self, extra_ms: int) -> None:
        """Tops up the live "queued but not yet played" counter as later
        streamed sentences arrive -- see module docstring's open design
        note. Deliberately NOT routed through a `dialogue.qnt` action:
        `draftReady` fires exactly once per draft (Formulating -> Speaking);
        this only ever runs while already in `Speaking`, extending the same
        countdown `speechPlays`/`overlapGrows` are decrementing on the
        automaton worker's own ticks. `AutomatonState`/`DialogueState` are
        plain dataclasses this module already owns a reference to; every
        actual STATE TRANSITION still goes exclusively through
        `DialogueMachine.step()`. Guarded by the same lock as
        `_run_automaton_step` so this read-modify-write can't interleave
        with the worker's own `speechPlays`/`overlapGrows` decrement.
        """
        async with self._automaton_lock:
            dialogue = self._automaton.dialogue
            if dialogue.agent is not AgentState.SPEAKING:
                return
            dialogue.timers.speech_left_ms += extra_ms

    async def _finish_draft_after_speaking(self, *, is_via_scenario: str | None) -> None:
        """Called once a scenario's action list is exhausted -- i.e. no more
        sentences are coming for this draft. If nothing was ever voiced (a
        pure `noop`/state-only scenario, e.g. `interject_keep_listening`),
        there is no draft to commit or drop -- `commit_agent_turn` requires
        `delivered_ms > 0` by design (memory.md §5) and this module must not
        call it with zero.
        """
        if not self._draft_text:
            self._reset_draft()
            return
        await self._finalize_speech_timer()
        # The automaton's own speechCompletes (fired from the per-tick
        # countdown, now carrying the corrected remaining duration) is what
        # actually returns Speaking to Listening, and is where
        # `_on_speech_completed` performs the commit -- nothing further to
        # do here.

    async def _finalize_speech_timer(self) -> None:
        """Corrects `speech_left_ms` from the streaming placeholder down to
        the TRUE remaining duration, now that every sentence for this draft
        has actually been synthesized and `_draft_total_synthesized_ms` is
        final.

        Why a placeholder was needed at all: FR-11 requires sending audio
        after the FIRST completed sentence, before the total answer length
        is known. dialogue.qnt's `draftReady` sets a single fixed
        `speechLeft` on the Formulating -> Speaking edge, and every later
        `speechPlays` tick (driven by the caller's own incoming audio, at
        whatever real cadence that arrives) decrements it. A multi-action
        scenario (`answer_question`: `say` "Секунду, посмотрю." -> RAG
        lookup -> `say_generated`'s LLM stream) needs real seconds between
        the first chunk landing and the next one being ready -- if
        `speech_left_ms` had been armed with only the first chunk's short
        duration, it could hit zero (`speechCompletes`) before the RAG/LLM
        work produces anything more, ending the turn on "Секунду, посмотрю."
        alone (found while building this module's acceptance tests: exactly
        this happened, at 5x realtime in the test harness's tick pacing, but
        the same failure mode is reachable in production any time synthesis
        can't keep up with real-time playback of what's already been sent).
        `_STREAMING_PLACEHOLDER_MS` sidesteps that by being large enough
        that no realistic combination of RAG+LLM+TTS latency reaches it;
        this method is the other half, correcting the timer back down once
        the true total is known so completion isn't delayed by the
        placeholder either.

        `_draft_timer_credited_ms` tracks everything armed into the timer
        for this draft (the placeholder, plus every `_extend_speech_left`
        top-up); comparing it against the timer's CURRENT value gives
        exactly how much has actually played so far, on the same "real
        elapsed ticks" basis nodes.py's `speechPlays` itself uses.
        """
        async with self._automaton_lock:
            dialogue = self._automaton.dialogue
            if dialogue.agent is not AgentState.SPEAKING:
                return
            already_played_ms = max(0, self._draft_timer_credited_ms - dialogue.timers.speech_left_ms)
            true_remaining_ms = max(0, self._draft_total_synthesized_ms - already_played_ms)
            dialogue.timers.speech_left_ms = true_remaining_ms

    def _reset_draft(self) -> None:
        self._draft_text = ""
        self._draft_start_ms = None
        self._draft_total_synthesized_ms = 0
        self._draft_timer_credited_ms = 0
        self._draft_scenario = None
        self._draft_history_snapshot = ""

    # -- Formulating -> Listening without audio (FR-12) -------------------

    async def _on_draft_abandoned(self) -> None:
        # FR-12: the draft itself is discarded, but whatever produced it
        # (`_handle_turn_ended`/`_handle_interject_accepted`'s background
        # `_answer_task` -- run_final/decide/RAG/stream_answer) is still
        # running unless explicitly stopped here. Left uncancelled, that
        # task can complete AFTER the automaton has already moved back to
        # Listening: `_start_or_extend_speaking` would then see `agent is
        # not FORMULATING`, silently skip the draftReady transition, but
        # still call `_send_audio()` for content nobody asked for anymore --
        # and the stray `decide()`/`stream_answer()` calls inside it
        # continue competing for the same GPU/llama-server slot the NEXT
        # turn needs (found while building this module's acceptance tests:
        # an abandoned turn's stale background work measurably slowed down
        # the very next real request). `_on_session_reset` already cancels
        # `_answer_task` for the same reason; this is the other case where a
        # draft dies without ever reaching Speaking.
        if self._answer_task is not None:
            self._answer_task.cancel()
            self._answer_task = None
        await logger.ainfo(
            "draft_abandoned", session_id=self._session_id, scenario=self._draft_scenario, text=self._draft_text
        )
        await self._send_json({"type": "answer.done", "text": "", "voiced_fraction": 0.0, "is_partial": True})
        self._reset_draft()

    # -- Speaking -> Listening: full completion (FR-16 case 2) -------------

    async def _on_speech_completed(self) -> None:
        self._vad.notify_agent_speaking(False)
        if self._draft_text and self._draft_start_ms is not None:
            self._memory.commit_agent_turn(
                text=self._draft_text,
                start_ms=self._draft_start_ms,
                delivered_ms=self._draft_total_synthesized_ms,
                planned_ms=self._draft_total_synthesized_ms,
            )
            await self._send_json(
                {"type": "answer.done", "text": self._draft_text, "voiced_fraction": 1.0, "is_partial": False}
            )
        self._reset_draft()

    # -- barge-in resolution (FR-13/FR-16) --------------------------------

    def _voiced_ms(self) -> int:
        # `_draft_timer_credited_ms`, not `_draft_total_synthesized_ms` --
        # see the comment on `total_answer_ms` in `_run_automaton_step` and
        # `_finalize_speech_timer`'s docstring: this difference is exactly
        # "real elapsed ticks since Speaking began", valid whether or not
        # the placeholder is still active, unlike subtracting the
        # synthesized-bytes total (which is unrelated to what the timer was
        # actually armed with while streaming).
        credited = max(self._draft_timer_credited_ms, self._draft_total_synthesized_ms)
        voiced = credited - self._automaton.dialogue.timers.speech_left_ms
        return max(0, min(voiced, self._draft_total_synthesized_ms))

    async def _commit_partial_draft(self) -> None:
        if self._draft_text and self._draft_start_ms is not None:
            delivered_ms = max(1, self._voiced_ms())
            self._memory.commit_agent_turn(
                text=self._draft_text,
                start_ms=self._draft_start_ms,
                delivered_ms=delivered_ms,
                planned_ms=self._draft_total_synthesized_ms,
            )
            voiced_fraction = min(1.0, delivered_ms / self._draft_total_synthesized_ms) if self._draft_total_synthesized_ms else 1.0
            await self._send_json(
                {"type": "answer.done", "text": self._draft_text, "voiced_fraction": voiced_fraction, "is_partial": True}
            )

    async def _on_barge_in_accepted(self) -> None:
        self._vad.notify_agent_speaking(False)
        await self._flush_audio(reason="barge_in")
        # `commit_partial` is executed by the scenario's own action list
        # (`bargein_yield`, dialogue/scenarios.yaml) via `_run_scenario` --
        # nothing further to do here beyond clearing local draft state once
        # that scenario has run its actions.
        decision = self._decision_recorder.pop_last("barge_in")
        ctx = ScenarioContext(bargein_accepted=True, transcript=self._last_partial_text)
        scenario = self._deps.scenario_registry.match(AgentState.DECIDING_BARGE_IN, ctx)
        if scenario is not None:
            await self._run_scenario_actions_only(scenario, ctx)
        self._reset_draft()
        await logger.ainfo(
            "barge_in_accepted", session_id=self._session_id, reason=(decision.reason if decision else "")
        )

    async def _on_barge_in_declined(self) -> None:
        decision = self._decision_recorder.pop_last("barge_in")
        ctx = ScenarioContext(bargein_accepted=False, transcript=self._last_partial_text)
        scenario = self._deps.scenario_registry.match(AgentState.DECIDING_BARGE_IN, ctx)
        if scenario is not None:
            await self._run_scenario_actions_only(scenario, ctx)
        await logger.ainfo(
            "barge_in_declined", session_id=self._session_id, reason=(decision.reason if decision else "")
        )

    async def _run_scenario_actions_only(self, scenario: Scenario, ctx: ScenarioContext) -> None:
        """Like `_run_scenario` but without the trailing
        `_finish_draft_after_speaking` call -- barge-in resolution scenarios
        (`bargein_yield`/`bargein_continue`) only ever contain
        `commit_partial`/`resume_speaking`/`noop`, none of which start a new
        draft, so there is nothing to finish.
        """
        for action in scenario.actions:
            if action.kind is ActionKind.COMMIT_PARTIAL:
                await self._commit_partial_draft()
            elif action.kind is ActionKind.RESUME_SPEAKING:
                await self._resume_speaking()

    async def _resume_speaking(self) -> None:
        # `bargeInDeclined` (nodes.py) already zeroed overlap_ms and left
        # `speech_left_ms` untouched -- the remaining synthesized tail
        # continues playing via the same audio stream that was never
        # actually paused server-side (only the automaton's *decision* to
        # keep going needed confirming). Nothing to resend.
        await logger.ainfo("barge_in_resume", session_id=self._session_id)

    # -- closing / idle hangup (FR-25/FR-26) -------------------------------

    async def _synthesize_farewell_if_due(self, elapsed_ms: int, user_speaking: bool) -> int | None:
        """Predicts whether THIS tick will cross `idle_limit_ms` -- if so,
        synthesizes the farewell BEFORE stepping, since `idleHangup`'s
        `dialogue.qnt` effect sets `speech_left_ms` atomically from
        `AutomatonInput.farewell_duration_ms` and needs that duration in
        hand before the transition fires. The guard mirrors `idleHangup`'s
        own guard (`nodes.py`) exactly, evaluated against the same
        pre-step state, so the prediction cannot disagree with what the
        automaton itself decides one line later.
        """
        dialogue = self._automaton.dialogue
        if dialogue.agent is not AgentState.LISTENING or user_speaking:
            return None
        if dialogue.timers.idle_ms + elapsed_ms < self._deps.thresholds.idle_limit_ms:
            return None
        synthesis = await self._deps.tts_worker.synthesize(_FAREWELL_TEXT)
        self._pending_farewell = synthesis
        return int(synthesis.audio_seconds * 1000)

    async def _on_closing_started(self) -> None:
        self._vad.notify_agent_speaking(True)
        # Reuse the synthesis `_synthesize_farewell_if_due` already produced
        # to compute `farewell_duration_ms` -- don't pay for a second TTS
        # call for the same fixed phrase.
        synthesis = self._pending_farewell or await self._deps.tts_worker.synthesize(_FAREWELL_TEXT)
        self._pending_farewell = None
        if not self._mute_tts:
            await self._send_audio(synthesis.pcm16)
        await logger.ainfo("closing_started", session_id=self._session_id)

    async def _on_closing_interrupted(self) -> None:
        self._vad.notify_agent_speaking(False)
        await self._flush_audio(reason="closing_cancelled")
        await logger.ainfo("closing_interrupted", session_id=self._session_id)

    async def _on_session_ended(self) -> None:
        self._vad.notify_agent_speaking(False)
        await self._send_json({"type": "session.ended", "reason": "idle_hangup"})
        self._closed = True
        if self._ws.client_state == WebSocketState.CONNECTED:
            await self._ws.close()


__all__ = [
    "DialogueSession",
    "SessionDependencies",
    "ensure_greeting_audio",
    "read_wav_pcm16",
    "RecordedDecision",
]
