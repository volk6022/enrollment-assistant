"""Shared fixtures for the T-09 acceptance-scenario integration tests
(A-01..A-11, spec.md §4). Everything here talks to REAL infrastructure --
`faster-whisper` on CUDA, `silero` on CPU, and a live `llama-server` on the
three ports `contracts/llm.md` §8 specifies -- there are no mocks in this
package, per the project's own testing rule (`plan.md` §12: "мокать LLM для
контрактных тестов нельзя").

Model loading (whisper CUDA init, silero torch.hub) is expensive (measured:
tens of seconds for silero's first load), so every fixture below is
session-scoped and built exactly once for the whole `tests/integration/` run.

Requires, before running this package:
  - `llama-server` up on 127.0.0.1:20099/20100/20101 (see
    `specs/001-streaming-dialogue/contracts/llm.md` §8 for flags).
  - `.env` present at the repo root with real local paths.
  - A CUDA GPU with faster-whisper's model available.
If any of that is missing, tests here fail loudly (connection refused /
CUDA error) rather than silently skipping -- this package intentionally
does NOT catch those errors and turn them into skips, since a green run is
supposed to mean the acceptance scenarios were actually exercised end to
end, not "skipped because infra wasn't up".
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest
import pytest_asyncio

from backend.config import settings
from backend.dialogue.nodes import DialogueThresholds
from backend.dialogue.scenarios import ScenarioRegistry
from backend.llm.client import LlamaClient
from backend.rag.config import RagSettings
from backend.rag.pipeline import RagPipeline
from backend.stt.whisper_worker import WhisperWorker
from backend.tts.silero_worker import SileroWorker
from backend.ws.session import SessionDependencies, ensure_greeting_audio, read_wav_pcm16

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "audio"

_PREFIX = struct.Struct("<I")


def pack_frame(offset_ms: int, pcm16: bytes) -> bytes:
    return _PREFIX.pack(offset_ms & 0xFFFFFFFF) + pcm16


def load_fixture_pcm16(name: str) -> bytes:
    """Reads a pre-synthesized (Silero TTS) mono 16kHz PCM16 WAV from
    `tests/fixtures/audio/` -- see that directory's generation script in the
    delivery report. These stand in for real human speech: VAD/whisper only
    care about spectral speech-likeness, not who/what produced the audio,
    and no real recorded human dialogue audio exists anywhere in this repo
    for these specific scripted scenarios (see report).
    """
    path = FIXTURES_DIR / f"{name}.wav"
    pcm, rate = read_wav_pcm16(path)
    assert rate == settings.audio.audio_sample_rate, f"{path}: {rate} != {settings.audio.audio_sample_rate}"
    return pcm


def chunk_frames(pcm16: bytes, start_offset_ms: int, chunk_ms: int, sample_rate: int) -> tuple[list[bytes], int]:
    """Splits `pcm16` into `chunk_ms`-sized binary frames on the session's
    `offset_ms` axis, matching what a real client streams
    (contracts/websocket.md §2.1). Returns `(frames, next_offset_ms)` so
    callers can chain silence/more speech immediately after.
    """
    bytes_per_chunk = chunk_ms * sample_rate // 1000 * 2
    frames: list[bytes] = []
    offset = start_offset_ms
    for i in range(0, len(pcm16), bytes_per_chunk):
        piece = pcm16[i : i + bytes_per_chunk]
        frames.append(pack_frame(offset, piece))
        offset += len(piece) // 2 * 1000 // sample_rate
    return frames, offset


def silence_frames(start_offset_ms: int, duration_ms: int, chunk_ms: int, sample_rate: int) -> tuple[list[bytes], int]:
    bytes_per_chunk = chunk_ms * sample_rate // 1000 * 2
    silence_chunk = b"\x00" * bytes_per_chunk
    pcm16 = silence_chunk * (duration_ms // chunk_ms)
    return chunk_frames(pcm16, start_offset_ms, chunk_ms, sample_rate)


@pytest_asyncio.fixture(scope="session")
async def session_deps() -> SessionDependencies:
    tts_worker = SileroWorker(
        speaker=settings.audio.tts_speaker,
        native_sample_rate=settings.audio.tts_sample_rate,
        model_variant=settings.audio.tts_model,
        output_sample_rate=settings.audio.tts_sample_rate,
        device=settings.audio.tts_device,
    )
    await tts_worker.start()

    greeting_path = settings.dialogue.greeting_audio_path
    await ensure_greeting_audio(greeting_path, tts_worker, sample_rate=settings.audio.audio_sample_rate)
    greeting_pcm16, greeting_rate = read_wav_pcm16(greeting_path)
    assert greeting_rate == settings.audio.audio_sample_rate
    greeting_duration_ms = int(len(greeting_pcm16) / 2 / settings.audio.audio_sample_rate * 1000)

    whisper_worker = WhisperWorker(
        model_size=settings.audio.stt_model,
        device=settings.audio.stt_device,
        compute_type=settings.audio.stt_compute_type,
        language=settings.audio.stt_language,
        partial_max_hz=settings.audio.stt_partial_max_hz,
        window_seconds=settings.audio.stt_window_seconds,
        overlap_seconds=settings.audio.stt_overlap_seconds,
        # Forced on for this test package regardless of .env's
        # STT_FINAL_PASS (OQ-03 default is off in production): these tests
        # push audio far faster than a real 100ms-cadence mic stream, so
        # `try_partial`'s Hz throttle can legitimately never get a
        # completed partial in before a simulated turn-end fires. The final
        # pass makes the committed transcript deterministic regardless of
        # that race, which is what A-01/A-05 etc. actually need to check --
        # they are not testing the partial-throttle behaviour itself (that
        # has its own unit coverage).
        final_pass=True,
        warmup_audio_pcm16=greeting_pcm16,
    )
    await whisper_worker.start()

    llama_client = LlamaClient(endpoint=settings.llm.llm_endpoint, timeout_s=settings.llm.llm_timeout_s)

    rag_cfg = settings.rag
    llm_cfg = settings.llm
    rag_settings = RagSettings(
        embedding_endpoint=llm_cfg.embedding_endpoint,
        reranker_endpoint=llm_cfg.reranker_endpoint,
        rag_top_k=rag_cfg.rag_top_k,
        rag_fused_top=rag_cfg.rag_fused_top,
        rag_dense_top=rag_cfg.rag_dense_top,
        rag_bm25_top=rag_cfg.rag_bm25_top,
        rag_rrf_k=rag_cfg.rag_rrf_k,
        rag_min_score=rag_cfg.rag_min_score,
        rag_max_length=rag_cfg.rag_max_length,
        rag_batch_size=rag_cfg.rag_batch_size,
        faiss_index_path=str(rag_cfg.faiss_index_path),
        kb_source_path=str(rag_cfg.kb_source_path),
    )
    from concurrent.futures import ThreadPoolExecutor

    rag_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-test")
    rag_pipeline = RagPipeline(artifacts_dir=rag_cfg.faiss_index_path, settings=rag_settings)

    scenario_registry = ScenarioRegistry.load(settings.dialogue.scenarios_path)

    d = settings.dialogue
    thresholds = DialogueThresholds(
        turn_limit_ms=int(d.dialogue_interject_after_s * 1000),
        idle_limit_ms=int(d.dialogue_idle_hangup_s * 1000),
        overlap_limit_ms=int(d.dialogue_barge_in_overlap_s * 1000),
        tail_min_ms=int(d.dialogue_barge_in_min_tail_s * 1000),
    )

    deps = SessionDependencies(
        settings=settings,
        llama_client=llama_client,
        whisper_worker=whisper_worker,
        tts_worker=tts_worker,
        rag_pipeline=rag_pipeline,
        rag_executor=rag_executor,
        scenario_registry=scenario_registry,
        thresholds=thresholds,
        greeting_pcm16=greeting_pcm16,
        greeting_duration_ms=greeting_duration_ms,
    )
    yield deps

    await llama_client.aclose()
    whisper_worker.close()
    tts_worker.close()
    rag_executor.shutdown(wait=True)


@pytest.fixture()
def chunk_ms() -> int:
    return settings.audio.audio_chunk_size_ms
