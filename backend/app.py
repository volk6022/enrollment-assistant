"""FastAPI-приложение: `/ws/dialogue`, `GET /health`, `GET /metrics`, `POST /answer`.

Полная оркестрация диалога (сокет → кольцо → VAD → STT → автомат → RAG/LLM → TTS →
сокет) реализована в `backend/ws/session.py` (T-09); этот модуль отвечает за:

- Сборку всех процесс-уровневых зависимостей ровно один раз в `lifespan`:
  `LlamaClient`, `WhisperWorker`/`SileroWorker` (прогретые, включая генерацию
  `GREETING_AUDIO_PATH` через `ensure_greeting_audio` — нужен и для FR-01, и для
  прогрева whisper реальной речью, plan.md §2), `RagPipeline` + выделенный
  однопоточный пул "rag" (plan.md §2/§9), `ScenarioRegistry` (падает на старте при
  битом `dialogue/scenarios.yaml`) и `DialogueThresholds`, собранные из
  `settings.dialogue.*_s * 1000` (plan.md §9 «Соответствие модели и кода»).
- `/ws/dialogue`: handshake (`session.ready` первым кадром, contracts/websocket.md §1)
  и делегирование всей остальной работы новому `DialogueSession` на каждое
  соединение.
- `/health` реально пингует три эндпоинта llama-server (LLM/эмбеддер/реранкер) и
  отдаёт статус STT/TTS-воркеров через `app.state.stt_worker`/`app.state.tts_worker`.
- `/answer` — заглушка T-01 оставлена как есть: одноходовый REST-путь не входит в
  объём T-09 (FR-01…FR-03, FR-09…FR-17 — все про `/ws/dialogue`), подключение
  RAG/LLM к нему не выполнялось, чтобы не пересекаться с T-03/T-04.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from backend.config import settings
from backend.dialogue.nodes import DialogueThresholds
from backend.dialogue.scenarios import ScenarioRegistry
from backend.llm.client import LlamaClient
from backend.rag.config import RagSettings
from backend.rag.pipeline import RagPipeline
from backend.stt.whisper_worker import WhisperWorker
from backend.telemetry import configure_logging, get_logger
from backend.tts.silero_worker import SileroWorker
from backend.ws.session import DialogueSession, SessionDependencies, ensure_greeting_audio, read_wav_pcm16

logger = get_logger(__name__)


def _build_thresholds() -> DialogueThresholds:
    """`dialogue.qnt`'s ordinal TALK_LIMIT/IDLE_LIMIT/OVERLAP_LIMIT/TAIL_MIN,
    in real milliseconds (tasks.md T-09, plan.md §9's correspondence table).
    """
    d = settings.dialogue
    return DialogueThresholds(
        turn_limit_ms=int(d.dialogue_interject_after_s * 1000),
        idle_limit_ms=int(d.dialogue_idle_hangup_s * 1000),
        overlap_limit_ms=int(d.dialogue_barge_in_overlap_s * 1000),
        tail_min_ms=int(d.dialogue_barge_in_min_tail_s * 1000),
    )


def _build_rag_settings() -> RagSettings:
    """Adapter from `backend.config.Settings` to `RagSettingsProtocol`
    (`backend/rag/config.py`'s own wiring note for whoever lands T-01/T-09).
    """
    rag_cfg, llm_cfg = settings.rag, settings.llm
    return RagSettings(
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


class ModelWorker(Protocol):
    """Интерфейс, которым `/health` опрашивает whisper/silero-воркеры.

    `WhisperWorker`/`SileroWorker` (T-05) сами предоставляют только
    `is_warm`, без метода `health()` — `_WorkerHealthAdapter` ниже оборачивает
    их для `/health`, не трогая их модуль (задача T-09 не переписывает чужой
    код). Пока воркер не подключён, `/health` честно отвечает "не загружен".
    """

    def health(self) -> dict[str, object]: ...


class _WorkerHealthAdapter:
    """`ModelWorker` для `WhisperWorker`/`SileroWorker`, которые сами такого
    метода не предоставляют — оборачиваем снаружи вместо правки T-05.
    """

    def __init__(self, name: str, worker: WhisperWorker | SileroWorker) -> None:
        self._name = name
        self._worker = worker

    def health(self) -> dict[str, object]:
        return {"loaded": True, "warm": self._worker.is_warm, "component": self._name}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log)
    await logger.ainfo(
        "backend_starting",
        api_host=settings.api.api_host,
        api_port=settings.api.api_port,
        llm_endpoint=settings.llm.llm_endpoint,
    )

    app.state.http_client = httpx.AsyncClient()
    app.state.started_at = time.monotonic()
    app.state.ws_connections_total = 0
    app.state.ws_connections_active = 0
    app.state.stt_worker = None
    app.state.tts_worker = None

    # -- T-09: process-level singletons shared by every DialogueSession -----
    # (plan.md §2/§9 -- three dedicated single-worker pools, one LlamaClient,
    # one RagPipeline; NFR-06 -- everything except llama-server itself lives
    # in this one process).
    tts_worker = SileroWorker(
        speaker=settings.audio.tts_speaker,
        native_sample_rate=settings.audio.tts_sample_rate,
        model_variant=settings.audio.tts_model,
        output_sample_rate=settings.audio.tts_sample_rate,
        device=settings.audio.tts_device,
    )
    await tts_worker.start()
    app.state.tts_worker = _WorkerHealthAdapter("tts", tts_worker)

    # FR-01's pre-recorded greeting doubles as whisper's warmup audio (plan.md
    # §2: "тишина прогревом не является") -- must exist BEFORE WhisperWorker
    # is constructed with it.
    greeting_path: Path = settings.dialogue.greeting_audio_path
    await ensure_greeting_audio(greeting_path, tts_worker, sample_rate=settings.audio.audio_sample_rate)
    greeting_pcm16, greeting_rate = read_wav_pcm16(greeting_path)
    if greeting_rate != settings.audio.audio_sample_rate:
        raise RuntimeError(
            f"{greeting_path}: sample rate {greeting_rate} != AUDIO_SAMPLE_RATE "
            f"{settings.audio.audio_sample_rate} -- regenerate the greeting file"
        )
    greeting_duration_ms = int(len(greeting_pcm16) / 2 / settings.audio.audio_sample_rate * 1000)

    whisper_worker = WhisperWorker(
        model_size=settings.audio.stt_model,
        device=settings.audio.stt_device,
        compute_type=settings.audio.stt_compute_type,
        language=settings.audio.stt_language,
        partial_max_hz=settings.audio.stt_partial_max_hz,
        window_seconds=settings.audio.stt_window_seconds,
        overlap_seconds=settings.audio.stt_overlap_seconds,
        final_pass=settings.audio.stt_final_pass,
        warmup_audio_pcm16=greeting_pcm16,
    )
    await whisper_worker.start()
    app.state.stt_worker = _WorkerHealthAdapter("stt", whisper_worker)

    llama_client = LlamaClient(endpoint=settings.llm.llm_endpoint, timeout_s=settings.llm.llm_timeout_s)

    rag_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
    rag_pipeline = RagPipeline(artifacts_dir=settings.rag.faiss_index_path, settings=_build_rag_settings())

    # Falls loudly on a typo in scenarios.yaml (dialogue/scenarios.py's own
    # contract) -- process must not start with a scenario that can never
    # match anything.
    scenario_registry = ScenarioRegistry.load(settings.dialogue.scenarios_path)

    app.state.session_deps = SessionDependencies(
        settings=settings,
        llama_client=llama_client,
        whisper_worker=whisper_worker,
        tts_worker=tts_worker,
        rag_pipeline=rag_pipeline,
        rag_executor=rag_executor,
        scenario_registry=scenario_registry,
        thresholds=_build_thresholds(),
        greeting_pcm16=greeting_pcm16,
        greeting_duration_ms=greeting_duration_ms,
    )
    await logger.ainfo(
        "backend_ready",
        scenarios=len(scenario_registry.scenarios),
        greeting_duration_ms=greeting_duration_ms,
    )

    try:
        yield
    finally:
        await logger.ainfo("backend_stopping")
        await app.state.http_client.aclose()
        await llama_client.aclose()
        whisper_worker.close()
        tts_worker.close()
        rag_executor.shutdown(wait=True)


app = FastAPI(title="Потоковый диалоговый ассистент приёмной комиссии", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _probe_llama_endpoint(client: httpx.AsyncClient, endpoint: str, timeout: float) -> dict[str, object]:
    """Реальный запрос `GET {endpoint}/health` — llama-server отдаёт этот путь из
    коробки. «Процесс жив» не проверяется — проверяется именно достижимость.
    """
    started = time.monotonic()
    try:
        response = await client.get(f"{endpoint}/health", timeout=timeout)
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        return {
            "endpoint": endpoint,
            "reachable": response.status_code == 200,
            "http_status": response.status_code,
            "latency_ms": latency_ms,
        }
    except httpx.HTTPError as exc:
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        return {
            "endpoint": endpoint,
            "reachable": False,
            "latency_ms": latency_ms,
            "error": str(exc),
        }


def _worker_health(worker: ModelWorker | None, name: str) -> dict[str, object]:
    if worker is None:
        return {"loaded": False, "detail": f"{name} worker not wired yet"}
    return worker.health()


@app.get("/health")
async def health() -> JSONResponse:
    client: httpx.AsyncClient = app.state.http_client
    timeout = settings.llm.llm_timeout_s

    llm_check, embedding_check, reranker_check = await asyncio.gather(
        _probe_llama_endpoint(client, settings.llm.llm_endpoint, timeout),
        _probe_llama_endpoint(client, settings.llm.embedding_endpoint, timeout),
        _probe_llama_endpoint(client, settings.llm.reranker_endpoint, timeout),
    )

    llama_reachable = all(
        check["reachable"] for check in (llm_check, embedding_check, reranker_check)
    )

    body = {
        "status": "ok" if llama_reachable else "degraded",
        "llm": llm_check,
        "embedding": embedding_check,
        "reranker": reranker_check,
        "stt": _worker_health(app.state.stt_worker, "stt"),
        "tts": _worker_health(app.state.tts_worker, "tts"),
    }
    return JSONResponse(content=body, status_code=200 if llama_reachable else 503)


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus-совместимые счётчики (contracts/websocket.md §5). Полный набор —
    T-12; здесь только то, что сам каркас app.py реально знает про себя: аптайм и
    вебсокет-соединения. METRICS_ENABLED=false отдаёт пустой тело, а не 404 —
    эндпоинт остаётся частью контракта независимо от флага.
    """
    if not settings.log.metrics_enabled:
        return PlainTextResponse(content="", media_type="text/plain; version=0.0.4")

    uptime_s = round(time.monotonic() - app.state.started_at, 1)
    lines = [
        "# HELP backend_uptime_seconds Время работы процесса backend.",
        "# TYPE backend_uptime_seconds gauge",
        f"backend_uptime_seconds {uptime_s}",
        "# HELP backend_ws_connections_total Всего открытых WebSocket-сессий.",
        "# TYPE backend_ws_connections_total counter",
        f"backend_ws_connections_total {app.state.ws_connections_total}",
        "# HELP backend_ws_connections_active Активных WebSocket-сессий сейчас.",
        "# TYPE backend_ws_connections_active gauge",
        f"backend_ws_connections_active {app.state.ws_connections_active}",
    ]
    return PlainTextResponse(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


class AnswerRequest(BaseModel):
    question: str


@app.post("/answer")
async def answer(body: AnswerRequest) -> JSONResponse:
    """`POST /answer` — одноходовый текстовый ответ без сокета, для тестов и
    совместимости (contracts/websocket.md §5). RAG/LLM-пайплайн (T-03/T-04) ещё не
    подключён к этому каркасу, поэтому отвечаем честной заглушкой: тот же набор
    ключей, что и в проде, но `tts_status: "not_implemented"` вместо выдуманного
    ответа.
    """
    payload = {
        "answer": None,
        "citations": [],
        "meta": {
            "engine": "streaming-dialogue",
            "implemented": False,
            "note": "RAG/LLM-пайплайн ещё не подключён (T-03/T-04)",
        },
        "need_clarification": False,
        "tts_text": None,
        "voice_answer": None,
        "audio_url": None,
        "audio_base64": None,
        "audio_mime": None,
        "tts_status": "not_implemented",
    }
    return JSONResponse(content=payload, status_code=200)


@app.websocket("/ws/dialogue")
async def ws_dialogue(websocket: WebSocket) -> None:
    """Full T-09 orchestration, delegated to `DialogueSession`
    (`backend/ws/session.py`). This endpoint's own job stops at bookkeeping
    connection counters and constructing a fresh session around the shared
    `SessionDependencies` singletons built in `lifespan` -- `session.ready`,
    the audio/JSON message loop, and every FR-01..FR-17 behaviour live
    entirely inside `DialogueSession.run()`.
    """
    app.state.ws_connections_total += 1
    app.state.ws_connections_active += 1
    session_id = uuid.uuid4().hex
    try:
        session = DialogueSession(websocket=websocket, session_id=session_id, deps=app.state.session_deps)
        await session.run()
    finally:
        app.state.ws_connections_active -= 1


def run() -> None:
    """Точка входа контейнера (`backend/Dockerfile`, CMD). Пинги вебсокета
    настраиваются здесь через параметры uvicorn — контракт §4 («ping каждые 20 с,
    таймаут 60 с») не требует кода в session.py, только правильную конфигурацию
    сервера.
    """
    uvicorn.run(
        "backend.app:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        ws_ping_interval=settings.api.ws_ping_interval_s,
        ws_ping_timeout=settings.api.ws_ping_timeout_s,
    )


if __name__ == "__main__":
    run()
