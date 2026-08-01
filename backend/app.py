"""FastAPI-приложение: `/ws/dialogue`, `GET /health`, `GET /metrics`, `POST /answer`.

Полная оркестрация диалога (сокет → кольцо → VAD → STT → автомат → RAG/LLM → TTS →
сокет) — задача T-09 (`backend/ws/session.py`). Здесь — только каркас: сборка
зависимостей в lifespan, graceful shutdown, и эндпоинты в объёме, который T-01
может честно реализовать без остальных волн:

- `/health` реально пингует три эндпоинта llama-server (LLM/эмбеддер/реранкер) и
  отдаёт статус STT/TTS-воркеров, если они уже подключены (T-05 присваивает их в
  `app.state.stt_worker` / `app.state.tts_worker`; до этого честно "не загружены").
- `/ws/dialogue` делает только handshake из contracts/websocket.md §1 (`session.ready`
  первым кадром) и закрывается с `session.ended(reason="not_implemented")` — сама
  оркестрация ждёт T-09.
- `/answer` возвращает 200 с полным набором ключей контракта §5, но честно помеченный
  `tts_status: "not_implemented"` — RAG/LLM-пайплайн (T-03/T-04) ещё не подключён.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Protocol

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from backend.config import settings
from backend.telemetry import configure_logging, get_logger

logger = get_logger(__name__)


class ModelWorker(Protocol):
    """Интерфейс, которым T-05 подключает whisper/silero-воркеры к `/health`.

    `WhisperWorker`/`SileroWorker` (backend/stt, backend/tts) обязаны предоставить
    этот метод и присвоить себя в `app.state.stt_worker`/`app.state.tts_worker` при
    старте. Пока воркер не подключён, `/health` честно отвечает "не загружен",
    а не притворяется, что процесс жив, и есть модель.
    """

    def health(self) -> dict[str, object]: ...


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
    # T-05 присваивает реальные воркеры сюда после прогрева (plan.md §2, «прогрев
    # whisper из своего воркер-потока»). До этого момента оба — None.
    app.state.stt_worker = None
    app.state.tts_worker = None

    try:
        yield
    finally:
        await logger.ainfo("backend_stopping")
        await app.state.http_client.aclose()


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
    """Заглушка — полная реализация в `backend/ws/session.py` (T-09).

    То, что можно честно сделать уже сейчас — сам протокольный handshake из
    contracts/websocket.md §1: `session.ready` первым кадром, ничего от клиента
    до этого момента не читаем. После этого оркестрации ещё нет, поэтому сессия
    сразу и явно закрывается с `session.ended(reason="not_implemented")`, а не
    висит и не притворяется рабочим диалогом.
    """
    await websocket.accept()
    app.state.ws_connections_total += 1
    app.state.ws_connections_active += 1

    session_id = uuid.uuid4().hex
    t0_utc = datetime.now(UTC)

    try:
        await websocket.send_json(
            {
                "type": "session.ready",
                "session_id": session_id,
                "t0_utc": t0_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        await logger.ainfo("ws_session_ready", session_id=session_id)

        await websocket.send_json(
            {
                "type": "status",
                "text": "Диалоговая оркестрация ещё не подключена (T-09).",
                "level": "info",
            }
        )
        await websocket.send_json({"type": "session.ended", "reason": "not_implemented"})
        await websocket.close()
    except WebSocketDisconnect:
        await logger.ainfo("ws_session_disconnected", session_id=session_id)
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
