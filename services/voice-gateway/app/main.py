from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio_codec import normalize_audio_for_stt
from .backend_client import BackendClient
from .call_session import CallSessionState, store
from .config import settings
from .metrics import metrics
from .prompts import DEFAULT_GREETING, HANDOFF_PROMPT
from .stt_client import SpeechKitShortAudioSTTClient
from .faster_whisper_stt import FasterWhisperSTTClient
from .tts_client import SpeechKitTTSClient
from .silero_tts import SileroTTSClient

logger = logging.getLogger("voice-gateway")
app = FastAPI(title="DVUI Voice Gateway", version="10.2.1")
backend = BackendClient()
# STT backend selected by config: local faster-whisper (default) or Yandex (fallback).
stt = (FasterWhisperSTTClient() if settings.stt_backend == "faster-whisper"
       else SpeechKitShortAudioSTTClient())
# TTS backend selected by config: local Silero (default) or Yandex SpeechKit (fallback).
tts = SileroTTSClient() if settings.tts_backend == "silero" else SpeechKitTTSClient()
logger.info("STT backend: %s | TTS backend: %s", settings.stt_backend, settings.tts_backend)
cache_dir = Path(settings.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class StartCallRequest(BaseModel):
    call_id: str | None = None
    session_id: str | None = None
    phone_number: str | None = None
    transport: str = "demo"
    direction: str = "inbound"
    metadata: dict | None = None
    synthesize_greeting: bool = False


class TranscriptTurnRequest(BaseModel):
    transcript: str = Field(..., min_length=1)
    session_id: str | None = None
    mode: str = "auto"
    top_k: int = 5


class HandoffRequest(BaseModel):
    session_id: str | None = None
    reason: str | None = None


def _backend_session_id(state: CallSessionState, requested: str | None = None) -> str:
    return state.session_id or requested or state.call_id


def _speech_text(result: dict) -> str:
    return (result.get("tts_text") or result.get("voice_answer") or result.get("answer") or "").strip()


def _repeat_prompt(message: str) -> dict:
    return {
        "answer": message,
        "voice_answer": message,
        "tts_text": message,
        "meta": {"engine": "voice-gateway", "repeat_prompt": True},
        "citations": [],
        "need_clarification": False,
    }


def _store_audio(call_id: str, audio_bytes: bytes, suffix: str, media_type: str) -> tuple[str, str]:
    path = cache_dir / f"{call_id}-{uuid.uuid4().hex[:8]}{suffix}"
    path.write_bytes(audio_bytes)
    url = f"{settings.public_base_url.rstrip('/')}/audio/{path.name}"
    return str(path), url


def _attach_tts(call_id: str, result: dict, state: CallSessionState) -> dict:
    speak_text = _speech_text(result)
    result["tts_text"] = speak_text
    if not speak_text:
        result["tts_status"] = "skipped"
        return result
    try:
        audio_bytes, suffix, media_type = tts.synthesize(speak_text)
        path, url = _store_audio(call_id, audio_bytes, suffix, media_type)
        state.tts_path = path
        store.put(state)
        result["audio_url"] = url
        result["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
        result["audio_mime"] = media_type
        result["tts_status"] = "ok"
        logger.info("TTS ok for call %s -> %s", call_id, url)
    except Exception as exc:
        logger.exception("TTS failed for call %s", call_id)
        result["audio_url"] = None
        result["tts_status"] = "failed"
        result["audio_error"] = str(exc)
    return result


@app.on_event("startup")
async def _warmup_models() -> None:
    """Load STT + TTS in a background thread at boot so the first real request
    isn't hit by the one-time model load (~80s). Health stays up meanwhile."""
    import asyncio

    def _warm() -> None:
        try:
            tts.synthesize("Прогрев синтеза речи.")
            logger.info("TTS warmup done")
        except Exception:
            logger.exception("TTS warmup failed")
        try:
            import numpy as np
            silence = np.zeros(16000, dtype=np.int16).tobytes()
            stt.recognize_bytes(silence, sample_rate_hertz=16000, audio_format="lpcm")
            logger.info("STT warmup done")
        except Exception:
            logger.exception("STT warmup failed")

    asyncio.get_event_loop().run_in_executor(None, _warm)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "voice-gateway", "version": app.version, "mode": settings.mode}


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


@app.get("/", response_class=HTMLResponse)
async def demo_index():
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/demo", response_class=HTMLResponse)
async def demo_index_alias():
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.post("/calls/start")
async def start_call(req: StartCallRequest):
    payload = req.model_dump()
    response = await backend.start_call(payload)
    call_id = response["call_id"]
    session_id = response["session_id"]
    state = store.put(CallSessionState(call_id=call_id, session_id=session_id))
    out = {**response, "transport": req.transport}
    if req.synthesize_greeting:
        out = _attach_tts(call_id, out, state)
    metrics.inc("call_started")
    return out


@app.post("/calls/{call_id}/turn")
async def text_turn(call_id: str, req: TranscriptTurnRequest):
    state = store.get(call_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown call_id")
    session_id = _backend_session_id(state, req.session_id)
    payload = {
        "call_id": call_id,
        "session_id": session_id,
        "transcript": req.transcript,
        "is_final": True,
        "mode": req.mode,
        "top_k": req.top_k,
        "channel": "phone",
    }
    try:
        logger.info("Backend turn start call=%s session=%s", call_id, session_id)
        result = await backend.send_turn(payload)
    except httpx.HTTPStatusError as exc:
        logger.exception("Backend HTTP error for call %s", call_id)
        raise HTTPException(status_code=502, detail=f"Backend error: {exc.response.text}")
    except Exception as exc:
        logger.exception("Unexpected backend error for call %s", call_id)
        raise HTTPException(status_code=502, detail=f"Backend error: {exc}")
    state.session_id = session_id
    state.last_transcript = req.transcript
    state.last_answer = result.get("answer", "")
    store.put(state)
    result = _attach_tts(call_id, result, state)
    metrics.inc("text_turn")
    return result


@app.post("/calls/{call_id}/recognize-and-answer")
async def recognize_and_answer(
    call_id: str,
    session_id: str | None = Form(None),
    mode: str = Form("auto"),
    top_k: int = Form(5),
    sample_rate_hertz: int = Form(8000),
    audio_format: str = Form("lpcm"),
    synthesize_reply: bool = Form(True),
    audio_file: UploadFile = File(...),
):
    state = store.get(call_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown call_id")
    session_id = _backend_session_id(state, session_id)
    uploaded = await audio_file.read()
    audio_bytes, sample_rate_hertz, audio_format = normalize_audio_for_stt(
        uploaded,
        filename=audio_file.filename,
        content_type=audio_file.content_type,
        sample_rate_hertz=sample_rate_hertz,
        audio_format=audio_format,
    )
    try:
        transcript = stt.recognize_bytes(audio_bytes, sample_rate_hertz=sample_rate_hertz, audio_format=audio_format)
        logger.info("STT ok call=%s transcript=%r", call_id, transcript)
    except Exception as exc:
        logger.exception("STT failed for call %s", call_id)
        result = _repeat_prompt("Я не расслышал ответ. Повторите, пожалуйста, ещё раз.")
        result["transcript"] = ""
        result["meta"] = {**(result.get("meta") or {}), "stt_error": str(exc)}
        if synthesize_reply:
            result = _attach_tts(call_id, result, state)
        metrics.inc("audio_turn_retry")
        return result

    if not transcript.strip():
        result = _repeat_prompt("Я не расслышал ответ. Повторите, пожалуйста, ещё раз.")
        result["transcript"] = ""
        if synthesize_reply:
            result = _attach_tts(call_id, result, state)
        metrics.inc("audio_turn_empty")
        return result

    payload = {
        "call_id": call_id,
        "session_id": session_id,
        "transcript": transcript,
        "is_final": True,
        "mode": mode,
        "top_k": top_k,
        "channel": "phone",
    }
    try:
        result = await backend.send_turn(payload)
        logger.info("Backend ok for call %s", call_id)
    except httpx.HTTPStatusError as exc:
        logger.exception("Backend HTTP error for call %s", call_id)
        raise HTTPException(status_code=502, detail=f"Backend error: {exc.response.text}")
    except Exception as exc:
        logger.exception("Unexpected backend error for call %s", call_id)
        raise HTTPException(status_code=502, detail=f"Backend error: {exc}")

    state.session_id = session_id
    state.last_transcript = transcript
    state.last_answer = result.get("answer", "")
    store.put(state)
    if synthesize_reply:
        result = _attach_tts(call_id, result, state)
    result["transcript"] = transcript
    metrics.inc("audio_turn")
    return result


@app.post("/calls/{call_id}/handoff")
async def request_handoff(call_id: str, req: HandoffRequest):
    state = store.get(call_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown call_id")
    payload = {"call_id": call_id, "reason": req.reason}
    try:
        result = await backend.request_handoff(payload)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Backend error: {exc.response.text}")
    result["tts_text"] = HANDOFF_PROMPT
    result = _attach_tts(call_id, result, state)
    metrics.inc("handoff_requested")
    return result


@app.get("/audio/{name}")
async def get_audio(name: str):
    path = cache_dir / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    suffix = path.suffix.lower()
    media_type = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)
