import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .call_policies import should_recommend_handoff
from .config import settings
from .db import SessionLocal
from .knowledge import answer_from_contacts, build_combined_clarification, infer_question_intent, wants_combined_clarification
from .legal_hierarchy import (
    is_documents_question,
    prefers_answer_first,
    question_profile,
    requires_precise_form,
    requires_precise_level,
    requires_precise_status,
)
from .llm_generator import generate_llm_answer
from .models import (
    DialogMessage,
    DialogSession,
    Document,
    DocumentVersion,
    VoiceCallEvent,
    VoiceCallSession,
    VoiceCallTurn,
)
from .rag import RAG
from .orchestrator_v11 import V11Orchestrator
from .main_helpers import parse_form, parse_level, parse_status
from .tts_text import build_tts_text

logger = logging.getLogger("backend")
app = FastAPI(title="DVUI Applicant Assistant API", version="11.3.0")
_rag = None
_rag_init_error = None
_v11 = None
_v11_init_error = None


def get_v11_orchestrator(raise_on_error: bool = False):
    global _v11, _v11_init_error
    if _v11 is not None:
        return _v11
    try:
        _v11 = V11Orchestrator()
        _v11_init_error = None
        return _v11
    except Exception as exc:
        _v11_init_error = str(exc)
        logger.exception("V11 orchestrator initialization failed: %s", exc)
        if raise_on_error:
            raise
        return None


def get_rag(raise_on_error: bool = False):
    global _rag, _rag_init_error
    if _rag is not None:
        return _rag
    try:
        _rag = RAG()
        _rag_init_error = None
        return _rag
    except Exception as exc:
        _rag_init_error = str(exc)
        logger.exception("RAG initialization failed: %s", exc)
        if raise_on_error:
            raise
        return None


def _rag_unavailable_response(question: str) -> dict:
    msg = (
        "База знаний сейчас инициализируется и временно недоступна. "
        "Попробуйте повторить вопрос чуть позже."
    )
    return {
        "voice_answer": msg,
        "answer": msg,
        "tts_text": msg,
        "citations": [],
        "meta": {"engine": "fallback", "rag_unavailable": True, "rag_error": _rag_init_error, "backend_engine": settings.effective_backend_engine},
    }


class AskRequest(BaseModel):
    session_id: str = Field(..., min_length=3, max_length=64)
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)
    mode: Literal["auto", "llm", "extractive"] = "auto"


class VoiceStartRequest(BaseModel):
    call_id: Optional[str] = None
    session_id: Optional[str] = None
    phone_number: Optional[str] = None
    transport: str = "demo"
    direction: str = "inbound"
    metadata: Optional[dict] = None


class VoiceTurnRequest(BaseModel):
    call_id: str = Field(..., min_length=3, max_length=64)
    session_id: str = Field(..., min_length=3, max_length=64)
    transcript: str = Field(..., min_length=1)
    is_final: bool = True
    top_k: int = Field(5, ge=1, le=10)
    mode: Literal["auto", "llm", "extractive"] = "auto"
    channel: Literal["phone", "chat"] = "phone"


class VoiceHandoffRequest(BaseModel):
    call_id: str
    reason: Optional[str] = None


class VoiceStartResponse(BaseModel):
    call_id: str
    session_id: str
    greeting: str
    tts_text: str


class VoiceTurnResponse(BaseModel):
    call_id: str
    session_id: str
    transcript: str
    answer: str
    tts_text: str
    voice_answer: str
    citations: list
    handoff_recommended: bool = False
    need_clarification: bool = False
    clarification_type: Optional[str] = None
    meta: dict


class VoiceHandoffResponse(BaseModel):
    call_id: str
    handoff_requested: bool
    message: str


DEFAULT_GREETING = (
    "Здравствуйте. Вас приветствует голосовой ассистент приемной комиссии. "
    "Задайте, пожалуйста, ваш вопрос о поступлении."
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        return
    if x_admin_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_active_collection(db: Session, slug: str = "legal_corpus") -> str:
    active = db.execute(
        select(DocumentVersion)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.slug == slug, DocumentVersion.is_active == True)  # noqa: E712
    ).scalar_one_or_none()
    if active and active.qdrant_collection:
        return active.qdrant_collection
    return os.getenv("QDRANT_COLLECTION_LEGAL") or os.getenv("QDRANT_COLLECTION_NPA") or os.getenv("QDRANT_COLLECTION") or "dvui_legal_2025"


def needs_status(question: str) -> bool:
    intent = infer_question_intent(question)
    if question_profile(question) == "organizational" or prefers_answer_first(question):
        return False
    if intent in {"without_ege", "benefits", "eligibility"}:
        return True
    q = (question or "").lower()
    return requires_precise_status(question) or any(k in q for k in ["без егэ", "льгот", "особое право", "квот", "целев", "могу ли", "имею ли право"])


def needs_level(question: str) -> bool:
    intent = infer_question_intent(question)
    if parse_level(question):
        return False
    if question_profile(question) == "organizational":
        return False
    if intent in {"min_scores", "exams", "without_ege"}:
        return True
    if prefers_answer_first(question) and not requires_precise_level(question):
        return False
    q = (question or "").lower()
    return requires_precise_level(question) or any(k in q for k in ["экзам", "испытан", "зачисл", "конкурс", "балл"])


def needs_form(question: str) -> bool:
    intent = infer_question_intent(question)
    if parse_form(question):
        return False
    if intent in {"min_scores", "exams", "without_ege"}:
        return False
    if prefers_answer_first(question) and not requires_precise_form(question):
        return False
    q = (question or "").lower()
    return requires_precise_form(question) or any(k in q for k in ["очная", "заочная", "форма", "дистанц", "онлайн", "дот"])


def _get_form(sess: DialogSession) -> Optional[str]:
    return getattr(sess, "study_form", None)


def _missing_slots(question: str, sess: DialogSession) -> list[str]:
    missing = []
    if needs_level(question) and not sess.level:
        missing.append("level")
    if needs_status(question) and not sess.status:
        missing.append("status")
    if needs_form(question) and not _get_form(sess):
        missing.append("form")
    return missing


def _build_effective_question(question: str, sess: DialogSession) -> str:
    hint = []
    if sess.status:
        hint.append(f"статус:{sess.status}")
    if sess.level:
        hint.append(f"уровень:{sess.level}")
    if _get_form(sess):
        hint.append(f"форма:{_get_form(sess)}")
    return question + (" " + " ".join(hint) if hint else "")


def _looks_like_new_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    return any(token in t for token in ["что", "как", "какие", "какой", "когда", "куда", "где", "можно", "нужно", "сколько", "ли "])


def _persist_assistant_message(db: Session, session_id: str, text: str, citations=None):
    db.add(DialogMessage(session_id=session_id, role="assistant", text=text, citations=citations))
    db.commit()


def _persist_call_event(db: Session, call_id: str, event_type: str, payload: Optional[dict] = None):
    db.add(VoiceCallEvent(call_id=call_id, event_type=event_type, payload_json=payload))
    db.commit()


def _mask_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 4:
        return "***"
    return f"***{digits[-4:]}"


def _maybe_answer_from_contacts(question: str) -> Optional[dict]:
    if question_profile(question) != "organizational":
        return None
    return answer_from_contacts(question)


def _choose_clarification(question: str, sess: DialogSession, hits) -> Optional[str]:
    intent = infer_question_intent(question)
    rag = get_rag()
    if rag is not None and rag.has_actionable_hits(question, hits):
        return None
    if question_profile(question) == "organizational":
        return None
    missing = _missing_slots(question, sess)
    if not missing:
        return None
    if wants_combined_clarification(question):
        if intent in {"min_scores", "exams"} and missing == ["level"]:
            return "level"
        return "combined"
    if prefers_answer_first(question) or is_documents_question(question):
        return None
    return "combined" if len(missing) > 1 else missing[0]


def _ask_status(sess: DialogSession, db: Session, original: str):
    msg = "Уточните, пожалуйста: вы поступаете как гражданин впервые, как действующий сотрудник ОВД, или по прямому набору?"
    sess.pending_clarification = "status"
    sess.pending_question = original
    db.commit()
    _persist_assistant_message(db, sess.session_id, msg)
    return {"voice_answer": msg, "answer": msg, "citations": [], "need_clarification": True, "clarification_type": "status", "meta": {"engine": "extractive"}}


def _ask_level(sess: DialogSession, db: Session, original: str):
    msg = "Уточните уровень обучения: СПО, бакалавриат, специалитет, магистратура или адъюнктура?"
    sess.pending_clarification = "level"
    sess.pending_question = original
    db.commit()
    _persist_assistant_message(db, sess.session_id, msg)
    return {"voice_answer": msg, "answer": msg, "citations": [], "need_clarification": True, "clarification_type": "level", "meta": {"engine": "extractive"}}


def _ask_form(sess: DialogSession, db: Session, original: str):
    msg = "Уточните форму обучения: очная, заочная или дистанционная?"
    sess.pending_clarification = "form"
    sess.pending_question = original
    db.commit()
    _persist_assistant_message(db, sess.session_id, msg)
    return {"voice_answer": msg, "answer": msg, "citations": [], "need_clarification": True, "clarification_type": "form", "meta": {"engine": "extractive"}}


def _ask_combined(sess: DialogSession, db: Session, original: str):
    msg = build_combined_clarification(original, _missing_slots(original, sess))
    sess.pending_clarification = "combined"
    sess.pending_question = original
    db.commit()
    _persist_assistant_message(db, sess.session_id, msg)
    return {"voice_answer": msg, "answer": msg, "citations": [], "need_clarification": True, "clarification_type": "combined", "meta": {"engine": "extractive"}}


def _maybe_ask_clarification(question: str, sess: DialogSession, db: Session, hits):
    slot = _choose_clarification(question, sess, hits)
    if slot == "status":
        return _ask_status(sess, db, question)
    if slot == "level":
        return _ask_level(sess, db, question)
    if slot == "form":
        return _ask_form(sess, db, question)
    if slot == "combined":
        return _ask_combined(sess, db, question)
    return None


def _apply_inline_slots(sess: DialogSession, text: str, db: Session):
    changed = False
    st = parse_status(text)
    if st and not sess.status:
        sess.status = st
        changed = True
    lv = parse_level(text)
    if lv and not sess.level:
        sess.level = lv
        changed = True
    fm = parse_form(text)
    if fm and not _get_form(sess):
        sess.study_form = fm
        changed = True
    if changed:
        db.commit()


def _llm_available() -> bool:
    return (os.getenv("LLM_PROVIDER") or "").strip().lower() == "ollama" and bool(os.getenv("OLLAMA_BASE_URL")) and bool(os.getenv("OLLAMA_MODEL"))


def _effective_engine() -> str:
    return settings.effective_backend_engine


def _answer_with_llm_or_fallback(question: str, hits, use_llm: bool, strict_llm: bool = False) -> dict:
    if use_llm:
        try:
            hits_for_llm = [{"text": h.text, "payload": h.payload, "score": h.score} for h in hits]
            return generate_llm_answer(question, hits_for_llm)
        except Exception as e:
            logger.exception("LLM failed: %s", e)
            rag = get_rag()
            if rag is None:
                return _rag_unavailable_response(question)
            fb = rag.format_answer_without_llm(question, hits)
            fb["meta"] = {**(fb.get("meta") or {}), "engine": "extractive", "llm_error": str(e)}
            if strict_llm:
                fb["voice_answer"] = "Не удалось обратиться к языковой модели. Отвечаю по найденным документам."
                fb["answer"] = "Не удалось обратиться к языковой модели (LLM). Ниже — выдержки из документов.\n\n" + fb.get("answer", "")
            return fb
    rag = get_rag()
    if rag is None:
        return _rag_unavailable_response(question)
    out = rag.format_answer_without_llm(question, hits)
    out["meta"] = {**(out.get("meta") or {}), "engine": "extractive"}
    return out


def _ensure_dialog_session(db: Session, session_id: str) -> DialogSession:
    sess = db.get(DialogSession, session_id)
    if not sess:
        sess = DialogSession(session_id=session_id)
        db.add(sess)
        db.commit()
        db.refresh(sess)
    return sess


def _process_text_request(req: AskRequest, db: Session) -> dict:
    engine = _effective_engine()
    collection = get_active_collection(db, slug="legal_corpus")
    sess = _ensure_dialog_session(db, req.session_id)

    db.add(DialogMessage(session_id=req.session_id, role="user", text=req.question, citations=None))
    db.commit()

    use_llm = False if req.mode == "extractive" else True if req.mode == "llm" else _llm_available()
    strict_llm = req.mode == "llm"

    if engine == "yandex_v11":
        try:
            orchestrator = get_v11_orchestrator(raise_on_error=True)
            result = orchestrator.answer(req.question, top_k=req.top_k)
            _persist_assistant_message(db, req.session_id, result.get("answer", ""), result.get("citations"))
            return result
        except Exception as exc:
            logger.exception("V11 pipeline failed: %s", exc)
            msg = "Yandex v11 pipeline сейчас недоступен. Проверьте настройки AI Studio и vector store."
            result = {"answer": msg, "voice_answer": msg, "tts_text": msg, "citations": [], "meta": {"engine": "fallback", "backend_engine": engine, "v11_error": str(exc)}}
            _persist_assistant_message(db, req.session_id, result.get("answer", ""), result.get("citations"))
            return result

    if sess.pending_clarification in ("status", "level", "form", "combined"):
        original = sess.pending_question or req.question
        expected = sess.pending_clarification
        _apply_inline_slots(sess, req.question, db)
        filled_expected = (
            (expected == "status" and bool(sess.status))
            or (expected == "level" and bool(sess.level))
            or (expected == "form" and bool(_get_form(sess)))
            or (expected == "combined" and any([sess.status, sess.level, _get_form(sess)]))
        )
        new_question_interrupt = not filled_expected and _looks_like_new_question(req.question)
        if new_question_interrupt:
            sess.pending_clarification = None
            sess.pending_question = None
            db.commit()
        else:
            contact_result = _maybe_answer_from_contacts(original)
            if contact_result:
                sess.pending_clarification = None
                sess.pending_question = None
                db.commit()
                _persist_assistant_message(db, req.session_id, contact_result.get("answer", ""), contact_result.get("citations"))
                return contact_result
            effective_q = _build_effective_question(original, sess)
            rag = get_rag()
            if rag is None:
                result = _rag_unavailable_response(original)
                _persist_assistant_message(db, req.session_id, result.get("answer", ""), result.get("citations"))
                return result
            hits = rag.search(effective_q, collection=collection, top_k=req.top_k)
            if expected != "combined":
                maybe = _maybe_ask_clarification(original, sess, db, hits)
                if maybe:
                    return maybe
            sess.pending_clarification = None
            sess.pending_question = None
            db.commit()
            result = _answer_with_llm_or_fallback(effective_q, hits, use_llm=use_llm, strict_llm=strict_llm)
            _persist_assistant_message(db, req.session_id, result.get("answer", ""), result.get("citations"))
            return result

    _apply_inline_slots(sess, req.question, db)
    contact_result = _maybe_answer_from_contacts(req.question)
    if contact_result:
        _persist_assistant_message(db, req.session_id, contact_result.get("answer", ""), contact_result.get("citations"))
        return contact_result
    effective_q = _build_effective_question(req.question, sess)
    rag = get_rag()
    if rag is None:
        result = _rag_unavailable_response(req.question)
        _persist_assistant_message(db, req.session_id, result.get("answer", ""), result.get("citations"))
        return result
    hits = rag.search(effective_q, collection=collection, top_k=req.top_k)
    maybe = _maybe_ask_clarification(req.question, sess, db, hits)
    if maybe:
        return maybe
    result = _answer_with_llm_or_fallback(effective_q, hits, use_llm=use_llm, strict_llm=strict_llm)
    _persist_assistant_message(db, req.session_id, result.get("answer", ""), result.get("citations"))
    return result


def _ensure_voice_call(db: Session, call_id: str, session_id: str, *, phone_number: Optional[str] = None, transport: str = "demo", direction: str = "inbound", metadata: Optional[dict] = None) -> VoiceCallSession:
    call = db.get(VoiceCallSession, call_id)
    if call:
        return call
    call = VoiceCallSession(
        call_id=call_id,
        session_id=session_id,
        phone_number_masked=_mask_phone(phone_number),
        transport=transport,
        direction=direction,
        metadata_json=metadata,
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    _persist_call_event(db, call_id, "call_started", {"session_id": session_id, "transport": transport})
    return call


@app.get("/health")
def health():
    return {"status": "ok", "service": "backend", "version": app.version, "backend_engine": settings.effective_backend_engine}


@app.post("/admin/reload")
def reload_cache(db: Session = Depends(get_db), _admin=Depends(require_admin)):
    collections = [get_active_collection(db, slug="legal_corpus")]
    extra = os.getenv("QDRANT_COLLECTION_NPA", os.getenv("QDRANT_COLLECTION", "")).strip()
    if extra and extra not in collections:
        collections.append(extra)
    rag = get_rag()
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG is unavailable")
    refreshed = []
    for collection in collections:
        n = rag.refresh_cache(collection)
        refreshed.append({"collection": collection, "chunks_cached": n})
    return {"status": "ok", "collections": refreshed}


@app.post("/ask")
def ask(req: AskRequest, db: Session = Depends(get_db)):
    return _process_text_request(req, db)


@app.post("/voice/start", response_model=VoiceStartResponse)
def voice_start(req: VoiceStartRequest, db: Session = Depends(get_db)):
    call_id = req.call_id or uuid.uuid4().hex[:24]
    session_id = req.session_id or f"call-{call_id}"
    _ensure_dialog_session(db, session_id)
    _ensure_voice_call(db, call_id, session_id, phone_number=req.phone_number, transport=req.transport, direction=req.direction, metadata=req.metadata)
    greeting = DEFAULT_GREETING
    return VoiceStartResponse(call_id=call_id, session_id=session_id, greeting=greeting, tts_text=build_tts_text(greeting))


@app.post("/voice/turn", response_model=VoiceTurnResponse)
def voice_turn(req: VoiceTurnRequest, db: Session = Depends(get_db)):
    call = _ensure_voice_call(db, req.call_id, req.session_id)
    if call.state != "active":
        raise HTTPException(status_code=409, detail="Call is not active")

    if not req.is_final:
        _persist_call_event(db, req.call_id, "partial_transcript", {"transcript": req.transcript})
        return VoiceTurnResponse(
            call_id=req.call_id,
            session_id=req.session_id,
            transcript=req.transcript,
            answer="",
            tts_text="",
            voice_answer="",
            citations=[],
            meta={"engine": "partial", "channel": req.channel},
        )

    db.add(VoiceCallTurn(call_id=req.call_id, session_id=req.session_id, role="user", transcript=req.transcript, meta_json={"channel": req.channel, "is_final": True}))
    db.commit()
    _persist_call_event(db, req.call_id, "user_turn", {"transcript": req.transcript})

    result = _process_text_request(AskRequest(session_id=req.session_id, question=req.transcript, top_k=req.top_k, mode=req.mode), db)
    tts_text = build_tts_text(result.get("voice_answer") or result.get("answer") or "")
    handoff = should_recommend_handoff(result)

    db.add(
        VoiceCallTurn(
            call_id=req.call_id,
            session_id=req.session_id,
            role="assistant",
            transcript=result.get("answer", ""),
            tts_text=tts_text,
            meta_json={"handoff_recommended": handoff, **(result.get("meta") or {})},
        )
    )
    db.commit()
    _persist_call_event(db, req.call_id, "assistant_turn", {"handoff_recommended": handoff, "tts_text": tts_text})

    return VoiceTurnResponse(
        call_id=req.call_id,
        session_id=req.session_id,
        transcript=req.transcript,
        answer=result.get("answer", ""),
        tts_text=tts_text,
        voice_answer=result.get("voice_answer") or tts_text,
        citations=result.get("citations", []),
        handoff_recommended=handoff,
        need_clarification=bool(result.get("need_clarification")),
        clarification_type=result.get("clarification_type"),
        meta=result.get("meta") or {},
    )


@app.post("/voice/handoff", response_model=VoiceHandoffResponse)
def voice_handoff(req: VoiceHandoffRequest, db: Session = Depends(get_db)):
    call = db.get(VoiceCallSession, req.call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    call.handoff_requested = True
    call.state = "handoff_requested"
    call.updated_at = datetime.now(timezone.utc)
    db.commit()
    _persist_call_event(db, req.call_id, "handoff_requested", {"reason": req.reason})
    return VoiceHandoffResponse(call_id=req.call_id, handoff_requested=True, message="Запрос на перевод к оператору зарегистрирован.")
