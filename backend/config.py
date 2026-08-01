"""Единственное место в проекте, где читается `.env` (FR-32).

Все переменные из `.env.example` собраны здесь, типизированы и сгруппированы во
вложенные модели (`LLMSettings`, `AudioSettings`, `DialogueSettings`, `RagSettings`,
`ApiSettings`, `LogSettings`) — так их удобно прокидывать в компоненты по отдельности,
не таская один гигантский объект. Остальной код обязан получать значения только
через `backend.config.settings`; прямые обращения к `os.environ`/`os.getenv` в любом
другом модуле — нарушение FR-32 (проверяется в «Финальной проверке» tasks.md, пункт 6).

Никаких значений по умолчанию для порогов/портов/путей: `.env.example` — не
подсказка, а обязательный к заполнению список (FR-32, «Хардкода порогов, портов и
путей в коде нет»). Пустой или отсутствующий `.env` обязан провалиться явным,
читаемым сообщением со списком недостающих переменных, а не `KeyError`/трейсом
pydantic — это и есть приёмка T-01.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _REPO_ROOT / ".env"


class ConfigurationError(RuntimeError):
    """`.env` отсутствует, неполон или содержит невалидные значения.

    Сообщение уже отформатировано для показа человеку — `str(err)` печатается
    как есть, без дополнительного форматирования вызывающей стороной.
    """


class _GroupSettings(BaseSettings):
    """Общая база для всех групп: один и тот же `.env`, без опечаток в незнакомых ключах."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def _blank_to_none(value: object) -> object:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class LLMSettings(_GroupSettings):
    """Слой llama-server: три эндпоинта (LLM, эмбеддер, реранкер) + параметры генерации.

    llama-server поднимается на хосте (`scripts/serve-models.ps1|sh`), не в docker —
    решение заказчика (`plan.md` §1). Отсюда backend знает только эндпоинты и файлы
    моделей для справки/валидации на старте, сам процесс не запускает.
    """

    llm_endpoint: str
    embedding_endpoint: str
    reranker_endpoint: str

    llm_n_parallel: int = Field(ge=1)
    llm_context_size: int = Field(gt=0)
    llm_max_tokens: int = Field(gt=0)
    llm_temperature: float = Field(ge=0.0, le=2.0)
    llm_decision_temperature: float = Field(ge=0.0, le=2.0)
    llm_timeout_s: float = Field(gt=0.0)

    models_dir: Path
    llm_model_file: str
    embedding_model_file: str
    reranker_model_file: str

    # Моки для unit-тестов; контрактные тесты (T-03) требуют false — они бьют
    # по живому llama-server намеренно (см. contracts/llm.md §2, HTTP 400 на
    # "правильный по документации" вызов ловится только реальным запросом).
    mock_llm: bool = False

    @field_validator("llm_endpoint", "embedding_endpoint", "reranker_endpoint")
    @classmethod
    def _must_be_http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("должен начинаться с http:// или https://")
        return value.rstrip("/")


class AudioSettings(_GroupSettings):
    """Захват аудио, VAD-преролл, whisper (STT) и silero (TTS)."""

    audio_sample_rate: int = Field(gt=0)
    audio_channels: int
    audio_chunk_size_ms: int = Field(gt=0)
    audio_ring_seconds: int = Field(gt=0)

    vad_preroll_ms: int = Field(ge=0)
    vad_silence_end_ms: int = Field(gt=0)
    vad_threshold: float = Field(ge=0.0, le=1.0)

    stt_model: str
    stt_device: Literal["cuda", "cpu"]
    stt_compute_type: str
    stt_partial_max_hz: float = Field(gt=0.0)
    stt_window_seconds: float = Field(gt=0.0)
    stt_overlap_seconds: float = Field(ge=0.0)
    stt_final_pass: bool
    stt_language: str

    tts_model: str
    tts_device: Literal["cuda", "cpu"]
    tts_speaker: str
    tts_sample_rate: int = Field(gt=0)

    @field_validator("audio_channels")
    @classmethod
    def _mono_only(cls, value: int) -> int:
        # Кольцо, VAD и whisper во всей системе считают аудио моно (plan.md §3.1).
        if value != 1:
            raise ValueError("система рассчитана только на моно (AUDIO_CHANNELS=1)")
        return value


class DialogueSettings(_GroupSettings):
    """Пороги автомата — секундные аналоги TALK_LIMIT/IDLE_LIMIT/OVERLAP_LIMIT/TAIL_MIN
    из `specs/formal/dialogue.qnt` (порядковые в модели, реальные секунды — только тут,
    см. `plan.md` §9 «Соответствие модели и кода»).
    """

    dialogue_interject_after_s: float = Field(gt=0.0)
    dialogue_idle_hangup_s: float = Field(gt=0.0)
    dialogue_barge_in_overlap_s: float = Field(gt=0.0)
    dialogue_barge_in_min_tail_s: float = Field(ge=0.0)
    # Порога уточнения здесь намеренно нет: он живёт литералом в `when:` сценария
    # `clarify_unclear` (dialogue/scenarios.yaml), рядом со своим `priority`.
    # Держать его ещё и тут значило бы иметь две правды об одном числе.

    transcript_buffer_chars: int = Field(gt=0)
    dialogue_history_max_turns: int = Field(gt=0)

    scenarios_path: Path
    greeting_audio_path: Path

    # Разработка: логировать каждый переход автомата / каталог для дампов сессий.
    debug_state_machine: bool = False
    debug_save_dialogues: str | None = None

    @field_validator("debug_save_dialogues", mode="before")
    @classmethod
    def _normalize_save_dir(cls, value: object) -> object:
        return _blank_to_none(value)


class RagSettings(_GroupSettings):
    """`RAG_TOP_K=10` и соседние параметры — сетка 1200 тестов, см. `plan.md` §6."""

    rag_top_k: int = Field(gt=0)
    rag_fused_top: int = Field(gt=0)
    rag_dense_top: int = Field(gt=0)
    rag_bm25_top: int = Field(gt=0)
    rag_rrf_k: int = Field(gt=0)
    rag_min_score: float = Field(ge=0.0, le=1.0)
    rag_max_length: int = Field(gt=0)
    rag_batch_size: int = Field(gt=0)

    faiss_index_path: Path
    kb_source_path: Path

    @model_validator(mode="after")
    def _top_k_within_pool(self) -> "RagSettings":
        if self.rag_top_k > self.rag_fused_top:
            raise ValueError(
                f"RAG_TOP_K ({self.rag_top_k}) не может превышать "
                f"RAG_FUSED_TOP ({self.rag_fused_top}) — финальному отбору "
                "неоткуда брать лишние чанки"
            )
        return self


class ApiSettings(_GroupSettings):
    """Сеть: адрес backend, вебсокет-пинги, CORS. `GUI_PORT` читается тут же —
    отдельной группы под GUI нет, а порт GUI такой же сетевой параметр топологии
    (используется docker-compose/T-11, не самим backend-процессом).
    """

    api_host: str
    api_port: int = Field(ge=1, le=65535)
    ws_ping_interval_s: float = Field(gt=0.0)
    ws_ping_timeout_s: float = Field(gt=0.0)
    cors_origins: str
    gui_port: int = Field(ge=1, le=65535)

    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _ping_timeout_after_interval(self) -> "ApiSettings":
        if self.ws_ping_timeout_s <= self.ws_ping_interval_s:
            raise ValueError(
                "WS_PING_TIMEOUT_S должен быть больше WS_PING_INTERVAL_S, иначе "
                "сессия обрывается раньше первого же пинга"
            )
        return self


class LogSettings(_GroupSettings):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_format: Literal["json", "text"]
    log_file: Path
    metrics_enabled: bool
    telemetry_endpoint: str | None = None

    @field_validator("telemetry_endpoint", mode="before")
    @classmethod
    def _normalize_endpoint(cls, value: object) -> object:
        return _blank_to_none(value)


@dataclass(frozen=True, slots=True)
class Settings:
    """Композиция всех групп — единственная точка правды для остального кода."""

    llm: LLMSettings
    audio: AudioSettings
    dialogue: DialogueSettings
    rag: RagSettings
    api: ApiSettings
    log: LogSettings


_GROUPS: tuple[tuple[str, type[_GroupSettings]], ...] = (
    ("LLM / llama-server", LLMSettings),
    ("Аудио / VAD / STT / TTS", AudioSettings),
    ("Диалог", DialogueSettings),
    ("RAG", RagSettings),
    ("API", ApiSettings),
    ("Логи и телеметрия", LogSettings),
)


def _describe_error(error: dict) -> str:
    var_name = str(error["loc"][0]).upper()
    if error["type"] == "missing":
        return f"    - {var_name}: переменная не задана"
    return f"    - {var_name}: {error['msg']}"


def _load_settings() -> Settings:
    built: dict[str, _GroupSettings] = {}
    problems: list[str] = []

    for label, group_cls in _GROUPS:
        try:
            built[group_cls.__name__] = group_cls()
        except ValidationError as exc:
            problems.append(f"  [{label}]")
            problems.extend(_describe_error(err) for err in exc.errors())

    if problems:
        env_hint = _ENV_FILE if _ENV_FILE.exists() else f"{_ENV_FILE} (файл не найден)"
        message = (
            "Конфигурация не загружена — в .env отсутствуют или невалидны "
            "переменные окружения:\n"
            + "\n".join(problems)
            + f"\n\nОжидаемый файл: {env_hint}\n"
            "Исправление: cp .env.example .env, затем заполнить недостающие "
            "значения (см. комментарии в .env.example)."
        )
        raise ConfigurationError(message)

    return Settings(
        llm=built["LLMSettings"],  # type: ignore[arg-type]
        audio=built["AudioSettings"],  # type: ignore[arg-type]
        dialogue=built["DialogueSettings"],  # type: ignore[arg-type]
        rag=built["RagSettings"],  # type: ignore[arg-type]
        api=built["ApiSettings"],  # type: ignore[arg-type]
        log=built["LogSettings"],  # type: ignore[arg-type]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Ленивая точка входа для тестов: `get_settings.cache_clear()` между кейсами
    с разными `.env`. Модульный singleton `settings` ниже вызывает её один раз при
    импорте — это то, что видит остальной код через `from backend.config import settings`.
    """
    return _load_settings()


try:
    settings: Settings = get_settings()
except ConfigurationError as exc:
    # sys.exit с текстом печатает только сообщение и завершает процесс кодом 1 —
    # без стектрейса импорта, который тут был бы бесполезен и всё равно скрыл бы
    # причину под сотней строк pydantic-internals.
    sys.exit(str(exc))
