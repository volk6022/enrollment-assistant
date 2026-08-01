"""Каркас структурного логирования — JSON, настраивается по LOG_LEVEL/LOG_FORMAT/LOG_FILE.

Это только логгер. Полноценная телеметрия — тайминги по компонентам, решения LLM
с `reason`, переходы автомата, блок `decisions` из `contracts/websocket.md` §3.6 —
задача T-12. Здесь — общая точка настройки и `get_logger()`, которым T-12 и все
остальные модули пользуются вместо `logging.getLogger(__name__)` напрямую, чтобы
формат был единым по всему процессу с первого дня.
"""
from __future__ import annotations

import logging
import sys

import structlog

from backend.config import LogSettings


def configure_logging(log: LogSettings) -> None:
    """Настраивает stdlib `logging` + `structlog` по конфигу. Вызывается ровно один
    раз, из lifespan `backend/app.py`, до создания любого другого логгера.
    """
    level = getattr(logging, log.log_level)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log.log_file is not None:
        log.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log.log_file, encoding="utf-8"))

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if log.log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    for handler in handlers:
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(level)

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Именованный структурный логгер. Если `configure_logging` ещё не вызывалась
    (например, в юнит-тестах отдельного модуля), structlog отдаёт логгер с разумными
    дефолтами — не падает, просто без файлового хэндлера и настроенного уровня.

    Тип возврата — `FilteringBoundLogger`: именно его отдаёт `make_filtering_bound_logger`
    в `configure_logging`, и на нём же (с 22.1) есть асинхронные методы `ainfo`/`adebug`/…,
    которыми пользуется остальной код (`await logger.ainfo(...)`).
    """
    return structlog.get_logger(name)
