"""Структурное логирование (JSON, LOG_LEVEL/LOG_FORMAT/LOG_FILE) + `/metrics`
counters -- T-12.

`configure_logging`/`get_logger` are the T-01 scaffold every module already
uses. `Metrics` is this task's addition: a plain in-process counter bag for
`GET /metrics` (`backend/app.py`), fed by `backend/ws/session.py` at exactly
the points that already produce the JSON log lines this module's docstring
used to defer -- state transitions (`_on_transition`), LLM decisions
(`_RecordingDecisionClient.decide`), generation/RAG calls. One instance lives
on `app.state.metrics`, shared (read *and* written) across every
`DialogueSession` via `SessionDependencies.metrics` -- deliberately not
per-session, since `/metrics` is a process-wide endpoint and NFR-06 already
commits this whole stack to one process, so there is no multi-process
aggregation problem to solve here.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from dataclasses import dataclass, field

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


@dataclass
class Metrics:
    """Process-wide counters rendered by `GET /metrics` (Prometheus text
    exposition format, contracts/websocket.md §5). Mutated only from the
    asyncio event loop -- same "state mutates on the loop" rule
    `backend/ws/session.py`'s module docstring states for everything else in
    this project, so no lock is needed despite being shared across every
    concurrent `DialogueSession` (plan.md's "Вне области": one session at a
    time in production anyway, but nothing here assumes that to stay
    correct).

    Deliberately separate from `backend/app.py`'s own
    `ws_connections_total`/`ws_connections_active` (T-01, kept as-is on
    `app.state` directly) -- this class only owns what T-12 adds.
    """

    state_transitions_total: Counter[tuple[str, str]] = field(default_factory=Counter)  # (prev, current)
    decisions_total: Counter[tuple[str, str]] = field(default_factory=Counter)  # (kind, "true"/"false")
    llm_stream_calls_total: int = 0
    llm_decide_calls_total: int = 0
    llm_prompt_tokens_total: int = 0
    llm_cached_tokens_total: int = 0
    llm_predicted_tokens_total: int = 0
    rag_queries_total: int = 0
    stt_partial_calls_total: int = 0
    tts_synth_calls_total: int = 0
    errors_total: Counter[str] = field(default_factory=Counter)

    def record_transition(self, prev: str, current: str) -> None:
        self.state_transitions_total[(prev, current)] += 1

    def record_decision(self, kind: str, result: bool) -> None:
        self.decisions_total[(kind, "true" if result else "false")] += 1

    def record_error(self, kind: str) -> None:
        self.errors_total[kind] += 1

    def render_prometheus(self) -> str:
        lines: list[str] = [
            "# HELP backend_state_transitions_total Automaton transitions (dialogue.qnt AgentState), by prev/current.",
            "# TYPE backend_state_transitions_total counter",
        ]
        for (prev, current), count in sorted(self.state_transitions_total.items()):
            lines.append(f'backend_state_transitions_total{{prev="{prev}",current="{current}"}} {count}')

        lines += [
            "# HELP backend_decisions_total LLM decisions (interject/barge_in, contracts/llm.md §4), by kind/result.",
            "# TYPE backend_decisions_total counter",
        ]
        for (kind, result), count in sorted(self.decisions_total.items()):
            lines.append(f'backend_decisions_total{{kind="{kind}",result="{result}"}} {count}')

        lines += [
            "# HELP backend_llm_stream_calls_total stream_answer() calls (contracts/llm.md §2).",
            "# TYPE backend_llm_stream_calls_total counter",
            f"backend_llm_stream_calls_total {self.llm_stream_calls_total}",
            "# HELP backend_llm_decide_calls_total decide() calls (contracts/llm.md §3).",
            "# TYPE backend_llm_decide_calls_total counter",
            f"backend_llm_decide_calls_total {self.llm_decide_calls_total}",
            "# HELP backend_llm_prompt_tokens_total Sum of prompt_n (re-evaluated prefix) across all llama-server calls.",
            "# TYPE backend_llm_prompt_tokens_total counter",
            f"backend_llm_prompt_tokens_total {self.llm_prompt_tokens_total}",
            "# HELP backend_llm_cached_tokens_total Sum of cache_n (reused prefix) across all llama-server calls -- "
            "llm.md §6: this is what cache_prompt ACTUALLY reused, not just that the flag was set.",
            "# TYPE backend_llm_cached_tokens_total counter",
            f"backend_llm_cached_tokens_total {self.llm_cached_tokens_total}",
            "# HELP backend_llm_predicted_tokens_total Sum of predicted_n (generated tokens) across all llama-server calls.",
            "# TYPE backend_llm_predicted_tokens_total counter",
            f"backend_llm_predicted_tokens_total {self.llm_predicted_tokens_total}",
            "# HELP backend_rag_queries_total RAG_QUERY scenario actions executed (RagPipeline.asearch calls).",
            "# TYPE backend_rag_queries_total counter",
            f"backend_rag_queries_total {self.rag_queries_total}",
            "# HELP backend_stt_partial_calls_total Completed WhisperWorker.try_partial() decodes.",
            "# TYPE backend_stt_partial_calls_total counter",
            f"backend_stt_partial_calls_total {self.stt_partial_calls_total}",
            "# HELP backend_tts_synth_calls_total SileroWorker.synthesize() calls.",
            "# TYPE backend_tts_synth_calls_total counter",
            f"backend_tts_synth_calls_total {self.tts_synth_calls_total}",
        ]

        lines += [
            "# HELP backend_errors_total Recoverable errors caught by DialogueSession, by kind.",
            "# TYPE backend_errors_total counter",
        ]
        for kind, count in sorted(self.errors_total.items()):
            lines.append(f'backend_errors_total{{kind="{kind}"}} {count}')

        return "\n".join(lines) + "\n"
