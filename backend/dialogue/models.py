"""Shared vocabulary for the dialogue layer: automaton states, draft
lifecycle, one turn of history, and the per-session state T-07's automaton
mutates.

`AgentState` and `Draft` are the Python mirror of `specs/formal/dialogue.qnt`
(plan.md §9 "Соответствие модели и кода"): same eight states, same five
draft values. Member *values* (not just names) are the exact identifiers
used in the `.qnt` source (`Greeting`, `DecidingBargeIn`, ...) so that
telemetry (plan.md §12: "каждый переход автомата" logged as JSON) and the
final cross-check ("действие в `.qnt` → ребро в графе", tasks.md "Финальная
проверка" §2) can compare log output against the model text directly,
without a second translation table someone has to keep in sync by hand.

`DialogueTurn` is `contracts/memory.md` §4's dataclass, verbatim. It is
deliberately frozen: once a turn is committed (memory.py, §5) nothing about
it should change again -- there is no code path in the contract that edits
a turn after the fact, only ones that add new ones.

`DialogueState` and `DialogueTimers` are this module's own design (plan.md
§9 lists them for `dialogue/models.py` but neither is spelled out field-by-
field anywhere) -- kept to exactly what plan.md §9's correspondence table
states explicitly (`AgentState`/`Draft`/timers), not a guess at what T-07's
automaton or T-09's orchestration will additionally want to hang off of it.
Extending this dataclass later is cheap; guessing wrong fields now and
having T-07 work around them is not.

`RagResult` wraps `RagPipeline.search()`'s `(list[dict], dict)` return
(backend/rag/pipeline.py) in a named type for dialogue code (T-08/T-09) to
pass around when assembling the "RAG-контекст" prompt block (llm.md §6).
It is intentionally NOT part of `DialogueMemory` (memory.py) -- chunk
content must never be written into the persisted transcript buffer (FR-20,
memory.md §6), so this type only ever exists as a short-lived value on the
way from `RagPipeline` to the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class AgentState(Enum):
    """The eight states of `dialogue.qnt`'s `AgentState`, same identifiers."""

    GREETING = "Greeting"
    LISTENING = "Listening"
    DECIDING_INTERJECT = "DecidingInterject"
    FORMULATING = "Formulating"
    SPEAKING = "Speaking"
    DECIDING_BARGE_IN = "DecidingBargeIn"
    CLOSING = "Closing"
    ENDED = "Ended"


class Draft(Enum):
    """The five values of `dialogue.qnt`'s `Draft`, same identifiers.

    `DROPPED` and `COMMITTED` are terminal for a given candidate reply and
    exist so the invariant the whole memory contract hinges on --
    `inv_dropped_never_committed` -- can be stated at all: a draft that
    never got past `BUILDING` before the user kept talking must become
    `DROPPED`, never `COMMITTED` (FR-12, memory.md §5).
    """

    NO_DRAFT = "NoDraft"
    BUILDING = "Building"
    VOICING = "Voicing"
    DROPPED = "Dropped"
    COMMITTED = "Committed"


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """One entry in `DialogueMemory.turns` (`contracts/memory.md` §4,
    verbatim field set). Frozen: turns are appended, never edited in place.

    `voiced_fraction`/`is_partial` are agent-only; `stt_confidence` is
    user-only -- `__post_init__` enforces that instead of leaving it as an
    unchecked convention, since a value silently set on the wrong role is
    exactly the kind of mistake that produces a plausible-looking but wrong
    dialogue history (memory.md §7's whole point).
    """

    role: Literal["user", "agent"]
    text: str
    start_ms: int
    end_ms: int
    voiced_fraction: float | None = None
    is_partial: bool = False
    stt_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.role not in ("user", "agent"):
            raise ValueError(f"role must be 'user' or 'agent', got {self.role!r}")
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) precedes start_ms ({self.start_ms}) -- "
                "overlap between turns is allowed, but a turn running backwards "
                "in time is not (memory.md §4/§7)"
            )
        if self.role == "user":
            if self.voiced_fraction is not None or self.is_partial:
                raise ValueError(
                    "voiced_fraction/is_partial are agent-only fields "
                    "(memory.md §4) but this turn has role='user'"
                )
        elif self.stt_confidence is not None:
            raise ValueError(
                "stt_confidence is a user-only field (memory.md §4) but "
                "this turn has role='agent'"
            )


@dataclass
class DialogueTimers:
    """Real millisecond counters for the quantities `dialogue.qnt` tracks as
    small bounded ordinals (`turnLen`/`idleLen`/`overlapLen`/`speechLeft`) --
    the model abstracts them so Apalache can check exhaustively; production
    code needs the actual durations those ordinals stand for. `.env` holds
    the thresholds compared against these (`DIALOGUE_INTERJECT_AFTER_S` etc,
    plan.md §9's correspondence table) -- this dataclass only holds the
    running counters themselves, never a threshold.
    """

    turn_ms: int = 0
    idle_ms: int = 0
    overlap_ms: int = 0
    speech_left_ms: int = 0


@dataclass
class DialogueState:
    """Per-session automaton state: exactly the `(agent, draft, timers)`
    triple plan.md §9's correspondence table maps to `dialogue.qnt`.

    Deliberately does NOT carry `userSpeaking` (already owned by
    `audio.vad.VadGate`, which is the only thing that can observe it) or
    `committed`/`dropped` counters (`dialogue.qnt` needs them only so its
    invariants can talk about counts; production code can derive the same
    counts from `DialogueMemory.turns` whenever something -- telemetry, a
    test -- actually needs them, rather than maintaining a second copy that
    could drift out of sync with the memory it's counting).
    """

    agent: AgentState = AgentState.GREETING
    draft: Draft = Draft.NO_DRAFT
    timers: DialogueTimers = field(default_factory=DialogueTimers)


@dataclass(frozen=True, slots=True)
class RagResult:
    """One RAG query's outcome (`RagPipeline.search()`'s return value, named
    instead of positional). `chunks` carries whatever fields
    `backend/rag/retrieve.py`/`rerank.py` attached (`rerank_score`,
    `rrf_score`, chunk `text`, source metadata) -- this type doesn't
    constrain that shape further, it only gives dialogue code a stable name
    to pass around instead of an untyped tuple.

    Never stored on `DialogueMemory`: chunk content is explicitly excluded
    from the persisted transcript buffer (FR-20, memory.md §6, "содержимое
    RAG-чанков в буфере не хранится") because it is redundant with the RAG
    context block already in the prompt and would crowd out real history.
    A `RagResult` lives only as long as it takes to build one prompt.
    """

    query: str
    chunks: tuple[dict[str, object], ...]
    timings_ms: dict[str, float]

    @property
    def has_results(self) -> bool:
        """False when nothing cleared `RAG_MIN_SCORE` -- the caller's cue
        for the honest "не нашёл" answer (A-11) instead of inventing one."""
        return len(self.chunks) > 0


__all__ = [
    "AgentState",
    "Draft",
    "DialogueTurn",
    "DialogueTimers",
    "DialogueState",
    "RagResult",
]
