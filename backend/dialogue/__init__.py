"""Dialogue layer: automaton state shapes and the interleaved history buffer.

Public surface this package exposes so far (T-06; `machine.py`/`nodes.py`/
`scenarios.py` land with T-07/T-08 and extend this list):

    from backend.dialogue import AgentState, Draft, DialogueTurn
    from backend.dialogue import DialogueTimers, DialogueState, RagResult
    from backend.dialogue import DialogueMemory

Nothing in this package reads `.env` directly (FR-32) -- callers pass
thresholds (`transcript_buffer_chars`, `dialogue_history_max_turns`, ...)
explicitly, sourced from `backend/config.py` (T-01).
"""
from __future__ import annotations

from backend.dialogue.memory import DialogueMemory
from backend.dialogue.models import (
    AgentState,
    DialogueState,
    DialogueTimers,
    DialogueTurn,
    Draft,
    RagResult,
)

__all__ = [
    "AgentState",
    "Draft",
    "DialogueTurn",
    "DialogueTimers",
    "DialogueState",
    "RagResult",
    "DialogueMemory",
]
