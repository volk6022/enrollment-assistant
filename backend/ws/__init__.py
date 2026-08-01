"""WebSocket dialogue orchestration (T-09): `/ws/dialogue`'s real implementation.

Public surface:

    from backend.ws.session import DialogueSession, SessionDependencies, ensure_greeting_audio

`backend/app.py` builds one `SessionDependencies` (process-level singletons:
`LlamaClient`, `WhisperWorker`, `SileroWorker`, `RagPipeline`, the "rag" pool,
`ScenarioRegistry`, `DialogueThresholds`) in its lifespan, then constructs one
`DialogueSession` per accepted `/ws/dialogue` connection.
"""
from __future__ import annotations

from backend.ws.session import DialogueSession, SessionDependencies, ensure_greeting_audio

__all__ = ["DialogueSession", "SessionDependencies", "ensure_greeting_audio"]
