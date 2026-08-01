"""A minimal `WebSocket` double that drives a real `DialogueSession` without
a real network socket.

This is deliberately NOT a mock of anything `backend/ws/session.py` needs to
be correct about -- STT/TTS/LLM/RAG are all real (`session_deps` in
`conftest.py`). It stands in only for the transport (`starlette`'s
`WebSocket`), so tests can push inbound frames and inspect outbound ones
directly and fast, alongside precise grey-box access to `DialogueSession`'s
own state (`._memory`, `._automaton`) for checks the wire protocol alone
can't observe (e.g. "this draft never reached `DialogueMemory.turns`",
FR-12/A-05) -- the same grey-box style `tests/unit/test_state_machine.py`
already uses for `AutomatonState`/`Draft`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from starlette.websockets import WebSocketState


@dataclass
class FakeWebSocket:
    client_state: WebSocketState = WebSocketState.CONNECTING
    inbound: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    sent_json: list[dict[str, Any]] = field(default_factory=list)
    sent_bytes: list[bytes] = field(default_factory=list)
    _sent_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def accept(self) -> None:
        self.client_state = WebSocketState.CONNECTED

    async def receive(self) -> dict[str, Any]:
        return await self.inbound.get()

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent_json.append(payload)
        self._sent_event.set()
        self._sent_event.clear()

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self) -> None:
        self.client_state = WebSocketState.DISCONNECTED

    # -- test-side helpers --------------------------------------------

    def push_bytes(self, frame: bytes) -> None:
        self.inbound.put_nowait({"type": "websocket.receive", "bytes": frame, "text": None})

    def push_text(self, text: str) -> None:
        self.inbound.put_nowait({"type": "websocket.receive", "bytes": None, "text": text})

    def push_disconnect(self) -> None:
        self.inbound.put_nowait({"type": "websocket.disconnect"})

    async def wait_for_json(
        self, predicate, *, timeout_s: float = 15.0
    ) -> dict[str, Any]:
        """Polls `sent_json` for the first (new) message matching
        `predicate`, waiting up to `timeout_s` real seconds. Used for
        assertions that depend on real LLM/RAG/TTS latency, not on the
        session's simulated audio-offset clock.
        """
        seen = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            for msg in self.sent_json[seen:]:
                if predicate(msg):
                    return msg
            seen = len(self.sent_json)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"no message matched {predicate!r} within {timeout_s}s; got {self.sent_json}")
            try:
                await asyncio.wait_for(self._sent_event.wait(), timeout=min(remaining, 0.5))
            except TimeoutError:
                pass
