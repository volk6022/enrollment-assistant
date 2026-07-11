from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

try:
    from redis import Redis
except Exception:  # pragma: no cover
    Redis = None

from .config import settings


@dataclass
class CallSessionState:
    call_id: str
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_transcript: str = ""
    last_answer: str = ""
    tts_path: str = ""


class CallSessionStore:
    def __init__(self) -> None:
        self._mem: Dict[str, CallSessionState] = {}
        self._redis = None
        if Redis is not None:
            try:
                self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def get(self, call_id: str) -> Optional[CallSessionState]:
        if self._redis is not None:
            data = self._redis.hgetall(f"call:{call_id}")
            if data:
                return CallSessionState(**data)
        return self._mem.get(call_id)

    def put(self, state: CallSessionState) -> CallSessionState:
        if self._redis is not None:
            self._redis.hset(f"call:{state.call_id}", mapping={k: str(v) for k, v in asdict(state).items()})
            self._redis.expire(f"call:{state.call_id}", 3600)
        self._mem[state.call_id] = state
        return state


store = CallSessionStore()
