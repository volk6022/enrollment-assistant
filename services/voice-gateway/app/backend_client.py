from __future__ import annotations

import asyncio

import httpx

from .config import settings


class BackendClient:
    def __init__(self) -> None:
        self.base_url = settings.backend_url.rstrip("/")
        self.retries = settings.backend_retries
        self.retry_delay_sec = settings.backend_retry_delay_sec

    async def _post(self, path: str, payload: dict, timeout: float) -> dict:
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.post(f"{self.base_url}{path}", json=payload)
                    r.raise_for_status()
                    return r.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(self.retry_delay_sec)
        raise last_exc  # pragma: no cover

    async def start_call(self, payload: dict) -> dict:
        return await self._post("/voice/start", payload, timeout=60.0)

    async def send_turn(self, payload: dict) -> dict:
        return await self._post("/voice/turn", payload, timeout=120.0)

    async def request_handoff(self, payload: dict) -> dict:
        return await self._post("/voice/handoff", payload, timeout=30.0)
