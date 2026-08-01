"""Session-wide time axis.

Every timestamp in the system -- audio ring positions, VAD events, dialogue
turn boundaries -- is a millisecond offset from a single `t0`
(`contracts/memory.md` §1). That is what makes it possible to sort
interleaved user/agent turns correctly (FR-18): both sides' timings land on
the same axis instead of two clocks that have to be reconciled after the
fact.

`time.monotonic()`, never `time.time()`: an NTP step or a DST change must not
retroactively rewrite history that already happened.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SessionClock:
    """`t0` is captured once, at session start. `now_ms()` is always relative
    to that instant, monotonic, and never goes backwards.
    """

    t0: float = field(default_factory=time.monotonic)

    def now_ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)

    def ms_to_monotonic(self, ms: int) -> float:
        """Inverse of `now_ms()`: turns a session-relative offset back into a
        `time.monotonic()` value. Needed wherever a timer has to be armed
        against an absolute deadline (e.g. `asyncio` `call_at`) instead of a
        relative delay.
        """
        return self.t0 + ms / 1000
