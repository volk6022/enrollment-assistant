from __future__ import annotations

from collections import Counter


class Metrics:
    def __init__(self) -> None:
        self.counter = Counter()

    def inc(self, name: str) -> None:
        self.counter[name] += 1

    def snapshot(self) -> dict:
        return dict(self.counter)


metrics = Metrics()
