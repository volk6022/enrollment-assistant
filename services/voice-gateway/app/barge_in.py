from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaybackState:
    is_playing: bool = False

    def start(self) -> None:
        self.is_playing = True

    def stop(self) -> None:
        self.is_playing = False

    def should_interrupt(self, user_started_speaking: bool) -> bool:
        return self.is_playing and user_started_speaking
