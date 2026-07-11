from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TranscriptBuffer:
    partials: List[str] = field(default_factory=list)

    def push_partial(self, text: str) -> None:
        if text:
            self.partials.append(text)

    def finalize(self, final_text: str) -> str:
        text = final_text.strip() if final_text else " ".join(self.partials).strip()
        self.partials.clear()
        return text
