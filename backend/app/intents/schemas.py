from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntentContext:
    raw_question: str
    intent: str
    profile: str
    level: Optional[str] = None
    study_form: Optional[str] = None
    status: Optional[str] = None
    needs_optional_clarification: bool = False
    rewritten_query: str = ""
    notes: list[str] = field(default_factory=list)
