"""Structured-output schemas for LlamaClient.decide() (contracts/llm.md §4).

Each schema pairs a Pydantic model — typed access to a parsed decision — with
the raw JSON-schema dict that goes into the `json_schema` field of the
`/completion` request. The dict is what actually constrains the grammar and
must match the contract verbatim; the model is what the rest of the codebase
reads. The two describing different shapes would be a silent bug, not a
style choice.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class InterjectDecision(BaseModel):
    """FR-07: собеседник говорит непрерывно 20 с — вмешаться или слушать дальше."""

    interject: bool
    understood: str = Field(description="что агент понял из речи на данный момент")
    reason: str

    SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "interject": {"type": "boolean"},
            "understood": {
                "type": "string",
                "description": "что агент понял из речи на данный момент",
            },
            "reason": {"type": "string"},
        },
        "required": ["interject", "understood", "reason"],
    }


class BargeInDecision(BaseModel):
    """FR-13: собеседник говорит поверх ответа агента — уступить или договорить."""

    interrupt: bool
    reason: str

    SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "interrupt": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["interrupt", "reason"],
    }


class IntentDecision(BaseModel):
    """FR-21/FR-22: единственный источник intent/confidence для scenarios.yaml."""

    intent: Literal["question", "clarification_needed", "goodbye", "smalltalk", "unclear"]
    confidence: float = Field(ge=0.0, le=1.0)
    query: str = Field(
        default="",
        description="переформулированный запрос для RAG; пусто, если RAG не нужен",
    )
    reason: str

    SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "intent": {
                "enum": [
                    "question",
                    "clarification_needed",
                    "goodbye",
                    "smalltalk",
                    "unclear",
                ],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "query": {
                "type": "string",
                "description": "переформулированный запрос для RAG; пусто, если RAG не нужен",
            },
            "reason": {"type": "string"},
        },
        "required": ["intent", "confidence", "reason"],
    }
