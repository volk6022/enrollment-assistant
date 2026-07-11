from __future__ import annotations

import re

from .base import BaseExtractor, RetrievalPacket, flatten_text


class WithoutEGEExtractor(BaseExtractor):
    intent = "without_ege"
    schema = {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "allowed": {"type": "boolean"},
                        "condition": {"type": "string"},
                    },
                    "required": ["category", "allowed"],
                    "additionalProperties": False,
                },
            },
            "source_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["cases"],
        "additionalProperties": False,
    }

    def _fallback(self, packet: RetrievalPacket) -> dict:
        text = flatten_text(packet)
        cases = []
        for line in text.splitlines():
            if re.search(r"внутренн(?:ие|их)\s+вступительн", line, re.I) or re.search(r"без\s+егэ", line, re.I):
                cases.append({"category": line.strip()[:180], "allowed": True})
        return {"cases": cases[:10], "source_points": self._source_points(packet)}
