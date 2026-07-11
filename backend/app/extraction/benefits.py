from __future__ import annotations

import re

from .base import BaseExtractor, RetrievalPacket, flatten_text


class BenefitsExtractor(BaseExtractor):
    intent = "benefits"
    schema = {
        "type": "object",
        "properties": {
            "benefits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "condition": {"type": "string"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            "source_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["benefits"],
        "additionalProperties": False,
    }

    def _fallback(self, packet: RetrievalPacket) -> dict:
        items = []
        for line in flatten_text(packet).splitlines():
            if re.search(r"льгот|квот|особ[а-я]* право|преимуществен", line, re.I):
                items.append({"name": line.strip()[:180]})
        return {"benefits": items[:12], "source_points": self._source_points(packet)}
