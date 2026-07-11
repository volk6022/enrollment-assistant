from __future__ import annotations

import re

from .base import BaseExtractor, RetrievalPacket, flatten_text


class DocumentsExtractor(BaseExtractor):
    intent = "documents"
    schema = {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "required": {"type": "boolean"},
                        "condition": {"type": "string"},
                    },
                    "required": ["name", "required"],
                    "additionalProperties": False,
                },
            },
            "source_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["documents"],
        "additionalProperties": False,
    }

    def _fallback(self, packet: RetrievalPacket) -> dict:
        docs = []
        for line in flatten_text(packet).splitlines():
            if re.match(r"^\s*(?:\d+[\).]|[-•])\s+", line):
                item = re.sub(r"^\s*(?:\d+[\).]|[-•])\s+", "", line).strip(" ;.")
                if 4 < len(item) < 160:
                    docs.append({"name": item, "required": True})
        return {"documents": docs[:20], "source_points": self._source_points(packet)}
