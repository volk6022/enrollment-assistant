from __future__ import annotations

from .base import BaseExtractor, RetrievalPacket, flatten_text


class ApplyExtractor(BaseExtractor):
    intent = "apply"
    schema = {
        "type": "object",
        "properties": {
            "methods": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "allowed": {"type": "boolean"},
                        "condition": {"type": "string"},
                    },
                    "required": ["name", "allowed"],
                    "additionalProperties": False,
                },
            },
            "source_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["methods"],
        "additionalProperties": False,
    }

    def _fallback(self, packet: RetrievalPacket) -> dict:
        text = flatten_text(packet).lower()
        methods = []
        for token, name in [("лично", "лично"), ("почт", "почтой"), ("электрон", "в электронной форме"), ("госуслуг", "через госуслуги")]:
            if token in text:
                methods.append({"name": name, "allowed": True})
        return {"methods": methods, "source_points": self._source_points(packet)}
